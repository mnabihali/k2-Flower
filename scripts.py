import argparse
import copy
import logging
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union

import k2
import optim
import sentencepiece as spm
import torch
import torch.multiprocessing as mp
from asr_datamodule import TedLiumAsrDataModule
from conformer import Conformer
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from local.convert_transcript_words_to_bpe_ids import convert_texts_into_ids
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from icefall import diagnostics
from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    display_and_save_batch,
    encode_supervisions,
    setup_logger,
    str2bool,
)

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler]


def compute_loss(
    params: AttributeDict,
    model: Union[torch.nn.Module, DDP],
    graph_compiler: BpeCtcTrainingGraphCompiler,
    batch: dict,
    is_training: bool,
    warmup: float = 1.0,
) -> Tuple[Tensor, MetricsTracker]:

    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    feature = batch["inputs"]
    # at entry, feature is (N, T, C)
    assert feature.ndim == 3
    feature = feature.to(device)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    with torch.set_grad_enabled(is_training):
        nnet_output, encoder_memory, memory_mask = model(
            feature, supervisions, warmup=warmup
        )

        supervision_segments, texts = encode_supervisions(
            supervisions, subsampling_factor=params.subsampling_factor
        )

        token_ids = convert_texts_into_ids(texts, graph_compiler.sp)
        decoding_graph = graph_compiler.compile(token_ids)

        dense_fsa_vec = k2.DenseFsaVec(
            nnet_output,
            supervision_segments,
            allow_truncate=params.subsampling_factor - 1,
        )

        ctc_loss = k2.ctc_loss(
            decoding_graph=decoding_graph,
            dense_fsa_vec=dense_fsa_vec,
            output_beam=params.beam_size,
            reduction=params.reduction,
            use_double_scores=params.use_double_scores,
        )

        if params.att_rate > 0.0:
            with torch.set_grad_enabled(is_training):
                mmodel = model.module if hasattr(model, "module") else model

                unsorted_token_ids = graph_compiler.texts_to_ids(supervisions["text"])
                att_loss = mmodel.decoder_forward(
                    encoder_memory,
                    memory_mask,
                    token_ids=unsorted_token_ids,
                    sos_id=graph_compiler.sos_id,
                    eos_id=graph_compiler.eos_id,
                    warmup=warmup,
                )
        else:
            att_loss = torch.tensor([0])

        ctc_loss_is_finite = torch.isfinite(ctc_loss)
        att_loss_is_finite = torch.isfinite(att_loss)
        if torch.any(~ctc_loss_is_finite) or torch.any(~att_loss_is_finite):
            logging.info(
                "Not all losses are finite!\n"
                f"ctc_loss: {ctc_loss}\n"
                f"att_loss: {att_loss}"
            )
            display_and_save_batch(batch, params=params, sp=graph_compiler.sp)
            ctc_loss = ctc_loss[ctc_loss_is_finite]
            att_loss = att_loss[att_loss_is_finite]

            if torch.all(~ctc_loss_is_finite) or torch.all(~att_loss_is_finite):
                raise ValueError(
                    "There are too many utterances in this batch "
                    "leading to inf or nan losses."
                )

        ctc_loss = ctc_loss.sum()
        att_loss = att_loss.sum()

        if params.att_rate > 0.0:
            loss = (1.0 - params.att_rate) * ctc_loss + params.att_rate * att_loss
        else:
            loss = ctc_loss

    assert loss.requires_grad == is_training

    info = MetricsTracker()

    info["frames"] = (
        torch.div(feature_lens, params.subsampling_factor, rounding_mode="floor")
        .sum()
        .item()
    )

    info["utterances"] = feature.size(0)

    info["utt_duration"] = feature_lens.sum().item()

    info["utt_pad_proportion"] = (
        ((feature.size(1) - feature_lens) / feature.size(1)).sum().item()
    )

    info["loss"] = loss.detach().cpu().item()
    info["ctc_loss"] = ctc_loss.detach().cpu().item()
    if params.att_rate > 0.0:
        info["att_loss"] = att_loss.detach().cpu().item()

    return loss, info
    
    
    
def train_one_epoch(
    params: AttributeDict,
    model: Union[torch.nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    graph_compiler: BpeCtcTrainingGraphCompiler,
    train_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    world_size: int = 1,
    rank: int = 0,
) -> None:

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)
        train_dl.dataset.epoch = epoch - 1
        params.cur_epoch = epoch
        
        model.train()
        tot_loss = MetricsTracker()
        
        for batch_idx, batch in enumerate(train_dl):
            num_exaples = len(train_dl)
            params.batch_idx_train += 1
            batch_size = len(batch["supervisions"]["text"])

            loss, loss_info = compute_loss(
                params=params,
                model=model,
                graph_compiler=graph_compiler,
                batch=batch,
                is_training=True,
                warmup=(params.batch_idx_train / params.model_warm_step),
            )

            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            scaler.scale(loss).backward()
            scheduler.step_batch(params.batch_idx_train)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        loss_value = tot_loss["loss"] / tot_loss["frames"]
        params.train_loss = loss_value
        if params.train_loss < params.best_train_loss:
            params.best_train_epoch = params.cur_epoch
            params.best_train_loss = params.train_loss
        
    return loss, num_examples