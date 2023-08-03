import argparse
import inspect
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional
import glob
import torch
from lhotse.cut import Cut
from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy
from lhotse.dataset import (
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SingleCutSampler,
    SpecAugment,
)
from lhotse.dataset.input_strategies import (
    AudioSamples,
    OnTheFlyFeatures,
)
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader

from icefall.utils import str2bool


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


parser = argparse.ArgumentParser(prog='Dataset for flower simulation')
parser.add_argument("--manifest_dir", type=Path, default=Path("data/fbank"), help="Path to directory with train/valid/test cuts.",)
parser.add_argument("--max_duration", type=int, default=200.0, help="Max pooled recordings duration (sec) in a single batch. You can reduce it if it causes CUDA OOM.",)
parser.add_argument("--bucketing_sampler", type=str2bool, default=False, help="When enabled, the batches will come from buckets of similar duration.",)
parser.add_argument("--num_buckets", type=int, default=30, help="Num of buckets for the DynamicBucketingSampler(may be increase it for large dataset).",)
parser.add_argument("--concatenate_cuts", type=str2bool, default=False, help="When enabled, utterances (cuts) will be concat to min the amount of pad.",)
parser.add_argument("--duration_factor", type=float, default=1.0, help="Determines the maximum duration of a concatenated cut",)
parser.add_argument("--gap", type=float, default=1.0, help="The amount of padding (in seconds) inserted between concatenated cuts",)
parser.add_argument("--on_the_fly_feats", type=str2bool, default=False, help="When enabled, use on-the-fly cut mixing and feature extraction.",)
parser.add_argument("--shuffle", type=str2bool, default=True, help="When enabled (=default), the examples will be shuffled for each epoch.",)
parser.add_argument("--drop_last", type=str2bool, default=True, help="Whether to drop last batch. Used by sampler.",)
parser.add_argument("--return_cuts", type=str2bool, default=True, help="True, each batch have the field: batch['supervisions']the cuts that",)
parser.add_argument("--num_workers", type=int, default=2, help="The number of training dataloader workers that collect the batches.",)
parser.add_argument("--enable_spec_aug", type=str2bool, default=True, help="When enabled, use SpecAugment for training dataset.",)
parser.add_argument("--spec_aug_time_warp_factor", type=int, default=80, help="Used only when --enable-spec-aug is True.",)
parser.add_argument("--enable_musan", type=str2bool, default=True, help="When Treu, select noise from MUSAN and mix it with training dataset.",)
parser.add_argument("--input_strategy", type=str, default="PrecomputedFeatures", help="AudioSamples or PrecomputedFeatures",)

args = parser.parse_args()

manifest_dir = args.manifest_dir
max_duration = args.max_duration
bucketing_sampler = args.bucketing_sampler
num_buckets = args.num_buckets
concatenate_cuts = args.concatenate_cuts
duration_factor = args.duration_factor
gap = args.gap
on_the_fly_feats = args.on_the_fly_feats
shuffle = args.shuffle
drop_last = args.drop_last
return_cuts = args.return_cuts
num_workers = args.num_workers
enable_spec_aug = args.enable_spec_aug
spec_aug_time_warp_factor = args.spec_aug_time_warp_factor
enable_musan = args.enable_musan
input_strategy = args.input_strategy








def train_dataloaders(cuts_train: CutSet, sampler_state_dict: Optional[Dict[str, Any]] = None,) -> DataLoader:

    transforms = []

    if enable_musan:
        logging.info("Enable MUSAN")
        logging.info("About to get Musan cuts")
        cuts_musan = load_manifest(manifest_dir / "musan_cuts.jsonl.gz")
        transforms.append(CutMix(cuts=cuts_musan, prob=0.5, snr=(10, 20), preserve_id=True))

    else:

        logging.info("Disable MUSAN")

    if concatenate_cuts:
        
        logging.info(
            f"Using cut concatenation with duration factor "
            f"{duration_factor} and gap {gap}."
        )
          
        transforms = [
            CutConcatenate(
                duration_factor=duration_factor, gap=gap
            )
        ] + transforms

    input_transforms = []
    if enable_spec_aug:
        logging.info("Enable SpecAugment")
        logging.info(f"Time warp factor: {spec_aug_time_warp_factor}")
        num_frame_masks = 10
        num_frame_masks_parameter = inspect.signature(
            SpecAugment.__init__
        ).parameters["num_frame_masks"]
        if num_frame_masks_parameter.default == 1:
            num_frame_masks = 2
        logging.info(f"Num frame mask: {num_frame_masks}")
        input_transforms.append(
            SpecAugment(
                time_warp_factor=spec_aug_time_warp_factor,
                num_frame_masks=num_frame_masks,
                features_mask_size=27,
                num_feature_masks=2,
                frames_mask_size=100,
            )
        )
    else:
        logging.info("Disable SpecAugment")

    logging.info(" ---> Creating the dataset <---- ")

    train = K2SpeechRecognitionDataset(
        input_strategy=eval(input_strategy)(),
        cut_transforms=transforms,
        input_transforms=input_transforms,
        return_cuts=return_cuts,
    )

    if on_the_fly_feats:
        train = K2SpeechRecognitionDataset(
            cut_transforms=transforms,
            input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
            input_transforms=input_transforms,
            return_cuts=return_cuts,
        )


    if bucketing_sampler:
        logging.info("Using DynamicBucketingSampler.")
        train_sampler = DynamicBucketingSampler(
            cuts_train,
            max_duration=max_duration,
            shuffle=shuffle,
            num_buckets=num_buckets,
            drop_last=drop_last,
        )

    else:
        logging.info("Using SingleCutSampler.")
        train_sampler = SingleCutSampler(
            cuts_train,
            max_duration=max_duration,
            shuffle=shuffle,
        )

    logging.info("About to create train dataloader")

    if sampler_state_dict is not None:
        logging.info("Loading sampler state dict")
        train_sampler.load_state_dict(sampler_state_dict)

    seed = torch.randint(0, 100000, ()).item()
    worker_init_fn = _SeedWorkers(seed)

    train_dl = DataLoader(
        train,
        sampler=train_sampler,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=False,
        worker_init_fn=worker_init_fn,
    )

    return train_dl




def remove_short_and_long_utt(c:Cut):
    return 1.0 <= c.duration <= 20.0


def load_dataset_tedlium() -> CutSet:

    dataloaders = []
    logging.info('About to return all cuts')

    for i, file in enumerate (glob.glob("/stek/ASR-FL/icefall/egs/tedlium3/ASR/train/*.jsonl.gz")):
        train_cuts = load_manifest_lazy (file)
        dataloaders.append(train_dataloaders(train_cuts))
    return dataloaders




