import argparse
import copy
import logging
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union, List
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
import flwr as fl
from collections import OrderedDict
import numpy as np
import torch
import random
import torch.nn as nn
import torch.functional as F
from scripts import *
from k2_dataset import *
import flwr as fl
from flwr.server.strategy.aggregate import aggregate
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from scripts import *
import torch.multiprocessing as mp
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    Scalar,
    NDArrays,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

torch.set_num_threads(1)

parser = argparse.ArgumentParser(prog='Flower client for simulations')

parser.add_argument('--modelname', type=str, default="TEDLIUM", help='Name of the model to be saved in trained_models... modelname-round-x.pth')
parser.add_argument('--centraltraining', action='store_true')
args=parser.parse_args()

###Global parameters of FL####
RAY_DEDUP_LOGS=0

_MODELNAME=args.modelname
centralizedTraining = args.centraltraining

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataloaders = load_dataset_tedlium()



def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--num-encoder-layers",
        type=int,
        default=4,
        help="Number of conformer encoder layers..",
    )

    parser.add_argument(
        "--num-decoder-layers",
        type=int,
        default=0,
        help="""Number of decoder layer of transformer decoder.
        Setting this to 0 will not create the decoder at all (pure CTC model)
        """,
    )

    parser.add_argument(
        "--att-rate",
        type=float,
        default=0.8,
        help="""The attention rate.
        The total loss is (1 -  att_rate) * ctc_loss + att_rate * att_loss
        """,
    )

    parser.add_argument(
        "--dim-feedforward",
        type=int,
        default=1536,
        help="Feedforward module dimension of the conformer model.",
    )

    parser.add_argument(
        "--nhead",
        type=int,
        default=2,
        help="Number of attention heads in the conformer multiheadattention modules.",
    )

    parser.add_argument(
        "--dim-model",
        type=int,
        default=384,
        help="Attention dimension in the conformer model.",
    )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="conformer_ctc/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--lang-dir",
        type=str,
        default="data/lang_bpe_500",
        help="""The lang dir
        It contains language related input files such as
        "lexicon.txt" and "bpe.model"
        """,
    )

    parser.add_argument(
        "--initial-lr",
        type=float,
        default=0.003,
        help="The initial learning rate.  This value should not need to be changed.",
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=5000,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=6,
        help="Number of epochs that affects how rapidly the learning rate decreases.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--print-diagnostics",
        type=str2bool,
        default=False,
        help="Accumulate stats on activations, print them and exit.",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=4000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 0.
        """,
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=30,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=100,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision training.",
    )

    add_model_arguments(parser)

    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 10,
            "reset_interval": 200,
            "valid_interval": 1000,
            # parameters for conformer
            "feature_dim": 80,
            "subsampling_factor": 4,
            # parameters for ctc loss
            "beam_size": 10,
            "reduction": "none",
            "use_double_scores": True,
            # parameters for Noam
            "model_warm_step": 3000,  # arg given to model, not for lrate
            "env_info": get_env_info(),
        }
    )

    return params


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {
            k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
            for k, v in params_dict
        }
    )
    net.load_state_dict(state_dict, strict=True)

class custom_strategy(fl.server.strategy.FedAvg):

    def __init__(
            self,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

    def __repr__(self) -> str:
        return "FedCustom"


    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Sample clients

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        sample_size=random.randint(min_num_clients, sample_size)
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create custom configs
        n_clients = len(clients)

        half_clients = n_clients // 2
        standard_config = {"lr": 0.001}
        higher_lr_config = {"lr": 0.003}
        fit_configurations = []
        #print("CONFIG.  sample:", sample_size,"min_num_clients:", min_num_clients, "n_clients:",n_clients)
        for idx, client in enumerate(clients):
            if idx < half_clients:
                fit_configurations.append((client, FitIns(parameters, standard_config)))
            else:
                fit_configurations.append(
                    (client, FitIns(parameters, higher_lr_config))
                )
        return fit_configurations

    def aggregate_fit(self,
                      server_round:int,
                      results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
                      failures: List[BaseException],
                      ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]: #Optional [fl.common.Parameters] :  #Tuple[Optional[Parameters], Dict[str, Scalar]]

        if not results:
            return None, {}

        if self.accept_failures and failures:
            return None, {}

        key_name = "train_loss" if weight_strategy == "loss" else "wer"

        weights = None

        if weight_strategy == 'num':
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for client, fit_res in results
            ]
            weights = aggregate(weights_results)

        elif weight_strategy == "loss" or weight_strategy == "wer":
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.metrics[key_name])
                for client, fit_res in results
            ]
            weights = aggregate(weights_results)
        
        if weights is not None:

            params_dict = zip(net.state_dict().keys(), weights)
            state_dict = OrderedDict({k: torch.Tensor(np.array(v)) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

            if centralizedTraining:
                print(f">>>>>>>>>>>>>>>>>>>>>>One centralized training epoch will be performed<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                train_asr(net, epochs=1, trainloader=centraloader)
                # Save the model
            else:
                print(f"Centralized training not in place")

            torch.save(net.state_dict(), f"trained_models/{_MODELNAME}-round-{server_round}.pth")
            new_parameters = get_parameters(net)
            return ndarrays_to_parameters(new_parameters), {}
        else:
            print(f"returning None weights, something went wrongh during aggregation..... !!!!!!!!!!!!!!!")
            return ndarrays_to_parameters(weights), {}


class asr_client(fl.client.Client):

    def __init__(self, cid, params, net, optimizer, scheduler, graph_compiler, trainloader, scaler, world_size, rank):
        self.cid = cid
        self.params = params
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.graph_compiler = graph_compiler
        self.trainloader=trainloader
        self.scaler = scaler
        self.world_size = world_size
        self.rank = rank

    def get_parameters(self, ins:GetParametersIns) -> GetParametersRes:
        print(f"[Client {self.cid}] get_parameters")
        ndarrays: List[np.ndarray] = get_parameters(self.net)
        parameters = ndarrays_to_parameters(ndarrays)
        status = Status(code=Code.OK, message='Success')
        torch.cuda.empty_cache()
        gc.collect()
        return GetParametersRes(status=status, parameters=parameters)

    def fit(self, ins: FitIns) -> FitRes:   #FitRes
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)
        set_parameters(self.net, ndarrays_original)
        
        loss, num_examples = train_one_epoch(self.params, self.net, self.optimizer, self.scheduler, self.graph_compiler, self.trainloader, self.scaler, self.world_size, self.rank)
        
        ndarrays_updated = get_parameters(self.net)
        parameters_updated = ndarrays_to_parameters(ndarrays_updated)
        status = Status(code=Code.OK, message='Success')
        metrics = {"train_loss": loss}#, "wer": avg_wer}
        torch.cuda.empty_cache()
        gc.collect()
        return FitRes(status=status, parameters=parameters_updated, num_examples=num_examples, metrics=metrics)

    
        
        
def client_fn(cid) -> asr_client:
    
    parser = get_parser()
    args = parser.parse_args()
    params = get_params()
    params.update(vars(args))

    rank = 0

    fix_random_seed(params.seed)

    lexicon = Lexicon(params.lang_dir)
    max_token_id = max(lexicon.tokens)
    num_classes = max_token_id + 1

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    if "lang_bpe" not in str(params.lang_dir):
        raise ValueError(
            f"Unsupported type of lang dir (we expected it to have "
            f"'lang_bpe' in its name): {params.lang_dir}"
        )

    graph_compiler = BpeCtcTrainingGraphCompiler(
        params.lang_dir,
        device=device,
        sos_token="<sos/eos>",
        eos_token="<sos/eos>",
    )

    logging.info("About to create model")
    model = Conformer(
        num_features=params.feature_dim,
        num_classes=num_classes,
        subsampling_factor=params.subsampling_factor,
        d_model=params.dim_model,
        nhead=params.nhead,
        dim_feedforward=params.dim_feedforward,
        num_encoder_layers=params.num_encoder_layers,
        num_decoder_layers=params.num_decoder_layers,
    )

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    assert params.save_every_n >= params.average_period
    model_avg: Optional[torch.nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model)
        
    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank])

    optimizer = optim.Eve(model.parameters(), lr=params.initial_lr)
    scheduler = optim.Eden(optimizer, params.lr_batches, params.lr_epochs)

    if checkpoints and checkpoints.get("optimizer") is not None:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])

    if checkpoints and checkpoints.get("scheduler") is not None:
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])
        
    scaler = GradScaler(enabled=params.use_fp16)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])


    trainloader = train_dataloaders[int(cid)]


    return asr_client(cid, params, net, optimizer, scheduler, graph_compiler, trainloader, scaler, world_size, rank)

client_resourses = {
    "num_cpus": 1,
    "num_gpus": 1,
    }

ram_memory = 16_000 * 1024 * 1024

my_strategy = custom_strategy(
    fraction_fit=0.1,
    fraction_evaluate=0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    )

fl.simulation.start_simulation(client_fn=client_fn,
                               num_clients=2351,
                               config=fl.server.ServerConfig(num_rounds=200),
                               strategy=my_strategy,
                               ray_init_args = {
                                   "include_dashboard": False, # we need this one for tracking
                                    "num_cpus": 2,
                                    "num_gpus": 1,
                                    "_memory": ram_memory,
                                    "object_store_memory": 10**9,
                                },
                                client_resources= client_resourses)


