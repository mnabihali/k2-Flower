import argparse
import logging
from pathlib import Path
from shutil import copyfile
from typing import Optional, Tuple
from torch.utils.data import DataLoader
import k2
from lhotse import load_manifest
from lhotse.dataset.speech_recognition import K2SpeechRecognitionDataset
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from asr_datamodule import TedLiumAsrDataModuke
from icefall.env import get_env_info
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    encode_supervisions,
    setup_logger,
    str2bool,
)


ls = []
cs = load_manifest("/stek/ASR-FL/icefall/egs/tedlium3/ASR/conformer_ctc2/data/fbank/tedlium_cuts_train.jsonl.gz")
spks = cs.speakers
cuts_by_speaker = {}
for spk in spks:
    print(spk)
    ls.append(spk)
    cuts_by_speaker[spk] = cs.filter(lambda c: c.supervisions[0].speaker == spk)
    # optionally save to file
    cuts_by_speaker[spk].to_file("/stek/ASR-FL/icefall/egs/tedlium3/ASR/train/{}.jsonl.gz".format(spk))

print('len', len(ls))








