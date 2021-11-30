from dataclasses import dataclass
from enum import Enum

@dataclass
class OptConfig:
    step_size: int = 8
    num_tokens_per_batch: int = 1048576
    lr: float = 0.001
    num_epochs: int = 10000
    beta1 = 0.0

@dataclass
class SeqConfig:
    max_source_length: int = 2048
    max_target_length: int = 512

@dataclass
class ExpConfig:
    device: str = 'cuda'
    model_name: str = 'google/byt5-small'
    dataset: str = 'gem-xsum'
    opt: OptConfig = OptConfig()
    seq: SeqConfig = SeqConfig()
    chkpt_dir: str = 'chkpts/default'
