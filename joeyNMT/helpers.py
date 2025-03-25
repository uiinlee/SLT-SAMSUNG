import copy
import yaml
import functools
import operator
import random
import re
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt

# import importlib_metadata
# import packaging.version
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, Sampler
from joeyNMT.vocabulary import GlossVocabulary, TextVocabulary
from joeyNMT.helpers_for_ddp import get_logger
logger = get_logger(__name__)
# Set numpy print options for better readability
np.set_printoptions(linewidth=sys.maxsize)  # format for printing numpy array

'''
 ----------------------------
 경로 관리 및 디렉토리 생성 함수
 ----------------------------
'''

def make_model_dir(model_dir: Union[str, Path], overwrite: bool = False, **kwargs) -> Path:
    """
    Create a new directory for the model.

    :param model_dir: Path object or string for the model directory.
    :param overwrite: whether to overwrite an existing directory.
    :return: Absolute Path object of the model directory.
    """
    if not isinstance(model_dir, Path):
        model_dir = Path(model_dir)
    model_dir = model_dir.absolute()
    if model_dir.is_dir():
        if not overwrite:
            raise FileExistsError(
                f"Model directory {model_dir} exists and overwriting is disabled."
            )
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


'''
 ----------------------------
 모델 및 데이터 전처리 함수
 ----------------------------
'''
def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers. Transformer helper function.

    :param module: the module to clone
    :param n: clone this many times
    :return: cloned modules
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    # ones = torch.ones(size, size, dtype=torch.bool)
    # return torch.tril(ones, out=ones).unsqueeze(0)
    ones = torch.ones(size, size, dtype=torch.bool)
    mask = torch.tril(ones, out=ones).unsqueeze(0)
    # logger.info(f"subsequent_mask: Created mask with shape {mask.shape} for size {size}")
    return mask

def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def log_data_info(
    train_data: Dataset,
    valid_data: Dataset,
    test_data: Dataset,
    gls_vocab: GlossVocabulary,
    txt_vocab: TextVocabulary,
    logging_function: Callable[[str], None],
):
    """
    Log statistics of data and vocabulary.

    :param train_data:
    :param valid_data:
    :param test_data:
    :param gls_vocab:
    :param txt_vocab:
    :param logging_function:
    """
    logging_function(
        "First training example:\n\t[GLS] {}\n\t[TXT] {}".format(
            " ".join(map(str, train_data[0].gls)),
            " ".join(map(str, train_data[0].txt))
        )
    )

    logging_function(
        "First 10 words (gls): {}".format(
            " ".join("(%d) %s" % (i, t) for i, t in enumerate(gls_vocab.itos[:10]))
        )
    )

    logging_function(
        "First 10 words (txt): {}".format(
            " ".join("(%d) %s" % (i, t) for i, t in enumerate(txt_vocab.itos[:10]))
        )
    )


    logging_function("Number of unique glosses (types): {}".format(len(gls_vocab)))
    logging_function("Number of unique words (types): {}".format(len(txt_vocab)))

def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def bpe_postprocess(string) -> str:
    """
    Post-processor for BPE output. Recombines BPE-split tokens.

    :param string:
    :return: post-processed string
    """
    return string.replace("@@ ", "")


def tile(x: Tensor, count: int, dim: int = 0) -> Tensor:
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.

    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    # yapf: disable
    x = (x.view(batch, -1)
         .transpose(0, 1)
         .repeat(count, 1)
         .transpose(0, 1)
         .contiguous()
         .view(*out_size))
    # yapf: enable
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x

def freeze_params(module: nn.Module) -> None:
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False

def adjust_mask_size(mask: Tensor, batch_size: int, hyp_len: int) -> Optional[Tensor]:
    """
    Adjust mask size along dim=1. used for forced decoding (trg prompting).

    :param mask: trg prompt mask in shape (batch_size, hyp_len)
    :param batch_size:
    :param hyp_len:
    :return: adjusted mask
    """
    if mask is None:
        return None

    if mask.size(1) < hyp_len:
        _mask = mask.new_zeros((batch_size, hyp_len))
        _mask[:, :mask.size(1)] = mask
    elif mask.size(1) > hyp_len:
        _mask = mask[:, :hyp_len]
    else:
        _mask = mask
    assert _mask.size(1) == hyp_len, (_mask.size(), batch_size, hyp_len)
    return _mask

'''
 ----------------------------
 IMU 데이터 전처리 함수
 ----------------------------
'''
def create_imu_mask(sequence_lengths: Tensor, max_length: int) -> Tensor:
    """
    IMU 데이터의 길이에 따라 마스크를 생성합니다.

    :param sequence_lengths: 각 IMU 시퀀스의 길이 (batch_size,).
    :param max_length: 가장 긴 시퀀스의 길이.
    :return: 마스크 텐서 (batch_size, max_length).
    """
    return torch.arange(max_length).expand(len(sequence_lengths), max_length) < sequence_lengths.unsqueeze(1)

def normalize_imu_data(data: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """
    IMU 데이터를 정규화합니다.

    :param data: 원시 IMU 데이터 (batch_size, seq_length, channels).
    :param mean: 채널별 평균 (channels,).
    :param std: 채널별 표준편차 (channels,).
    :return: 정규화된 데이터.
    """
    return (data - mean) / std

def pad_imu_sequences(sequences: List[Tensor], max_length: int, channels: int) -> Tuple[Tensor, Tensor]:
    """
    IMU 데이터를 패딩하여 고정된 길이로 맞춥니다.

    :param sequences: 시퀀스 데이터 리스트 [(seq_length, channels), ...].
    :param max_length: 패딩할 최대 길이.
    :param channels: 데이터의 채널 수.
    :return: 패딩된 텐서와 각 데이터의 길이.
    """
    batch_size = len(sequences)
    padded_sequences = torch.zeros(batch_size, max_length, channels)
    lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)

    for i, seq in enumerate(sequences):
        padded_sequences[i, :seq.size(0), :] = seq

    return padded_sequences, lengths
'''
----------------------------
파일 입출력 함수
----------------------------
'''
def write_list_to_file(output_path: Path, array: List[Any]) -> None:
    """
    Write list of str to file in `output_path`.

    :param output_path: output file path
    :param array: list of strings
    """
    with output_path.open("w", encoding="utf-8") as opened_file:
        for entry in array:
            if isinstance(entry, np.ndarray):
                entry = entry.tolist()
            opened_file.write(f"{entry}\n")

def read_list_from_file(input_path: Path) -> List[str]:
    """
    Read list of str from file in `input_path`.

    :param input_path: input file path
    :return: list of strings
    """
    if input_path is None:
        return []
    return [
        line.rstrip("\n")
        for line in input_path.read_text(encoding="utf-8").splitlines()
    ]

def save_hypothese(output_path: Path, hypotheses: List[str], n_best: int = 1) -> None:
    """
    Save list hypothese to file.

    :param output_path: output file path
    :param hypotheses: hypotheses to write
    :param n_best: n_best size
    """
    if n_best < 1:
        raise ValueError("n_best is lower than 1.")

    if n_best > 1:
        for n in range(n_best):
            write_list_to_file(
                output_path.parent / f"{output_path.stem}-{n}.{output_path.suffix}",
                [hypotheses[i] for i in range(n, len(hypotheses), n_best)],
            )
    else:
        write_list_to_file(output_path, hypotheses)

def store_attention_plots(
    attentions: np.ndarray,
    targets: List[List[str]],
    sources: List[List[str]],
    output_prefix: str,
    indices: List[int],
    tb_writer: Optional[SummaryWriter] = None,
    steps: int = 0,
) -> None:
    """
    Saves attention plots.

    :param attentions: attention scores
    :param targets: list of tokenized targets
    :param sources: list of tokenized sources
    :param output_prefix: prefix for attention plots
    :param indices: indices selected for plotting
    :param tb_writer: Tensorboard summary writer (optional)
    :param steps: current training steps, needed for tb_writer
    """

    for i in indices:
        if i >= len(sources):
            continue
        plot_file = f"{output_prefix}.{i}.png"
        src = sources[i]
        trg = targets[i]
        attention_scores = attentions[i].T
        try:
            fig = plt.heatmap(
                scores=attention_scores,
                column_labels=trg,
                row_labels=src,
                output_path=plot_file,
                dpi=100,
            )
            if tb_writer is not None:
                # lower resolution for tensorboard
                fig = plt.heatmap(
                    scores=attention_scores,
                    column_labels=trg,
                    row_labels=src,
                    output_path=None,
                    dpi=50,
                )
                tb_writer.add_figure(f"attention/{i}.", fig, global_step=steps)
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                f"Couldn't plot example {i}: "
                f"src len {len(src)}, trg len {len(trg)}, "
                f"attention scores shape {attention_scores.shape}"
            )
            continue
'''
----------------------------
체크포인트 관리 함수
----------------------------
'''
def get_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """
    Returns the latest checkpoint (by creation time, not the steps number!)
    from the given directory.
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir: directory containing checkpoints
    :return: latest checkpoint file
    """
    if (ckpt_dir / "latest.ckpt").is_file():
        return ckpt_dir / "latest.ckpt"

    list_of_files = list(ckpt_dir.glob("*.ckpt"))
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=lambda f: f.stat().st_ctime)
        return latest_checkpoint

    # check existence
    raise FileNotFoundError(f"No checkpoint found in directory {ckpt_dir}.")

# def load_checkpoint(path: Path, map_location: Union[torch.device, Dict]) -> Dict:
#     """
#     Load model from saved checkpoint.

#     :param path: path to checkpoint
#     :param map_location: torch device or a dict for mapping
#     :return: checkpoint (dict)
#     """
#     assert path.is_file(), f"Checkpoint {path} not found."
#     checkpoint = torch.load(path, map_location=map_location)
#     return checkpoint

def load_checkpoint(path: Union[str, Path], map_location: Union[torch.device, Dict]) -> Dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint (str or Path)
    :param map_location: torch device or a dict for mapping
    :return: checkpoint (dict)
    """
    if isinstance(path, str):
        path = Path(path)
    assert path.is_file(), f"Checkpoint {path} not found."
    checkpoint = torch.load(path, map_location=map_location)
    return checkpoint

def resolve_ckpt_path(load_model: Optional[Path], model_dir: Path) -> Path:
    """
    Get checkpoint path. if `load_model` is not specified,
    take the best or latest checkpoint from model dir.

    :param load_model: Path(cfg['training']['load_model']) or
                       Path(cfg['testing']['load_model'])
    :param model_dir: Path(cfg['model_dir'])
    :return: resolved checkpoint path
    """
    if load_model is None:
        if (model_dir / "best.ckpt").is_file():
            load_model = model_dir / "best.ckpt"
        else:
            load_model = get_latest_checkpoint(model_dir)
    assert load_model.is_file(), load_model
    return load_model

def delete_ckpt(to_delete: Path) -> None:
    """
    Delete checkpoint

    :param to_delete: checkpoint file to be deleted
    """
    logger = get_logger(__name__)
    try:
        logger.info("Deleting checkpoint: %s", to_delete.as_posix())
        to_delete.unlink()
    except FileNotFoundError as e:
        logger.warning(
            "Wanted to delete old checkpoint %s but "
            "file does not exist. (%s)",
            to_delete,
            e,
        )

def symlink_update(target, link_name):
    """
    This function finds the file that the symlink currently points to, sets it
    to the new target, and returns the previous target if it exists.

    :param target: A path to a file that we want the symlink to point to.
                    no parent dir, filename only, i.e. "10000.ckpt"
    :param link_name: This is the name of the symlink that we want to update.
                      link name with parent dir, i.e. "models/my_model/best.ckpt"

    :return:
        - current_last: This is the previous target of the symlink, before it is
            updated in this function. If the symlink did not exist before or did
            not have a target, None is returned instead.
    """
    target = Path(target)
    link_name = Path(link_name)
    if link_name.is_symlink():
        current_last = link_name.resolve()
        link_name.unlink()
        link_name.symlink_to(target)
        return current_last
    link_name.symlink_to(target)
    return None

'''
----------------------------
데이터 전처리 및 후처리 함수
----------------------------
'''
def flatten(array: List[List[Any]]) -> List[Any]:
    """
    Flatten a nested 2D list. faster even with a very long array than
    [item for subarray in array for item in subarray] or newarray.extend().

    :param array: a nested list
    :return: flattened list
    """
    return functools.reduce(operator.iconcat, array, [])

def expand_reverse_index(reverse_index: List[int], n_best: int = 1) -> List[int]:
    """
    Expand resort_reverse_index for n_best prediction

    ex. 1) reverse_index = [1, 0, 2] and n_best = 2, then this will return
    [2, 3, 0, 1, 4, 5].

    ex. 2) reverse_index = [1, 0, 2] and n_best = 3, then this will return
    [3, 4, 5, 0, 1, 2, 6, 7, 8]

    :param reverse_index: reverse_index returned from batch.sort_by_src_length()
    :param n_best:
    :return: expanded sort_reverse_index
    """
    if n_best == 1:
        return reverse_index

    resort_reverse_index = []
    for ix in reverse_index:
        for n in range(n_best):
            resort_reverse_index.append(ix * n_best + n)
    assert len(resort_reverse_index) == len(reverse_index) * n_best
    return resort_reverse_index

def remove_extra_spaces(s: str) -> str:
    """
    Remove extra spaces
    - used in pre_process() / post_process() in tokenizer.py

    :param s: input string
    :return: string w/o extra white spaces
    """
    s = re.sub("\u200b", "", s)
    s = re.sub("[ 　]+", " ", s)

    s = s.replace(" ?", "?")
    s = s.replace(" !", "!")
    s = s.replace(" ,", ",")
    s = s.replace(" .", ".")
    s = s.replace(" :", ":")
    return s.strip()

def unicode_normalize(s: str) -> str:
    """
    apply unicodedata NFKC normalization
    - used in pre_process() in tokenizer.py

    :param s: input string
    :return: normalized string
    """
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'")
    s = s.replace("“", '"')
    s = s.replace("”", '"')
    return s
'''
----------------------------
로깅 및 설정 함수
----------------------------
'''

def make_logger(model_dir: Path, log_file: str = "train.log") -> Callable[[str], None]:
    """
    Initialize and return a logger.

    :param model_dir: Path to the model directory where the log file will be saved
    :param log_file: Name of the log file
    :return: Logger instance
    """
    logger = get_logger(__name__, log_file=model_dir / log_file)
    return logger

# def log_cfg(cfg: Dict, logger: Callable[[str], None], prefix: str = "cfg") -> None:
#     """
#     Log configuration dictionary.

#     :param cfg: Configuration dictionary
#     :param logger: Logging function (e.g., logger.info)
#     :param prefix: Prefix for each log entry
#     """
#     for key, value in cfg.items():
#         if isinstance(value, dict):
#             logger(f"{prefix}.{key}:")
#             log_cfg(value, logger, prefix=f"{prefix}.{key}")
#         else:
#             logger.info(f"{prefix}.{key}: {value}")

def log_cfg(cfg: Dict, logger: Any, prefix: str = "cfg") -> None:
    """
    Log configuration dictionary.

    :param cfg: Configuration dictionary
    :param logger: Logging object
    :param prefix: Prefix for each log entry
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = ".".join([prefix, k])
            log_cfg(v, logger, prefix=p)
        else:
            p = ".".join([prefix, k])
            logger.info("{:34s} : {}".format(p, v))


# def parse_global_args(cfg: Dict, rank: int, mode: str = "train") -> Dict:
#     """
#     Parse global arguments from configuration.

#     :param cfg: Configuration dictionary
#     :param rank: DDP local rank
#     :param mode: Mode of operation, e.g., "train"
#     :return: Parsed arguments dictionary
#     """
#     args = cfg.copy()
#     args["rank"] = rank
#     args["mode"] = mode
#     return args

def set_validation_args(test_cfg: Dict) -> Dict:
    """
    Set validation-specific arguments based on test configuration.

    :param test_cfg: Test configuration dictionary
    :return: Validation arguments dictionary
    """
    val_args = test_cfg.copy()
    # Add or modify validation-specific arguments here if needed
    return val_args
'''
----------------------------
데이터셋 및 반복기 함수
----------------------------
'''

def make_data_iter(
    dataset: Dataset,
    batch_size: int,
    batch_type: str = "sentence",
    train: bool = True,
    shuffle: bool = True,
    sampler: Optional[Sampler] = None,
    num_workers: int = 0,
    device: torch.device = torch.device("cpu"),
    eos_index: int = 0,
    pad_index: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for the given dataset.

    :param dataset: Dataset object
    :param batch_size: Number of samples per batch
    :param batch_type: Type of batching, e.g., "sentence"
    :param train: Whether the DataLoader is for training
    :param shuffle: Whether to shuffle the data
    :param sampler: Sampler object, e.g., DistributedSampler
    :param num_workers: Number of worker processes
    :param device: Device to load data onto
    :param eos_index: End-of-sequence token index
    :param pad_index: Padding token index
    :return: DataLoader object
    """
    # Define a collate function based on batch_type if needed
    def collate_fn(batch):
        # Implement custom collation logic here
        # For simplicity, using default collation
        return torch.utils.data.dataloader.default_collate(batch)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None) and shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False,
    )

def create_imu_mask(sequence_lengths: Tensor, max_length: int) -> Tensor:
    """
    Create a mask for IMU sequences based on their lengths.

    :param sequence_lengths: Tensor of sequence lengths (batch_size,).
    :param max_length: Maximum sequence length.
    :return: Mask tensor of shape (batch_size, max_length).
    """
    return torch.arange(max_length).expand(len(sequence_lengths), max_length) < sequence_lengths.unsqueeze(1)


# def adjust_mask_size(trg_prompt_mask: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
#     """
#     Adjust the size of the trg_prompt_mask to match the given batch_size and seq_len.
    
#     :param trg_prompt_mask: A tensor representing a prompt mask. It could be 1D or 2D, e.g. [prompt_len] or [1, prompt_len].
#     :param batch_size: The desired batch size.
#     :param seq_len: The desired sequence length.
#     :return: A mask tensor of shape [batch_size, seq_len].
#     """

#     # Ensure mask is at least 2D: [1, prompt_len] or [batch_size, prompt_len]
#     if trg_prompt_mask.dim() == 1:
#         trg_prompt_mask = trg_prompt_mask.unsqueeze(0)  # [1, prompt_len]

#     # 현재 mask 형태: [current_batch_size, current_len]
#     current_batch_size, current_len = trg_prompt_mask.size()

#     # batch_size가 1인 경우 등, batch 크기 맞추기
#     if current_batch_size != batch_size:
#         # batch 차원을 batch_size만큼 반복
#         trg_prompt_mask = trg_prompt_mask.repeat(batch_size, 1)
    
#     # seq_len 맞추기
#     current_batch_size, current_len = trg_prompt_mask.size()

#     if current_len < seq_len:
#         # seq_len이 더 길다면, 뒤에 False(또는 0)로 패딩
#         pad_len = seq_len - current_len
#         padding = torch.zeros((current_batch_size, pad_len), dtype=trg_prompt_mask.dtype, device=trg_prompt_mask.device)
#         trg_prompt_mask = torch.cat([trg_prompt_mask, padding], dim=1)
#     elif current_len > seq_len:
#         # seq_len보다 길다면 잘라내기
#         trg_prompt_mask = trg_prompt_mask[:, :seq_len]

#     return trg_prompt_mask
