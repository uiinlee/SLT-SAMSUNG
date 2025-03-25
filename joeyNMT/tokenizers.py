# coding: utf-8
"""
Tokenizer module
"""
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Union

import sentencepiece as sp
from subword_nmt import apply_bpe

from joeyNMT.config import ConfigurationError
from joeyNMT.helpers import remove_extra_spaces, unicode_normalize
from joeyNMT.helpers_for_ddp import get_logger
from joeyNMT.vocabulary import Vocabulary
from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer

logger = get_logger(__name__)


class BasicTokenizer:
    # pylint: disable=too-many-instance-attributes
    SPACE = chr(32)  # ' ': half-width white space (ascii)
    SPACE_ESCAPE = chr(9601)  # '▁': sentencepiece default

    def __init__(
        self,
        level: str = "word",
        lowercase: bool = False,
        normalize: bool = False,
        max_length: int = -1,
        min_length: int = -1,
        **kwargs,
    ):
        # pylint: disable=unused-argument
        self.level = level
        self.lowercase = lowercase
        self.normalize = normalize

        # filter by length
        self.max_length = max_length
        self.min_length = min_length

        # pretokenizer
        self.pretokenizer = kwargs.get("pretokenizer", "none").lower()
        assert self.pretokenizer in ["none", "moses"], \
            "Currently, we support moses tokenizer only."
        # sacremoses
        if self.pretokenizer == "moses":
            try:
                (  # pylint: disable=import-outside-toplevel
                    MosesDetokenizer,
                    MosesPunctNormalizer,
                    MosesTokenizer,
                )
                # sacremoses package has to be installed.
                # https://github.com/alvations/sacremoses
            except ImportError as e:
                logger.error(e)
                raise ImportError from e

            self.lang = kwargs.get("lang", "en")
            self.moses_tokenizer = MosesTokenizer(lang=self.lang)
            self.moses_detokenizer = MosesDetokenizer(lang=self.lang)
            if self.normalize:
                self.moses_normalizer = MosesPunctNormalizer()

        # these attributes will be set later in `set_vocab(...)`
        self.vocab = None
        self.unk_token = None
        self.bos_token = None
        self.eos_token = None
        self.sep_token = None
        self.specials = []
        self.lang_tags = []

    def pre_process(self, raw_input: str, allow_empty: bool = False) -> str:
        """
        Pre-process text
            - ex.) Lowercase, Normalize, Remove emojis,
                Pre-tokenize(add extra white space before punc) etc.
            - applied for all inputs both in training and inference.

        :param raw_input: raw input string
        :param allow_empty: whether to allow empty string
        :return: preprocessed input string
        """
        if not allow_empty:
            assert isinstance(raw_input, str) and raw_input.strip() != "", \
                "The input sentence is empty! Please make sure " \
                "that you are feeding a valid input."

        if self.normalize:
            raw_input = remove_extra_spaces(unicode_normalize(raw_input))

        if self.pretokenizer == "moses":
            if self.normalize:
                raw_input = self.moses_normalizer.normalize(raw_input)
            raw_input = self.moses_tokenizer.tokenize(raw_input, return_str=True)

        if self.lowercase:
            raw_input = raw_input.lower()

        if not allow_empty:
            # ensure the string is not empty.
            assert raw_input is not None and len(raw_input) > 0, raw_input
        return raw_input

    def __call__(self, raw_input: str, is_train: bool = False) -> List[str]:
        """Tokenize single sentence"""
        if raw_input is None:
            return None

        if self.level == "word":
            sequence = raw_input.split(self.SPACE)
        elif self.level == "char":
            sequence = list(raw_input.replace(self.SPACE, self.SPACE_ESCAPE))

        if is_train and self._filter_by_length(len(sequence)):
            return None
        return sequence

    def _filter_by_length(self, length: int) -> bool:
        """
        Check if the given seq length is out of the valid range.

        :param length: (int) number of tokens
        :return: True if the length is invalid(= to be filtered out), False if valid.
        """
        return (self.max_length > 0 and length > self.max_length) or \
               (self.min_length > 0 and length < self.min_length)

    def _remove_special(self, sequence: List[str], generate_unk: bool = False):
        """
        Filter out special tokens from the sequence.
        If generate_unk=False, also remove self.unk_token from `specials`.
        """
        # If generate_unk=False, we exclude self.unk_token as well
        # to avoid it being removed from the sequence
        filtered_specials = self.specials if generate_unk else self.specials + [self.unk_token]

        valid = [token for token in sequence if token not in filtered_specials]
        if len(valid) == 0:  # if empty, return <unk>
            valid = [self.unk_token]
        return valid

    def post_process(
        self,
        sequence: Union[List[str], str],
        generate_unk: bool = True,
        cut_at_sep: bool = True
    ) -> str:
        """
        Detokenize sequence of tokens.

        :param sequence: list of tokens or a single string
        :param generate_unk: if True, allow <unk> to remain
        :param cut_at_sep: if True, cut off tokens at first <sep> encountered
        :return: detokenized string
        """
        if isinstance(sequence, list):
            if cut_at_sep and self.sep_token in sequence:
                # find first <sep> and cut
                try:
                    sep_pos = sequence.index(self.sep_token)
                    sequence = sequence[sep_pos + 1:]
                except ValueError:
                    pass

            sequence = self._remove_special(sequence, generate_unk=generate_unk)

            if self.level == "word":
                if self.pretokenizer == "moses":
                    sequence = self.moses_detokenizer.detokenize(sequence)
                else:
                    sequence = self.SPACE.join(sequence)
            elif self.level == "char":
                sequence = "".join(sequence).replace(self.SPACE_ESCAPE, self.SPACE)

        # Remove extra spaces
        if self.normalize:
            sequence = remove_extra_spaces(sequence)

        # ensure the string is not empty.
        assert sequence is not None and len(sequence) > 0, sequence
        return sequence
    
    def set_vocab(self, vocab: 'Vocabulary') -> None:
        """
        Set vocab for the tokenizer.

        :param vocab: (Vocabulary) 
            - vocab.stoi: dict(token -> index)
            - vocab.specials: list of special tokens (e.g. ["<pad>", "<unk>", "<s>", "</s>"])
            - vocab.lang_tags: list of language-specific tags, if any.
        """
        from joeyNMT.vocabulary import UNK_TOKEN, BOS_TOKEN, EOS_TOKEN
        # pylint: disable=attribute-defined-outside-init

        # 1) 우선 기본값(문자열)으로 초기화
        self.unk_token = UNK_TOKEN
        self.bos_token = BOS_TOKEN
        self.eos_token = EOS_TOKEN
        self.sep_token = None

        # 2) vocab 안에 UNK, BOS, EOS, SEP 등이 있는지 확인 후 반영

        # --- unk_token 설정 ---
        if UNK_TOKEN in vocab.stoi:
            self.unk_token = UNK_TOKEN
            logger.info(f"Using unk_token from vocab: '{UNK_TOKEN}' (idx {vocab.stoi[UNK_TOKEN]})")
        else:
            logger.warning(f"'{UNK_TOKEN}' not found in vocab. Using default: '{self.unk_token}'")

        if BOS_TOKEN in vocab.stoi:
            self.bos_token = BOS_TOKEN
            logger.info(f"Using bos_token from vocab: '{BOS_TOKEN}' (idx {vocab.stoi[BOS_TOKEN]})")
        else:
            logger.warning(f"'{BOS_TOKEN}' not found in vocab. Using default: '{self.bos_token}'")

        if EOS_TOKEN in vocab.stoi:
            self.eos_token = EOS_TOKEN
            logger.info(f"Using eos_token from vocab: '{EOS_TOKEN}' (idx {vocab.stoi[EOS_TOKEN]})")
        else:
            logger.warning(f"'{EOS_TOKEN}' not found in vocab. Using default: '{self.eos_token}'")

        if "<sep>" in vocab.stoi:
            self.sep_token = "<sep>"
            logger.info(f"Set sep_token to '<sep>' (idx {vocab.stoi['<sep>']})")
        else:
            logger.info("No '<sep>' token found in vocab; sep_token = None")


        # 3) specials & lang_tags 갱신
        #    - tokenizer가 사용할 special 토큰 목록, 언어 태그 목록을 가져온다.
        specials = getattr(vocab, "specials", [])
        lang_tags = getattr(vocab, "lang_tags", [])
        # remove <unk> from specials if present
        self.specials = [tok for tok in (specials + lang_tags) if tok != self.unk_token]
        self.lang_tags = lang_tags

        logger.debug(f"Tokenizer specials after removing <unk>: {self.specials}")

    def encode(self, list_of_sequences, bos=False, eos=False):
        """
        JoeyNMT 내부에서 쓰는 signature: 
        (encoded_batch, lengths, prompt_mask) 형태를 반환.

        :param list_of_sequences: [[토큰], [토큰], ...] 형태의 배치(또는 raw string의 리스트).
        :param bos: 문장 시작 토큰 추가 여부
        :param eos: 문장 끝 토큰 추가 여부
        :return: 
          - encoded_batch: 각 시퀀스(토큰 리스트)를 bos/eos 붙인 결과 2차원 리스트
          - lengths: 각 시퀀스 길이 리스트
          - prompt_mask: 여기선 일단 전부 0으로 두거나, 프롬프트가 있다면 해당 위치에 마스크
        """
        encoded_batch = []
        lengths = []
        prompt_masks = []

        if self.vocab is None:
            # safety check
            raise ValueError("Tokenizer has no vocab set. Please call `set_vocab(...)` first.")

        for seq in list_of_sequences:
            # 1) handle str vs list
            if isinstance(seq, str):
                tokens = self.__call__(self.pre_process(seq))
            else:
                tokens = seq
            if tokens is None:
                # if length filter was triggered or something else
                tokens = [self.unk_token]
            # 2) bos/eos
            if bos:
                tokens = [self.bos_token] + tokens
            if eos:
                tokens = tokens + [self.eos_token]
            # 3) vocab lookup → int IDs
            ids = []
            for t in tokens:
                if t in self.vocab.stoi:
                    ids.append(self.vocab.stoi[t])
                else:
                    ids.append(self.vocab.DEFAULT_UNK_ID)

            encoded_batch.append(ids)
            lengths.append(len(ids))
            prompt_masks.append([0] * len(ids))

        return encoded_batch, lengths, prompt_masks
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(level={self.level}, "
            f"lowercase={self.lowercase}, normalize={self.normalize}, "
            f"filter_by_length=({self.min_length}, {self.max_length}), "
            f"pretokenizer={self.pretokenizer})"
        )


class SentencePieceTokenizer(BasicTokenizer):

    def __init__(
        self,
        level: str = "bpe",
        lowercase: bool = False,
        normalize: bool = False,
        max_length: int = -1,
        min_length: int = -1,
        **kwargs,
    ):
        super().__init__(level, lowercase, normalize, max_length, min_length, **kwargs)
        assert self.level == "bpe"

        self.model_file: Path = Path(kwargs["model_file"])
        assert self.model_file.is_file(), f"model file {self.model_file} not found."

        self.spm = sp.SentencePieceProcessor()
        self.spm.load(kwargs["model_file"])

        self.nbest_size: int = kwargs.get("nbest_size", 5)
        self.alpha: float = kwargs.get("alpha", 0.0)

    def __call__(self, raw_input: str, is_train: bool = False) -> List[str]:
        """Tokenize"""
        if raw_input is None:
            return None

        if is_train and self.alpha > 0:
            tokenized = self.spm.sample_encode_as_pieces(
                raw_input,
                nbest_size=self.nbest_size,
                alpha=self.alpha,
            )
        else:
            tokenized = self.spm.encode(raw_input, out_type=str)

        if is_train and self._filter_by_length(len(tokenized)):
            return None
        return tokenized

    def post_process(
        self,
        sequence: Union[List[str], str],
        generate_unk: bool = True,
        cut_at_sep: bool = True
    ) -> str:
        """Detokenize"""
        if isinstance(sequence, list):
            if cut_at_sep:
                try:
                    sep_pos = sequence.index(self.sep_token)  # cut off prompt
                    sequence = sequence[sep_pos:]
                except ValueError as e:  # pylint: disable=unused-variable # noqa: F841
                    pass
            sequence = self._remove_special(sequence, generate_unk=generate_unk)

            # Decode back to str
            sequence = self.spm.decode(sequence)
            sequence = sequence.replace(self.SPACE_ESCAPE, self.SPACE).strip()

        # Apply moses detokenizer
        if self.pretokenizer == "moses":
            sequence = self.moses_detokenizer.detokenize(sequence.split())

        # Remove extra spaces
        if self.normalize:
            sequence = remove_extra_spaces(sequence)

        # ensure the string is not empty.
        assert sequence is not None and len(sequence) > 0, sequence
        return sequence

    def set_vocab(self, vocab) -> None:
        """Set vocab"""
        super().set_vocab(vocab)
        self.spm.SetVocabulary(vocab._itos)  # pylint: disable=protected-access

    def copy_cfg_file(self, model_dir: Path) -> None:
        """Copy config file to model_dir"""
        if (model_dir / self.model_file.name).is_file():
            logger.warning(
                "%s already exists. Stop copying.",
                (model_dir / self.model_file.name).as_posix(),
            )
        shutil.copy2(self.model_file, (model_dir / self.model_file.name).as_posix())

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(level={self.level}, "
            f"lowercase={self.lowercase}, normalize={self.normalize}, "
            f"filter_by_length=({self.min_length}, {self.max_length}), "
            f"pretokenizer={self.pretokenizer}, "
            f"tokenizer={self.spm.__class__.__name__}, "
            f"nbest_size={self.nbest_size}, alpha={self.alpha})"
        )


class SubwordNMTTokenizer(BasicTokenizer):

    def __init__(
        self,
        level: str = "bpe",
        lowercase: bool = False,
        normalize: bool = False,
        max_length: int = -1,
        min_length: int = -1,
        **kwargs,
    ):
        super().__init__(level, lowercase, normalize, max_length, min_length, **kwargs)
        assert self.level == "bpe"

        codes_file = Path(kwargs["codes"])
        assert codes_file.is_file(), f"codes file {codes_file} not found."

        self.separator: str = kwargs.get("separator", "@@")
        self.dropout: float = kwargs.get("dropout", 0.0)

        bpe_parser = apply_bpe.create_parser()
        for action in bpe_parser._actions:  # workaround to ensure utf8 encoding
            if action.dest == "codes":
                action.type = argparse.FileType('r', encoding='utf8')
        bpe_args = bpe_parser.parse_args([
            "--codes", codes_file.as_posix(), "--separator", self.separator
        ])
        self.bpe = apply_bpe.BPE(
            bpe_args.codes,
            bpe_args.merges,
            bpe_args.separator,
            None,
            bpe_args.glossaries,
        )
        self.codes: Path = codes_file

    def __call__(self, raw_input: str, is_train: bool = False) -> List[str]:
        """Tokenize"""
        if raw_input is None:
            return None

        dropout = self.dropout if is_train else 0.0
        tokenized = self.bpe.process_line(raw_input, dropout).strip().split()
        if is_train and self._filter_by_length(len(tokenized)):
            return None
        return tokenized

    def post_process(
        self,
        sequence: Union[List[str], str],
        generate_unk: bool = True,
        cut_at_sep: bool = True
    ) -> str:
        """Detokenize"""
        if isinstance(sequence, list):
            if cut_at_sep:
                try:
                    sep_pos = sequence.index(self.sep_token)  # cut off prompt
                    sequence = sequence[sep_pos:]
                except ValueError as e:  # pylint: disable=unused-variable # noqa: F841
                    pass
            sequence = self._remove_special(sequence, generate_unk=generate_unk)

            # Remove separators, join with spaces
            sequence = self.SPACE.join(sequence
                                       ).replace(self.separator + self.SPACE, "")
            # Remove final merge marker.
            if sequence.endswith(self.separator):
                sequence = sequence[:-len(self.separator)]

        # Moses detokenizer
        if self.pretokenizer == "moses":
            sequence = self.moses_detokenizer.detokenize(sequence.split())

        # Remove extra spaces
        if self.normalize:
            sequence = remove_extra_spaces(sequence)

        # ensure the string is not empty.
        assert sequence is not None and len(sequence) > 0, sequence
        return sequence

    def set_vocab(self, vocab) -> None:
        """Set vocab"""
        # pylint: disable=protected-access
        super().set_vocab(vocab)
        self.bpe.vocab = set(vocab._itos) - set(vocab.specials) - set(vocab.lang_tags)

    def copy_cfg_file(self, model_dir: Path) -> None:
        """Copy config file to model_dir"""
        shutil.copy2(self.codes, (model_dir / self.codes.name).as_posix())

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(level={self.level}, "
            f"lowercase={self.lowercase}, normalize={self.normalize}, "
            f"filter_by_length=({self.min_length}, {self.max_length}), "
            f"pretokenizer={self.pretokenizer}, "
            f"tokenizer={self.bpe.__class__.__name__}, "
            f"separator={self.separator}, dropout={self.dropout})"
        )


def _build_tokenizer(cfg: Dict) -> BasicTokenizer:
    """Builds tokenizer."""
    tokenizer = None
    tokenizer_cfg = cfg.get("tokenizer_cfg", {})

    # assign lang for moses tokenizer
    if tokenizer_cfg.get("pretokenizer", "none") == "moses":
        tokenizer_cfg["lang"] = cfg["lang"]

    if cfg["level"] in ["word", "char"]:
        tokenizer = BasicTokenizer(
            level=cfg["level"],
            lowercase=cfg.get("lowercase", False),
            normalize=cfg.get("normalize", False),
            max_length=cfg.get("max_length", -1),
            min_length=cfg.get("min_length", -1),
            **tokenizer_cfg,
        )
    elif cfg["level"] == "bpe":
        tokenizer_type = cfg.get("tokenizer_type", cfg.get("bpe_type", "sentencepiece"))
        if tokenizer_type == "sentencepiece":
            assert "model_file" in tokenizer_cfg
            tokenizer = SentencePieceTokenizer(
                level=cfg["level"],
                lowercase=cfg.get("lowercase", False),
                normalize=cfg.get("normalize", False),
                max_length=cfg.get("max_length", -1),
                min_length=cfg.get("min_length", -1),
                **tokenizer_cfg,
            )
        elif tokenizer_type == "subword-nmt":
            assert "codes" in tokenizer_cfg
            tokenizer = SubwordNMTTokenizer(
                level=cfg["level"],
                lowercase=cfg.get("lowercase", False),
                normalize=cfg.get("normalize", False),
                max_length=cfg.get("max_length", -1),
                min_length=cfg.get("min_length", -1),
                **tokenizer_cfg,
            )
        else:
            raise ConfigurationError(
                f"{tokenizer_type}: Unknown tokenizer type. "
                "Valid options: {'sentencepiece', 'subword-nmt'}."
            )
    else:
        raise ConfigurationError(
            f"{cfg['level']}: Unknown tokenization level. "
            "Valid options: {'word', 'bpe', 'char'}."
        )
    return tokenizer


def build_tokenizer(cfg: Dict) -> Dict[str, BasicTokenizer]:
    gls_lang = cfg["gls"]["lang"]
    txt_lang = cfg["txt"]["lang"]
    tokenizer = {
        gls_lang: _build_tokenizer(cfg["gls"]),
        txt_lang: _build_tokenizer(cfg["txt"]),
    }
    logger.info("%s tokenizer: %s", gls_lang, tokenizer[gls_lang])
    logger.info("%s tokenizer: %s", txt_lang, tokenizer[txt_lang])
    return tokenizer