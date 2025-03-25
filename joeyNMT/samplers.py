# samplers.py

import random
from torch.utils.data import Sampler, Dataset
from typing import List, Optional, Iterator

class SentenceBatchSampler(Sampler[List[int]]):
    def __init__(self, data_source: Dataset, batch_size: int, drop_last: bool, seed: int, sampler: Sampler[int]):
        """
        Custom batch sampler for sentence-based batching.

        :param data_source: The dataset to sample from.
        :param batch_size: Number of samples per batch.
        :param drop_last: Whether to drop the last batch if it's incomplete.
        :param seed: Random seed for shuffling.
        :param sampler: Sampler to generate indices.
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.sampler = sampler
        self.batches = self.create_batches()

    def set_seed(self, seed: int):
        """Set the seed for random operations."""
        self.seed = seed
        random.seed(self.seed)
        self.batches = self.create_batches()

    def create_batches(self) -> List[List[int]]:
        """Create batches of indices."""
        indices = list(self.sampler)
        if self.drop_last:
            indices = indices[:len(indices) - len(indices) % self.batch_size]
        # Sort by sentence length for bucketing
        indices = sorted(indices, key=lambda x: len(self.data_source[x]['sgn']))
        # Create batches
        batches = [
            indices[i:i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]
        return batches
    
    def get_state(self):
        return None

    def set_state(self, state):
        pass

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)



class TokenBatchSampler(Sampler):
    """
    배치 내 총 토큰 수를 기준으로 샘플을 배치로 묶어주는 배치 샘플러입니다.

    이 샘플러는 각 배치 내의 총 토큰 수가 설정된 `batch_size`를 초과하지 않도록
    샘플들을 동적으로 묶어, 메모리 사용을 최적화하고 모델 학습의 효율성을 높입니다.
    """
    def __init__(self, data_source, batch_size, drop_last=False, seed=None):
        """
        TokenBatchSampler를 초기화합니다.

        :param data_source: 샘플링할 데이터셋. `torch.utils.data.Dataset`을 상속받은 객체이어야 합니다.
        :param batch_size: 배치 내 최대 토큰 수.
        :param drop_last: 마지막 배치가 불완전할 경우 드롭할지 여부.
        :param seed: 데이터 셔플링을 위한 랜덤 시드. 동일한 시드를 사용하면 재현 가능한 배치 구성이 가능합니다.
        """
        self.data_source = data_source
        self.batch_size = batch_size  # 총 토큰 수
        self.drop_last = drop_last
        self.seed = seed
        self.batches = self.create_batches()

    def create_batches(self):
        """
        데이터셋의 샘플들을 토큰 수에 따라 동적으로 배치로 그룹화합니다.

        :return: 인덱스의 리스트를 요소로 가지는 배치 리스트.
        """
        if self.seed is not None:
            random.seed(self.seed)
            random.shuffle(self.data_source)
        batches = []
        current_batch = []
        current_tokens = 0
        for idx in range(len(self.data_source)):
            tokens = len(self.data_source[idx])
            if current_tokens + tokens > self.batch_size:
                if len(current_batch) > 0:
                    batches.append(current_batch)
                current_batch = [idx]
                current_tokens = tokens
            else:
                current_batch.append(idx)
                current_tokens += tokens
        if len(current_batch) > 0 and not self.drop_last:
            batches.append(current_batch)
        return batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

class BatchSampler(Sampler[List[int]]):
    """
    Default Batch Sampler

    이 샘플러는 고정된 배치 크기를 사용하여 데이터를 배치로 묶습니다.
    일반적인 상황에서 사용되며, 사용자 정의 배치 샘플러를 사용하지 않을 때 유용합니다.
    """
    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool = False):
        """
        BatchSampler를 초기화합니다.

        :param sampler: 인덱스를 제공하는 Sampler 객체.
        :param batch_size: 배치당 샘플 수.
        :param drop_last: 마지막 배치가 불완전할 경우 드롭할지 여부.
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        total = len(self.sampler)
        if self.drop_last:
            return total // self.batch_size
        else:
            return (total + self.batch_size - 1) // self.batch_size