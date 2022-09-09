import math
import torch
import numpy as np
import more_itertools
from torch.utils.data import DataLoader
from torch.utils.data import Sampler, SubsetRandomSampler


class RepeatedSampler(Sampler):
    """A sampler that repeats the data n times"""

    def __init__(self, sampler, n_repeat):
        super().__init__(sampler)

        self._sampler = sampler
        self.n_repeat = n_repeat

    def __iter__(self):
        for i in range(self.n_repeat):
            for elem in self._sampler:
                yield elem

    def __len__(self):
        return len(self._sampler) * self.n_repeat


class RepeatedDataLoader(DataLoader):
    """A data loader that returns an iterator cycling through data n times"""

    def __init__(self, *args, n_repeat=1, **kwargs):
        super().__init__(*args, **kwargs)
        if n_repeat != 1:
            self._DataLoader__initialized = False  # this is an ugly hack for pytorch1.3 to be able to change the attr
            self.batch_sampler = RepeatedSampler(self.batch_sampler, n_repeat)
            self._DataLoader__initialized = True


class LengthSortBatchSamplerWithFirstMaxLength(SubsetRandomSampler):
    def __init__(
        self,
        indices,
        batch_size,
        lengths,
        is_distributed=False,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
    ):
        super().__init__(indices)

        self.is_distributed = is_distributed
        self.batch_size = batch_size
        self.lengths = lengths
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        if self.is_distributed:
            if num_replicas is None:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")
                num_replicas = dist.get_world_size()
            if rank is None:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")
                rank = dist.get_rank()
            if rank >= num_replicas or rank < 0:
                raise ValueError(
                    "Invalid rank {}, rank should be in the interval"
                    " [0, {}]".format(rank, num_replicas - 1)
                )

            self.num_replicas = num_replicas
            self.rank = rank
            self.num_samples = math.ceil(self.__len__() / num_replicas)
            self.total_size = self.num_samples * num_replicas

    def __len__(self):
        num_batches = len(self.lengths) // self.batch_size
        if len(self.lengths) % self.batch_size != 0 and not self.drop_last:
            num_batches += 1
        return num_batches

    def __iter__(self):
        # construct batches
        # indices = [(i, length) for i, length in enumerate(self.lengths)]
        filtered_indices = [
            (i, self.lengths[idx]) for i, idx in enumerate(self.indices)
        ]
        sorted_indices_and_length = sorted(filtered_indices, key=lambda x: x[1])
        sorted_indices = [tup[0] for tup in sorted_indices_and_length]

        batches = list(more_itertools.chunked(sorted_indices, self.batch_size))

        if self.drop_last and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]

        # the batch having the longest item; w/ max length it will be first to approach OOM problem
        longest_batch = [batches[-1]]

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            # shuffle batch indices
            shuffled_batches = np.array(batches[:-1])[
                torch.randperm(len(batches) - 1, generator=g).tolist()
            ].tolist()
            batches = longest_batch + shuffled_batches
        else:
            batches = longest_batch + batches[:-1]

        if self.is_distributed:
            assert len(batches) == self.total_size

            # subsample
            batches = batches[self.rank : self.total_size : self.num_replicas]
            assert len(batches) == self.num_samples

        return iter(batches)

    def set_epoch(self, epoch: int):
        self.epoch = epoch


class ImportanceSamplingBatchSampler(Sampler):
    def __init__(self, trajectories, batch_size, drop_last=False):
        self.trajectories = trajectories
        self.batch_size = batch_size
        self.drop_last = drop_last
        # use the return_to_go for each trajectory as importance weighting
        returns_to_go = [traj["returns_to_go"][0] for traj in trajectories]
        self.importance_weight = returns_to_go / sum(returns_to_go)

    def __len__(self):
        num_batches = len(self.trajectories) // self.batch_size
        if len(self.trajectories) % self.batch_size != 0 and not self.drop_last:
            num_batches += 1
        return num_batches

    def __iter__(self):
        # sample batch size of indices based on importance weight
        batch_inds = np.random.choice(
            np.arange(len(self.trajectories)),
            size=min(len(self.trajectories), self.batch_size),
            replace=True,
            p=self.importance_weight,
        )
        return iter([batch_inds])
