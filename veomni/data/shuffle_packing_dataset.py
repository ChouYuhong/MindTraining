# Copyright 2025 Bytedance Ltd. and/or its affiliates
import copy
import random
import traceback
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import IterableDataset

from ..utils import logging
from ..utils.constants import IGNORE_INDEX

logger = logging.get_logger(__name__)


class ShufflePackingDataset(IterableDataset):
    """
    A packing dataset that buffers, shuffles, and packs items into exact `max_seq_len`.
    If an item exceeds the remaining length of the current sample, it is split, 
    and the leftover is used for the next sample.
    """

    def __init__(
        self,
        dataset: IterableDataset,
        max_seq_len: int,
        micro_batch_size: int,
        buffer_size: int,
        collate_fn: Callable,
        seed: int = 42,
        get_length_fn: Optional[Callable] = lambda x: len(x["input_ids"]),
    ) -> None:
        self.dataset = dataset
        self.max_seq_len = max_seq_len
        self.micro_batch_size = micro_batch_size
        self.buffer_size = buffer_size
        self.collate_fn = collate_fn
        self.seed = seed
        self.get_length_fn = get_length_fn

        self._rng = random.Random(self.seed)
        self._data_iter = None
        
        self._buffer = []
        self._working_queue = []
        self._leftover_item = None

    def __iter__(self):
        self._data_iter = iter(self.dataset)

        # Only initialize if they haven't been loaded from state_dict
        if not hasattr(self, "_buffer") or self._buffer is None:
            self._buffer = []
        if not hasattr(self, "_working_queue") or self._working_queue is None:
            self._working_queue = []
        if not hasattr(self, "_leftover_item"):
            self._leftover_item = None

        micro_batch = []
        current_sample_fragments = []
        current_len = 0

        while True:
            try:
                # 1. Fetch next item from leftover, working queue, or buffer
                item = None
                if self._leftover_item is not None:
                    item = self._leftover_item
                    self._leftover_item = None
                elif len(self._working_queue) > 0:
                    item = self._working_queue.pop(0)
                else:
                    # Working queue is empty, fill buffer and shuffle
                    while len(self._buffer) < self.buffer_size:
                        try:
                            next_item = next(self._data_iter)
                            self._buffer.append(next_item)
                        except StopIteration:
                            break
                    
                    if len(self._buffer) == 0:
                        break  # Exhausted

                    self._rng.shuffle(self._buffer)
                    self._working_queue = self._buffer
                    self._buffer = []
                    continue

                # 2. Process item and pack into exactly max_seq_len
                item_len = self.get_length_fn(item)

                if current_len + item_len <= self.max_seq_len:
                    current_sample_fragments.append(item)
                    current_len += item_len

                    if current_len == self.max_seq_len:
                        # Perfect fit
                        merged_sample = self._merge_fragments(current_sample_fragments)
                        micro_batch.append(merged_sample)
                        current_sample_fragments = []
                        current_len = 0
                else:
                    # Item is too long, we need to split it
                    split_idx = self.max_seq_len - current_len
                    part1, part2 = self._split_item(item, split_idx)

                    current_sample_fragments.append(part1)
                    merged_sample = self._merge_fragments(current_sample_fragments)
                    micro_batch.append(merged_sample)

                    # Reset and keep the leftover for the next sample
                    current_sample_fragments = []
                    current_len = 0
                    self._leftover_item = part2

                # 3. Check if micro_batch is full
                if len(micro_batch) == self.micro_batch_size:
                    collated_batch = self.collate_fn(micro_batch)
                    if collated_batch is not None:
                        yield collated_batch
                    else:
                        logger.warning("collate_fn returned None, skip")
                    micro_batch = []

            except Exception as e:
                if isinstance(e, StopIteration):
                    break
                logger.error(f"ShufflePackingDataset iter exception: {e}\n{traceback.format_exc()}")
                raise

    def _split_item(self, item: Dict[str, Any], split_idx: int) -> tuple:
        """Splits an item's 1D tensors/lists into two parts at split_idx."""
        part1 = {}
        part2 = {}
        for k, v in item.items():
            if isinstance(v, torch.Tensor) and v.dim() == 1:
                part1[k] = v[:split_idx]
                part2[k] = v[split_idx:]
            elif isinstance(v, list):
                part1[k] = v[:split_idx]
                part2[k] = v[split_idx:]
            else:
                # Keep scalars (like ds_idx or source_name) in both parts
                part1[k] = v
                part2[k] = v
        return part1, part2

    def _merge_fragments(self, fragments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merges multiple fragments into a single sample and handles boundary IGNORE_INDEX."""
        if len(fragments) == 1:
            return fragments[0]

        merged = {}
        for k in fragments[0].keys():
            if isinstance(fragments[0][k], torch.Tensor) and fragments[0][k].dim() == 1:
                merged[k] = torch.cat([f[k] for f in fragments], dim=-1)
            elif isinstance(fragments[0][k], list):
                merged[k] = sum([f[k] for f in fragments], [])
            else:
                merged[k] = fragments[0][k]

        # Prevent cross-document attention loss pollution
        if "labels" in merged:
            offset = 0
            for i, f in enumerate(fragments):
                length = self.get_length_fn(f)
                if i > 0:
                    if isinstance(merged["labels"], torch.Tensor):
                        merged["labels"][offset] = IGNORE_INDEX
                    elif isinstance(merged["labels"], list):
                        merged["labels"][offset] = IGNORE_INDEX
                offset += length

        return merged

    def state_dict(self):
        """Basic state dict for checkpointing."""
        state = {
            "buffer": copy.deepcopy(self._buffer),
            "working_queue": copy.deepcopy(self._working_queue),
            "leftover_item": copy.deepcopy(self._leftover_item),
            "rng_state": self._rng.getstate(),
        }
        if hasattr(self.dataset, "state_dict"):
            state["upstream_dataset_state"] = self.dataset.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Restore state from checkpoint."""
        self._buffer = state_dict.get("buffer", [])
        self._working_queue = state_dict.get("working_queue", [])
        self._leftover_item = state_dict.get("leftover_item", None)
        if "rng_state" in state_dict:
            self._rng.setstate(state_dict["rng_state"])
        if "upstream_dataset_state" in state_dict and hasattr(self.dataset, "load_state_dict"):
            self.dataset.load_state_dict(state_dict["upstream_dataset_state"])

    def set_epoch(self, epoch: int):
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)