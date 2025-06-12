import os
import textwrap
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from unittest.mock import patch

import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather_object
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from typing import Union
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
from transformers.utils.import_utils import _is_package_available

_vllm_available = _is_package_available("vllm")

if _vllm_available:
    from vllm import LLM, SamplingParams


RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class GRPOTrainerStopSequences(GRPOTrainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        *otherargs,
        stop=["**Finished.**", "Human:"],
        **kwargs,
    ):
        super().__init__(model, reward_funcs, args, *otherargs, **kwargs)
        self.stop = stop
        # overwrite self sampling params
        self.sampling_params = SamplingParams(
            n=self.num_generations,
            temperature=args.temperature,
            max_tokens=self.max_completion_length,
            stop=stop,
        )
