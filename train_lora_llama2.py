# coding=gbk
#from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
#replace_llama_attn_with_flash_attn()

import sys
 
#sys.setrecursionlimit(10000)
import logging
import os
import typing
import copy
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import datasets
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from transformers import Trainer

from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

logger = logging.getLogger(__name__)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    test_data_path: str = field(
        default=None, metadata={"help": "Path to the testing data."}
    )
    data_cache_path: str = field(
        default=None, metadata={"help": "Path to the cache data."}
    )
    lazy_preprocess: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    flash_attn: bool = False
    fp16: bool = False
    load_best_model_at_end: bool = True
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=2)
    
    logging_steps: int = field(default=200)
    eval_steps: int = field(default=200)
    save_total_limit: int = field(default=3)
    save_steps: int = field(default=200)
    save_strategy: str = field(default="steps")
    evaluation_strategy: str = field(default="steps")
    
    
@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def last_index(lst, value):
    return next((len(lst) - i - 1 for i, x in enumerate(lst[::-1]) if x != value), -1)


def safe_ids(ids, max_value, pad_id):
    return [i if i < max_value else pad_id for i in ids]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
dummy_message = {
    "system": """\
            You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    "id": "dummy_message",
    "conversations": [
            {"from": "human", "value": "Who are you?"},
            {"from": "gpt", "value": "I am your virtual friend."},
            {"from": "human", "value": "What can you do?"},
            {"from": "gpt", "value": "I can chat with you."}
        ]
    }


def tokenize(item, tokenizer):
    roles = {"human": "user", "gpt": "assistant"}
    input_ids = []
    labels = []
    # if "instruction" in item and len(item["instruction"]) > 0:
        # system = item["instruction"]
    # else:
        # system = dummy_message["system"]
    system = dummy_message["system"]
    system = B_SYS + system + E_SYS
    # add system before the first content in conversations
    item["conversations"][0]['value'] = system + item["conversations"][0]['value']
    for i, turn in enumerate(item["conversations"]):
        role = turn['from']
        content = turn['value']
        content = content.strip()
        if role == 'human':
            content = f"{B_INST} {content} {E_INST} "
            content_ids = tokenizer.encode(content)
            labels += [IGNORE_TOKEN_ID] * (len(content_ids))
        else:
            # assert role == "gpt"
            content = f"{content} "
            content_ids = tokenizer.encode(content, add_special_tokens=False) + [tokenizer.eos_token_id]   # add_special_tokens=False remove bos token, and add eos at the end
            labels += content_ids
        input_ids += content_ids

    input_ids = input_ids[:tokenizer.model_max_length]
    labels = labels[:tokenizer.model_max_length]

    trunc_id = last_index(labels, IGNORE_TOKEN_ID) + 1
    input_ids = input_ids[:trunc_id]
    labels = labels[:trunc_id]
    
    if len(labels) == 0:
        return tokenize(dummy_message, tokenizer)
        
    input_ids = safe_ids(input_ids, tokenizer.vocab_size, tokenizer.pad_token_id)
    labels = safe_ids(labels, tokenizer.vocab_size, IGNORE_TOKEN_ID)
    
    return input_ids, labels

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        item = self.raw_data[i]
        input_ids, labels = tokenize(
            copy.deepcopy(item),
            self.tokenizer)
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        ret = dict(
            input_ids=input_ids,
            labels=labels,
        )
        self.cached_data_dict[i] = ret

        return ret


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        
        self.input_ids = []
        self.labels = []
        
        for example in raw_data:
            input_ids, labels = tokenize(copy.deepcopy(example), tokenizer)
            
            input_ids = torch.tensor(input_ids)
            labels = torch.tensor(labels)
            
            self.input_ids.append(input_ids)
            self.labels.append(labels)
        
        assert len(self.input_ids) == len(self.labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
        )

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_TOKEN_ID)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)
    
    test_json = json.load(open(data_args.test_data_path, "r"))
    test_dataset = dataset_cls(test_json, tokenizer=tokenizer)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    return dict(train_dataset=train_dataset,
                eval_dataset=test_dataset,
                data_collator=data_collator
                )




def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


def train():
    # global local_rank  

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    
    
    if training_args.flash_attn:
        replace_llama_attn_with_flash_attn()
        
        
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

            
            
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        load_in_8bit = True,
        device_map=device_map,
        # torch_dtype=torch.float16,
    )
    print(f"====First====:")
    print_trainable_parameters(model)
    

    # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    model = prepare_model_for_int8_training(model)
    
    lora_config = LoraConfig(
        r = lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    
    print(lora_config)
    
    model = get_peft_model(model, lora_config)
    
    print(f"====Peft====:")  # 这个模型有内置函数print_trainable_parameters
    # model.print_trainable_parameters()
    print_trainable_parameters(model)
    
    if training_args.flash_attn:
        for name, module in model.named_modules():
            if "norm" in name:
                module = module.to(compute_dtype)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    module = module.to(compute_dtype)
  
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
  
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token   

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)


    trainer = transformers.Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    model.config.use_cache = False

    if training_args.deepspeed or training_args.gradient_checkpointing:
        model.config.use_cache = False
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    print("Ready to save model")
    trainer.save_pretrained(training_args.output_dir)
    print(f"Model saved to {training_args.output_dir}.")

if __name__ == "__main__":
    train()
