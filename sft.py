import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Union, Any
from tqdm import tqdm
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import(
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    set_seed,
    TrainingArguments
)
from trl import(
    DataCollatorForCompletionOnlyLM,
    SFTTrainer,
)

from .sftconfig import SFTConfig

set_seed(42)

checkpoint = "Qwen/Qwen2.5-1.5B-Instruct"

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
)

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ],
    lora_dropout=0.0,
    bias="none",
    task_type='CAUSAL_LM'
)

model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    quantization_config=bnb_config,
    torch_dtype='auto',
    device_map="auto",
    attn_implementation="sdpa",
    use_cache=False
)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# tokenizer.pad_token
# tokenizer.chat_template 
# tokenizer.padding_side = "right"

dataset = load_dataset(
    "json",
    data_files={"train": "./datasets/sft/train.json", 
                "valid": "./datasets/sft/val.json"}
)

parser = HfArgumentParser((SFTConfig))
training_args, = parser.parse_args_into_dataclasses()

collator = DataCollatorForCompletionOnlyLM(response_template='<|im_start|>assistant\n', instruction_template='<|im_start|>user\n', tokenizer=tokenizer)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    processing_class=tokenizer,
    data_collator=collator,
    peft_config=peft_config,
)

trainer.train()
trainer.save_model(training_args.output_dir)