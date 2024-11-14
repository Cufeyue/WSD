from dataclasses import dataclass, field
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    set_seed,
)
from trl import (
    DataCollatorForCompletionOnlyLM,
    SFTTrainer,
)
from sftconfig import SFTConfig  # Ensure this file exists and contains the SFTConfig dataclass

set_seed(42)

@dataclass
class ScriptArguments:
    """
    Dataclass to keep all script arguments.
    """
    checkpoint: str = field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        metadata={"help": "The model checkpoint to use."},
    )

def main():

    parser = HfArgumentParser((SFTConfig, ScriptArguments))
    training_args, script_args = parser.parse_args_into_dataclasses()

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
        script_args.checkpoint,
        quantization_config=bnb_config,
        torch_dtype='auto',
        device_map="auto",
        attn_implementation="sdpa",
        use_cache=False
    )

    tokenizer = AutoTokenizer.from_pretrained(script_args.checkpoint)

    dataset = load_dataset(
        "json",
        data_files={"train": "./datasets/sft/train.json", 
                    "valid": "./datasets/sft/val.json"}
    )


    collator = DataCollatorForCompletionOnlyLM(response_template='assistant\n', instruction_template='user\n', tokenizer=tokenizer)
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

if __name__ == "__main__":
    main()