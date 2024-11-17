from dataclasses import dataclass, field
from transformers import TrainingArguments
from peft import LoraConfig
from typing import Optional, Dict, Union, Any

@dataclass
class SFTConfig(TrainingArguments):
    """
    Configuration class for Supervised Fine-Tuning (SFT) tasks, inheriting from `TrainingArguments`.

    This class includes all the parameters from `TrainingArguments` and can be extended with additional parameters specific to SFT tasks.
    """
    output_dir: str = field(
        default='./checkpoints',
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    run_name: str = field(
        default="SFT-1.5B-Instruct-Qwen2.5",
        metadata={"help": "Name of the run for tracking experiments in Weights & Biases."}
    ) 
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "The batch size per device (GPU/XPU/TPU/MPS/NPU core/CPU) for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8,
        metadata={"help": "The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for evaluation."}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Number of updates steps to accumulate the gradients for, before performing a backward/update pass."}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."}
    )
    gradient_checkpointing_kwargs: dict = field(
        default=False,
        metadata={"help": "Key word arguments to be passed to the `gradient_checkpointing_enable` method."}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "The initial learning rate for [`AdamW`] optimizer."}
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "The scheduler type to use. See the documentation of [`SchedulerType`] for all possible values. Possible values['cosine','linear','warmup_stable_decay'], 'warmup_stable_decay' = get_wsd_schedule."}
    )
    lr_scheduler_kwargs: dict = field(
        default=None,
        metadata={'help':'The extra arguments for the lr_scheduler. See the documentation of each scheduler for possible values.'}
    )
    warmup_ratio: float = field(
        default=0.10,
        metadata={"help": "Ratio of total training steps used for a linear warmup from 0 to `learning_rate`."}
    )
    warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of steps used for a linear warmup from 0 to `learning_rate`. Overrides any effect of `warmup_ratio`."}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training. Requires Ampere or higher NVIDIA architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training."}
    )
    eval_strategy: str = field(
        default="steps",
        metadata={"help": "The evaluation strategy to adopt during training. Possible values are: `\"no\"`, `\"steps\"`, `\"epoch\"`."}
    )
    eval_steps: int = field(
        default=0.1,
        metadata={"help": "Number of update steps between two evaluations if `eval_strategy=\"steps\"`. Will default to the same value as `logging_steps` if not set. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps."}
    )
    eval_accumulation_steps: int = field(
        default=None,
        metadata={"help": "Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU."}
    )
    num_train_epochs: float = field(
        default=1.0,
        metadata={"help": "Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training)."}
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`."}
    )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={"help": "If `True`, overwrite the content of the output directory. Use this to continue training if `output_dir` points to a checkpoint directory."}
    )
    dataset_text_field: str = field(
        default="text",
        metadata={"help": "Name of the text field of the dataset. If provided, the trainer will automatically create a `ConstantLengthDataset` based on `dataset_text_field`."}
    )
    log_level: str = field(
        default="info",
        metadata={"help": "Logger log level to use on the main process. Possible choices are the log levels as strings: 'debug', 'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and keeps the current log level for the Transformers library."}
    )
    logging_dir: str = field(
        default='./logging',
        metadata={"help": "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."}
    )
    logging_strategy: str = field(
        default="steps",
        metadata={"help": "The logging strategy to adopt during training. Possible values are: `\"no\"`, `\"epoch\"`, `\"steps\"`."}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Number of update steps between two logs if `logging_strategy=\"steps\"`. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps."}
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to adopt during training. Possible values are: `\"no\"`, `\"epoch\"`, `\"steps\"`."}
    )
    save_steps: int = field(
        default=0.1,
        metadata={"help": "Number of updates steps before two checkpoint saves if `save_strategy=\"steps\"`. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps."}
    )
    save_total_limit: int = field(
        default=None,
        metadata={"help": "If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in `output_dir`. When `load_best_model_at_end` is enabled, the \"best\" checkpoint according to `metric_for_best_model` will always be retained in addition to the most recent ones."}
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "Whether to run training or not. This argument is not directly used by [`Trainer`], it's intended to be used by your training/evaluation scripts instead."}
    )
    do_eval: bool = field(
        default=True,
        metadata={"help": "Whether to run evaluation on the validation set or not. Will be set to `True` if `eval_strategy` is different from `\"no\"`."}
    )
    max_seq_length: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum sequence length for the `ConstantLengthDataset` and for automatically creating the dataset. If `None`, it uses the smaller value between `tokenizer.model_max_length` and `1024`."}
    )
    dataloader_num_workers: Optional[int] = field(
            default=4,
            metadata={'help': 'Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the main process.'}
    )
    dataset_num_proc: Optional[int] = field(
        default=24,
        metadata={"help": "Number of processes to use for processing the dataset. Only used when `packing=False`."}
    )
    dataset_batch_size: Union[int, None] = field(
        default=1000,
        metadata={"help": "Number of examples to tokenize per batch. If `dataset_batch_size <= 0` or `dataset_batch_size is None`, tokenizes the full dataset as a single batch."}
    )
    save_safetensors: bool = field(
        default=True,
        metadata={"help": "Use [safetensors](https://huggingface.co/docs/safetensors) saving and loading for state dicts instead of default `torch.load` and `torch.save`."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training. To ensure reproducibility across runs, use the [`~Trainer.model_init`] function to instantiate the model if it has some randomly initialized parameters."}
    )
    resume_from_checkpoint: str = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model. This argument is not directly used by [`Trainer`], it's intended to be used by your training/evaluation scripts instead."}
    )
    report_to: str = field(
        default="wandb",
        metadata={"help": "The list of integrations to report the results and logs to. Supported platforms are `\"azure_ml\"`, `\"clearml\"`, `\"codecarbon\"`, `\"comet_ml\"`, `\"dagshub\"`, `\"dvclive\"`, `\"flyte\"`, `\"mlflow\"`, `\"neptune\"`, `\"tensorboard\"`, and `\"wandb\"`. Use `\"all\"` to report to all integrations installed, `\"none\"` for no integrations."}
    )
    optim: str = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use, such as `\"adamw_hf\"`, `\"adamw_torch\"`, `\"adamw_torch_fused\"`, `\"adamw_apex_fused\"`, `\"adamw_anyprecision\"`, `\"adafactor\"`. See `OptimizerNames` in [training_args.py](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py) for a full list of optimizers."}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in [`AdamW`] optimizer."}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum gradient norm (for gradient clipping)."}
    )
    packing: bool = field(
        default=False,
        metadata={"help": "Controls whether the `ConstantLengthDataset` packs the sequences of the dataset."}
    )
    model_init_kwargs: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model from a string."}
    )
    dataset_kwargs: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "Dictionary of optional keyword arguments to pass when creating packed or non-packed datasets."}
    )
    eval_packing: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to pack the eval dataset. If `None`, uses the same value as `packing`."}
    )
    num_of_sequences: int = field(
        default=1024,
        metadata={"help": "Number of sequences to use for the `ConstantLengthDataset`."}
    )
    chars_per_token: float = field(
        default=3.6,
        metadata={"help": "Number of characters per token to use for the `ConstantLengthDataset`. See [chars_token_ratio](https://github.com/huggingface/trl/blob/08f550674c553c36c51d1027613c29f14f3676a5/examples/stack_llama/scripts/supervised_finetuning.py#L53) for more details."}
    )
    use_liger: bool = field(
        default=False,
        metadata={"help": "Monkey patch the model with Liger kernels to increase throughput and reduce memory usage."}
    )

