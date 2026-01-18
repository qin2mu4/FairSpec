#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os


import sys
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import datasets
import torch
from build_dataset import build_instruction_dataset, DataCollatorForSupervisedDataset
import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    Qwen2ForCausalLM,
    Qwen2Tokenizer,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed, DataCollatorWithPadding, EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict
from peft.tuners.lora import LoraLayer

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from utils import compute_metrics

from trl import DPOConfig, DPOTrainer, OnlineDPOTrainer, PairRMJudge, OnlineDPOConfig
from datasets import load_dataset
import swanlab


swanlab_log_path = './swanlab_log/log'
os.makedirs(swanlab_log_path, exist_ok=True)
swanlab.init(
    logdir=swanlab_log_path,
    mode="local",
    project="",
)
port = '12332'
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "sft_lora_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "sft_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        # kwargs["tokenizer"].save_pretrained(peft_model_path)
        kwargs["processing_class"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(args.output_dir, "sft_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        # kwargs["tokenizer"].save_pretrained(peft_model_path)
        kwargs["processing_class"].save_pretrained(peft_model_path)


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    r"""
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    # cast all non INT8/INT4 parameters to fp32
    for param in model.parameters():
        if ((param.dtype == torch.float16) or (param.dtype == torch.bfloat16)) and loaded_in_kbit:
            param.data = param.data.to(torch.float32)

    for name, module in model.named_modules():
        if 'norm' in name:
            module = module.to(torch.float32)

    if loaded_in_kbit and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, _input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    return model


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )


    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    data_cache_dir: Optional[str] = field(default=None, metadata={"help": "The datasets processed stored"})

    max_seq_length: Optional[int] = field(default=1024)


@dataclass
class MyTrainingArguments(DPOConfig):
    trainable : Optional[str] = field(default="q_proj,v_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_dropout : Optional[float] = field(default=0.1)
    lora_alpha : Optional[float] = field(default=32.)
    modules_to_save : Optional[str] = field(default=None)
    peft_path : Optional[str] = field(default=None)
    flash_attn : Optional[bool] = field(default=False)
    double_quant: Optional[bool] = field(default=True)
    quant_type: Optional[str] = field(default="nf4")
    load_in_kbits: Optional[int] = field(default=16)
    
    lora_nums: Optional[int] = field(default=2)
    blc_alpha: Optional[float] = field(default=0.0)
    blc_weight: Optional[float] = field(default=0.0)

    fairness_coef: Optional[float] = field(default=0.0)

    model_name: Optional[str] = field(default='llama2')
    fair_e: Optional[str] = field(default='')
    sens_size: Optional[str] = field(default='')


logger = logging.getLogger(__name__)
os.environ['MASTER_ADDR'] = 'localhost'
port_success = False
while not port_success:
    try:
        os.environ['MASTER_PORT'] = port
        torch.distributed.init_process_group(backend='nccl',rank=0, world_size = 1)
        port_success = True
        print(f'port: {port} initialized successfully')
    except:
        port = str(int(port)+1)


def main():


    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],)


    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    training_args.do_train = True

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print('last_checkpoint',last_checkpoint)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "bos_token": '<s>',
        "eos_token": '</s>',
        "unk_token": '<unk>',
        "pad_token": '[PAD]'
    }
    
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.tokenizer_name_or_path:
        if training_args.model_name == 'qwen25':
            tokenizer = Qwen2Tokenizer.from_pretrained(model_args.tokenizer_name_or_path, **tokenizer_kwargs)
            causal_model_class = Qwen2ForCausalLM
        elif training_args.model_name == 'llama2':
            tokenizer = LlamaTokenizer.from_pretrained(model_args.tokenizer_name_or_path, **tokenizer_kwargs)
            causal_model_class = LlamaForCausalLM
        elif 'llama3' in training_args.model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path, **tokenizer_kwargs)
            causal_model_class = LlamaForCausalLM
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    eval_dataset=None

    logger.info(f"Training files: {' '.join(data_args.dataset_dir)}")
    train_dataset = load_dataset(data_args.dataset_dir, split="train")
    if training_args.do_eval:
        split_dataset = train_dataset.train_test_split(test_size=0.1)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
        logger.info(f"Num eval_samples  {len(eval_dataset)}")
    logger.info(f"Num train_samples  {len(train_dataset)}")
    logger.info(f"Training example: {train_dataset[0]}")

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    if training_args.load_in_kbits in [4, 8]:
        load_in_4bit = training_args.load_in_kbits == 4
        load_in_8bit = training_args.load_in_kbits == 8
        if training_args.modules_to_save is not None:
            load_in_8bit_skip_modules = training_args.modules_to_save.split(',')
        else:
            load_in_8bit_skip_modules = None
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=training_args.load_in_kbits == 4,
            load_in_8bit=training_args.load_in_kbits == 8,
            llm_int8_threshold=6.0,
            load_in_8bit_skip_modules=load_in_8bit_skip_modules,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
        )
    else:
        load_in_4bit = False
        load_in_8bit = False
        quantization_config = None
    if quantization_config is not None:
        logger.info(f"quantization_config:{quantization_config.to_dict()}")

    config.fair_e = training_args.fair_e
    config.sens_size = training_args.sens_size
    model = causal_model_class.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        torch_dtype=torch_dtype,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        quantization_config=quantization_config,
    )
    model.enable_input_require_grads()
    if training_args.load_in_kbits in [4, 8]:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    model.config.use_cache = False

    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    logger.info(f"Model vocab size: {model_vocab_size}")
    logger.info(f"len(tokenizer):{len(tokenizer)}")
    if model_vocab_size != len(tokenizer):
        logger.info(f"Resize model vocab size to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    if training_args.peft_path is not None: # --------------------------> train from the trained lora model
        logger.info("Peft from pre-trained model")

        peft_model = PeftModel.from_pretrained(model, training_args.peft_path,
            # device_map=device_map
            )
    else: # --------------------------> train from the sketch
        logger.info("Init new peft model") 
        target_modules = training_args.trainable.split(',') # lora paras
        modules_to_save = training_args.modules_to_save # not lora paras, but is trainable, i.e., not freeze
        if modules_to_save is not None:
            modules_to_save = modules_to_save.split(',')
        lora_rank = training_args.lora_rank
        lora_dropout = training_args.lora_dropout
        lora_alpha = training_args.lora_alpha
        
        lora_nums = training_args.lora_nums
        blc_alpha = training_args.blc_alpha
        blc_weight = training_args.blc_weight
        
        
        logger.info(f"target_modules: {target_modules}")
        logger.info(f"lora_rank: {lora_rank}")
        logger.info(f"lora_nums: {lora_nums}")
        logger.info(f"blc_alpha: {blc_alpha}")
        logger.info(f"blc_weight: {blc_weight}")

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=lora_rank, 
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_nums=lora_nums,
            blc_alpha=blc_alpha,
            blc_weight=blc_weight,
            modules_to_save=modules_to_save,
            fair_e = training_args.fair_e,
            sens_size = training_args.sens_size
        )
        peft_model = get_peft_model(model, peft_config)

    if training_args.gradient_checkpointing and \
        (not peft_model.modules_to_save or 'embed_tokens' not in peft_model.modules_to_save):
        # enable requires_grad to avoid exception during backward pass when using gradient_checkpoint without tuning embed.
        if hasattr(peft_model.base_model, "enable_input_require_grads"):
            peft_model.base_model.enable_input_require_grads()
        elif hasattr(peft_model.base_model, "get_input_embeddings"):
            def make_inputs_require_grad(_module, _input, _output):
                _output.requires_grad_(True)
            peft_model.base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    for name, module in peft_model.named_modules():
        if isinstance(module, LoraLayer):
            if training_args.bf16:
                module = module.to(torch.bfloat16)
            if training_args.fp16:
                module = module.to(torch.float16)
        if 'norm' in name:
            module = module.to(torch.float16)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
                if training_args.fp16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.float16)
    peft_model.print_trainable_parameters()
    logger.info(f"peft_model.modules_to_save: {peft_model.modules_to_save}")
    old_state_dict = peft_model.state_dict
    peft_model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(peft_model, type(peft_model))


    training_args.remove_unused_columns = False

    trainer = DPOTrainer(
        model=peft_model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        print('\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n')
        print(training_args.output_dir)



if __name__ == "__main__":
    main()
