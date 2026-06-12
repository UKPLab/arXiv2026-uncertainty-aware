import copy
import dataclasses
import itertools
import json
import logging
import os
import subprocess
import sys

args = sys.argv
cache_dir = args[-1]
os.environ["TRANSFORMERS_CACHE"] = '/path/to/.cache/huggingface/transformers/'
os.environ["HF_HOME"] = '/path/to/.cache/huggingface/transformers/'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["WANDB_API_KEY"] = "TODO"
os.environ["VLLM_CACHE_ROOT"] = cache_dir
os.environ["VLLM_CONFIG_ROOT"] = cache_dir
os.environ["FLASHINFER_WORKSPACE_BASE"] = cache_dir
os.environ["TRITON_CACHE_DIR"] = cache_dir
os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir

os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["TRANSFORMERS_CACHE"] = '/path/to/.cache/huggingface/transformers/'
os.environ["HF_HOME"] = '/path/to/.cache/huggingface/transformers/'
os.environ["HF_HUB_HOME"] = '/path/to/.cache/huggingface/transformers/'
os.environ["HF_HUB_CACHE"] = '/path/to/.cache/huggingface/transformers/hub'
os.environ["HF_DATASETS_CACHE"] = '/path/to/.cache/huggingface/transformers/datasets'
os.environ["HF_TOKEN"] = "TODO"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
from enum import Enum

import torch
import transformers
import wandb

from sacrebleu.metrics import BLEU
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    # Trainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    get_linear_schedule_with_warmup,
    AutoModelForSequenceClassification,
    get_constant_schedule_with_warmup
)
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers.trainer_utils import is_main_process, PredictionOutput

from arxiv.arguments import *
from arxiv.methods.base import Method
from arxiv.methods.classification import SequenceClassificationMethod
from arxiv.methods.generation import CausalSeq2SeqMethod, Seq2SeqMethod, MathDialSampleMethod, \
 ReviewGenerationDirectMethod, ReviewScorePredictionMethod, ReviewScoreConsistencyMethod, ReviewQualityMethod, \
 ReviewMBRMethod, EvalReviewsMethod, DialogActPredictionMethod, MathdialQualityMethod, MathDialMBR, EvalMathDialMethod, \
 MathDialCalibrationMethod, MathDialMinMax, ReviewCalibrationMethod, ReviewMinMaxMethod, ReviewScorePredictionNoOracleMethod
from arxiv.optimizers.lr_schedulers import CosineAnnealingWarmupRestarts, \
 get_inverse_square_root_schedule_with_warmup
from arxiv.tokenization.character_tokenizer import CharacterTokenizer
from utils import NumpyEncoder

# import wandb
# wandb.init(mode="disabled")

logging.basicConfig(stream=sys.stdout, level=logging.NOTSET)
logger = logging.getLogger(__name__)
# os.environ["WANDB_DISABLED"] = "true"
# os.environ['WANDB_MODE'] = 'disabled'


method_classes = [
    CausalSeq2SeqMethod,
    SequenceClassificationMethod,
    Seq2SeqMethod,
    MathDialSampleMethod,
    ReviewGenerationDirectMethod,
    ReviewScorePredictionMethod,
    ReviewScoreConsistencyMethod,
    ReviewQualityMethod,
    ReviewMBRMethod,
    EvalReviewsMethod,
    DialogActPredictionMethod,
    MathdialQualityMethod,
    MathDialMBR,
    EvalMathDialMethod,
    MathDialCalibrationMethod,
    MathDialMinMax,
    ReviewCalibrationMethod,
    ReviewMinMaxMethod,
    ReviewScorePredictionNoOracleMethod
]

optimizer_map = {
    "AdamW": torch.optim.AdamW,
}


def _setup_logging(training_args: TrainingArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.setLevel(logging.INFO if is_main_process(
        training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)


def get_config_class(model_args, optimizer_args):
    return AutoConfig

def get_lr_scheduler(optimizer, optimizer_args, training_args, dataset, data_args):
    max_steps = (len(dataset) * training_args.num_train_epochs) // (training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size)
    if data_args.is_training:
        if data_args.lr_scheduler == "cosine":
            return get_cosine_schedule_with_warmup(
                optimizer,
                training_args.warmup_steps,
                max_steps,
            )
        elif data_args.lr_scheduler == "linear":
            return get_linear_schedule_with_warmup(
                optimizer,
                training_args.warmup_steps,
                max_steps
            )
        elif data_args.lr_scheduler == "constant":
            return get_constant_schedule_with_warmup(
                optimizer,
                training_args.warmup_steps,
                last_epoch=-1
            )
        else:
            raise NotImplementedError
    else:
        return None

def get_optimizer(model, optimizer_args, training_args, training_data):
    old_model = None
    optimizer_class = optimizer_map[optimizer_args.optimizer_name]

    if optimizer_class == torch.optim.AdamW:
        model.bfloat16()
        optimizer = torch.optim.AdamW(
            [p for n, p in model.named_parameters() if p.requires_grad],
            lr=training_args.learning_rate,
            betas=(optimizer_args.beta1, optimizer_args.beta2),
            eps=optimizer_args.eps,
            weight_decay=training_args.weight_decay
        )
    else:
        raise NotImplementedError()

    return optimizer, old_model

def get_tokenizer_class(config, model_args):
    return AutoTokenizer

def get_tokenizer_name(config, model_args):
    if model_args.tokenizer_name:
        return model_args.tokenizer_name
    else:
        return model_args.model_name_or_path


class RunMode(Enum):
    TRAIN = 1
    PREDICT = 2


def main(run_mode: RunMode):
    training_args_class = Seq2SeqTrainingArguments
    parser_arguments = (ModelArguments, DataTrainingArguments if run_mode ==
                                                                 RunMode.TRAIN else DataPredictionArguments,
                        OptimizerArguments,
                        training_args_class)
    parser = HfArgumentParser(parser_arguments)

    raw_args = sys.argv[1:-1] # -1 is the cache path!
    json_index = -1 if raw_args[-1].endswith(".json") and (len(
        raw_args) == 1 or not raw_args[-2].startswith('-') or '=' in raw_args[-2]) else 0
    if len(raw_args) > 0 and raw_args[json_index].endswith(".json"):
        with open(raw_args[json_index]) as fp:
            json_args_dict = json.load(fp)
        del raw_args[json_index]

        if run_mode == RunMode.TRAIN:
            train_parser = HfArgumentParser(training_args_class)
            training_args_dict = vars(train_parser.parse_args(
                raw_args + ['--output_dir', json_args_dict['output_dir']]))
            training_args_dict.update(json_args_dict)
            json_args_dict = training_args_dict

        model_args, data_args, optimizer_args, training_args = parser.parse_dict(
            json_args_dict, allow_extra_keys=True)
    else:
        model_args, data_args, optimizer_args, training_args = parser.parse_args_into_dataclasses()

    logging.info(data_args)

    logging.info(
        f"My rank is {training_args.local_rank} with {torch.cuda.device_count()} GPUs.")
    if training_args.local_rank != -1:
        torch.cuda.set_device(training_args.local_rank)

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    _setup_logging(training_args)

    config_class = get_config_class(model_args, optimizer_args)

    config = config_class.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
    )
    if model_args.num_labels is not None:
        config.num_labels = model_args.num_labels


    tokenizer_class = get_tokenizer_class(config, model_args)
    tokenizer = tokenizer_class.from_pretrained(
        get_tokenizer_name(config, model_args),
        padding_side="left",
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    method_class = next(
        (m for m in method_classes if m.name == model_args.method), None)
    if method_class is None:
        raise Exception(f"No method class for name {model_args.method}.")
    method_definition: Method = method_class(
        model_args, data_args, optimizer_args, config, tokenizer)
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # here we let VLLM do the heavy lifting and dont load models twice or so
    model = method_definition.get_model(run_mode, config).to(training_args.device) if run_mode == RunMode.TRAIN else None
    if model is not None:
        model.config.keys_to_ignore_at_inference = [
            "decoder_attentions"
        ]

        model.config.dropout = model_args.dropout

    if run_mode == RunMode.TRAIN:
        extra_trainer_args = {
            'train_dataset': method_definition.get_train_dataset(),
            'eval_dataset': method_definition.get_validation_dataset(),
        }

    data_collator = method_definition.get_data_collator()
    trainer_class = method_definition.get_trainer_class()

    # if run_mode == RunMode.TRAIN:
    if run_mode == RunMode.TRAIN:
        optimizer, old_model = get_optimizer(model, optimizer_args, training_args, extra_trainer_args["train_dataset"])
        lr_scheduler = get_lr_scheduler(optimizer, optimizer_args, training_args, extra_trainer_args["train_dataset"], data_args)
    else:
        optimizer, old_model = None, None #get_optimizer(model, optimizer_args, training_args, extra_trainer_args["eval_dataset"])
        lr_scheduler = None#get_lr_scheduler(optimizer, optimizer_args, training_args, extra_trainer_args["eval_dataset"], data_args)

    if run_mode == RunMode.TRAIN:
        trainer = trainer_class(
            model=model,
            args=training_args,
            tokenizer=method_definition.tokenizer,
            data_collator=data_collator,
            compute_metrics=method_definition.compute_metrics,
            optimizers=(optimizer, lr_scheduler),
            method=method_definition,
            optimizer_args=optimizer_args,
            old_model=old_model,
            model_args=model_args,
            data_args=data_args,
            predict=run_mode==RunMode.PREDICT,
            **extra_trainer_args,
        )
        train_result = trainer.train()

        trainer.save_model(trainer.args.output_dir)

        test_dataset = method_definition.get_validation_dataset()
        test_dataloader = DataLoader(
            test_dataset, 
            shuffle=False, 
            batch_size=trainer.args.eval_batch_size,
            collate_fn=trainer.data_collator
        )
        results = trainer.evaluate(test_dataloader)
        print(results)
        for k in list(results.keys()):
            if "_test" not in k:
                k_new = k + "_test"
                results[k_new] = results[k]
                del results[k]
        wandb.log(results)


    elif run_mode == RunMode.PREDICT:
        test_dataset = method_definition.get_test_dataset()

        if method_definition.name not in ["review_mbr", "eval_reviews", "mathdial_mbr", "eval_mathdial", "mathdial_calibration", "mathdial_minmax", "review_calibration", "review_minmax"]:
            extra_trainer_args = {
                'eval_dataset': test_dataset
            }
            trainer = trainer_class(
                model=model,
                args=training_args,
                tokenizer=method_definition.tokenizer,
                data_collator=data_collator,
                compute_metrics=method_definition.compute_metrics,
                optimizers=(optimizer, lr_scheduler),
                method=method_definition,
                optimizer_args=optimizer_args,
                old_model=old_model,
                model_args=model_args,
                data_args=data_args,
                predict=run_mode==RunMode.PREDICT,
                **extra_trainer_args,
            )
            results = trainer.predict(test_dataset)
        else:
            results = method_definition.predict(test_dataset)
        if method_definition.name == "review_score_prediction_no_oracle":
            results = method_definition.postprocess_scores(results)

        if data_args.prediction_output_file is not None:
            with open(data_args.prediction_output_file, 'wt') as f:
                try:
                    json.dump(
                        dataclasses.asdict(results) if type(
                            results) == PredictionOutput else results,
                        f,
                        cls=NumpyEncoder
                    )
                except:
                    json.dump({}, f)
