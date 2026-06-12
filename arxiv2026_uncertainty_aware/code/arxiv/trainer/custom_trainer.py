import contextlib
import copy
import io
import logging
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch
import wandb
from huggingface_hub import login
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)



wandb.login()

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

class SimpleTrainer(object):

    def __init__(
        self,
        model=None,
        args=None,
        tokenizer=None,
        data_collator=None,
        compute_metrics=None,
        optimizers=None,
        train_dataset=None,
        eval_dataset=None,
        method=None,
        old_model=None,
        optimizer_args=None,
        model_args=None,
        data_args=None,
        predict=False
    ):
        self.args = args
        self.optimizer_args = optimizer_args
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
        self.optimizer = optimizers[0]
        self.lr_scheduler = optimizers[1]
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.method = method

        self.model_args = model_args
        self.data_args = data_args

        self.training_step = 0
        self.losses = []

        if not predict:
            self.wandb_init(args, optimizer_args, model_args, data_args)

        self.g = torch.Generator()
        self.g.manual_seed(0)

        self.train_fn = self.optimizer_args.train_fn
        
        # login(token=self.model_args.use_auth_token)

        if predict:
            self.vllm_engine = LLM(
                model=self.model_args.model_name_or_path,
                tokenizer=self.model_args.tokenizer_name,
                dtype="bfloat16", 
                tensor_parallel_size=1, # num gpus
                gpu_memory_utilization=0.9, 
                max_num_seqs=args.per_device_train_batch_size,
                max_model_len=self.model_args.max_input_length, # max input length, there is no truncation inside vllm!!!
                seed=self.model_args.manual_seed,
                trust_remote_code=True
            )
        else:
            self.model = model.bfloat16() # this is already done when the model is loaded
            if self.optimizer_args.sft_warmup_steps > 0:
                self.train_fn = "sft"

        self.sampling_params = SamplingParams(
            n=self.model_args.num_return_sequences, 
            temperature=self.model_args.generation_temperature, 
            max_tokens=self.model_args.generation_max_len,
            logprobs=0,
            top_k=-1,
            top_p=1.0,
        )
        print(self.sampling_params)

    def seed_worker(self, worker_id):
        import random
        worker_seed = 0 #torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def wandb_init(self, args, optimizer_args, model_args, data_args):
        exp_name = str(data_args.dataset_name.split("/")[-1].replace(".py", "")) + "-" + optimizer_args.optimizer_name.replace("_SVRG", "")
        wandb.init(
            project=model_args.wandb_project,
            name=exp_name,
            config={
                "learning_rate": args.learning_rate,
                "architecture": model_args.model_name_or_path,
                "dataset": str(data_args.dataset_name.split("/")[-1].replace(".py", "")),
                "epochs": args.num_train_epochs,
                "method": model_args.method,
                "optimizer": optimizer_args.optimizer_name,
                "batch_size": args.per_device_train_batch_size,
                "clip_radius": optimizer_args.clip_radius,
                "ess": optimizer_args.ess
            }
        )

    def _prepare_batch(self, batch, is_eval=False):
        if self.train_fn == "sft" and not is_eval:
            batch["input_ids"] = batch["labels"] # [:,:-1]
        batch["attention_mask"] = (batch["input_ids"] != -100).int() * (batch["input_ids"] != self.tokenizer.pad_token_id).int()
        for k, v in batch.items():
            if hasattr(v, "to"):
                batch[k] = v.to(self.model.device)
        batch["input_ids"][batch["input_ids"] == -100] = self.tokenizer.pad_token_id
        batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
        return batch

    def _get_dataloaders(self):
        train_dataloader = DataLoader(
            self.train_dataset, 
            shuffle=True,#self.optimizer_args.train_shuffle, 
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator
        )
        eval_dataloader = DataLoader(
            self.eval_dataset, 
            shuffle=False, 
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator
        )
        return train_dataloader, eval_dataloader

    def _get_test_dataloader(self):
        eval_dataloader = DataLoader(
            self.eval_dataset, 
            shuffle=False, 
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator
        )
        return eval_dataloader

    def _sft_loss(self, batch):
        loss = self.model(**batch)["loss"]
        return loss

    def _postprocess_vllm_generate(self, output, input_text):
        formatted_output = {
            "input": input_text,
            "sequence": output.text,
            "tokens": output.token_ids,
            "logprobs": [],
        }
        for dict_, token in zip(output.logprobs, formatted_output["tokens"]):
            formatted_output["logprobs"].append(dict_[token].logprob)
        return formatted_output

    def _postprocess_vllm_classify(self, output, input_text):
        formatted_outputs = []
        # print(output)
        for class_, probability in enumerate(output.outputs.probs):
            formatted_output = {
                "sequence": str(class_),
                "tokens": class_,
                "probs": probability,
            }
            formatted_outputs.append(formatted_output)
        return formatted_outputs

    def _postprocess_vllm(self, outputs):
        formatted_outputs = []
        for sample in outputs:
            input_text = self.tokenizer.batch_decode([sample.prompt_token_ids], skip_special_tokens=True)[0].replace(self.model_args.prompt_prefix, "").replace("model\n", "")
            if self.model_args.generation_max_len == 1:
                formatted_output = self._postprocess_vllm_classify(sample, input_text)
                formatted_outputs.extend(formatted_output)
            else:
                for output in sample.outputs:
                    formatted_output = self._postprocess_vllm_generate(output, input_text)
                    formatted_outputs.append(formatted_output)
                    # for dict_, token in zip(sample.prompt_logprobs, sample.prompt_token_ids):
                    #     if dict_ is not None:
                    #         formatted_output["prompt_logprobs"].append(dict_[token].logprob)
            # print("\n"*2)
        return formatted_outputs


    def _training_step(self, batch, epoch, step_idx, num_total_steps):
        params = [p for p in self.model.parameters() if p.requires_grad]
        with self.optimizer.sampled_params(train=True) if "ivon" in self.optimizer_args.optimizer_name.lower() else nullcontext():
            self.optimizer.zero_grad()
            batch = self._prepare_batch(batch)
            if self.train_fn == "sft":
                loss = self._sft_loss(batch)
            else:
                raise NotImplementedError()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0) # TODO: this is missing also below 
        self.optimizer.step()
        self.lr_scheduler.step()

        self.training_step += 1
        self.on_step_end()
        return loss

    @torch.no_grad()
    def evaluate(self, eval_dataloader):
        self.model.eval()
        outputs, labels = [], []
        outputs_posterior = []
        test_loss, test_loss_posterior = [], []
        for batch in tqdm(eval_dataloader):
            batch = self._prepare_batch(batch, is_eval=True)
            with torch.no_grad():
                local_outputs = self.model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_new_tokens=512, num_return_sequences=1, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
            local_outputs[local_outputs == -100] = self.tokenizer.pad_token_id
            local_outputs = self.tokenizer.batch_decode(local_outputs, skip_special_tokens=True)
            outputs.append(local_outputs)

        predictions = self.method.postprocess_predictions(outputs, self.eval_dataset)
        results = self.method.compute_metrics(predictions, self.eval_dataset, self.reward_model)
        if len(outputs_posterior) > 0:
            predictions_posterior = self.method.postprocess_predictions(outputs_posterior, self.eval_dataset)
            results_posterior = self.method.compute_metrics(predictions_posterior, self.eval_dataset)
            for k, v in results_posterior.items():
                results[k+" Posterior"] = v
        if len(test_loss) > 0:
            test_loss = torch.cat(test_loss, dim=0).float().mean()
            results["Test Loss"] = test_loss.item()
        if len(test_loss_posterior) > 0:
            test_loss = torch.cat(test_loss_posterior, dim=0).float().mean()
            results["Test Loss Posterior"] = test_loss.item()
        return results

    def train(self):
        step_idx = 0
        has_refreshed = False

        for epoch in range(self.args.num_train_epochs):
            self.model.eval()
    
            train_dataloader, test_dataloader = self._get_dataloaders()
            num_total_steps = self.args.num_train_epochs * len(train_dataloader)

            for batch in tqdm(train_dataloader):

                loss = self._training_step(batch, epoch, step_idx, num_total_steps)
                wandb.log({
                    "Loss": loss
                })
                self.losses.append(loss.detach().cpu().item())
                step_idx += 1
                if step_idx % self.args.eval_steps == 0:
                    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
                    for logger in loggers:
                        logger.setLevel(logging.INFO)
                    results = self.evaluate(test_dataloader)
                    wandb.log(results, commit=False)
                    logging.info(f"Results: {results}")
            
            results = self.evaluate(test_dataloader)
            wandb.log(results)
            logging.info(f"Results: {results}")
        return results

    def predict(self, test_dataset):
        final_outputs = []
        counter = 0
        counter_correct = 0
        test_dataloader = self._get_test_dataloader()
        for batch in tqdm(test_dataloader):
            all_prompts = []
            for idx in range(batch["input_ids"].shape[0]):
                tokens = []
                for token in batch["input_ids"][idx].tolist():
                    if token != self.tokenizer.pad_token_id:
                        tokens.append(token)
                all_prompts.append(TokensPrompt(prompt_token_ids=tokens))
            if self.model_args.generation_max_len == 1:
                outputs = self.vllm_engine.classify(all_prompts)
            else:
                outputs = self.vllm_engine.generate(all_prompts, sampling_params=self.sampling_params)
            counter += 1
            outputs = self._postprocess_vllm(outputs)
            final_outputs.extend(outputs)
        return final_outputs

    def on_step_end(self):
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        for logger in loggers:
            logger.setLevel(logging.INFO)
        if self.training_step % 32 == 0:
            if torch.cuda.is_available():
                max_gpu_allocated = torch.cuda.max_memory_allocated() / 10 ** 9
                logging.info(f" Mean loss: {round(float(np.mean(np.array(self.losses))), 4)} Steps: {self.training_step} LR: {self.optimizer.param_groups[0]['lr']}")
                wandb.log({"mean_loss": round(float(np.mean(np.array(self.losses))), 4)}, commit=False)
        if self.training_step % 128 == 0:
            if torch.cuda.is_available():
                max_gpu_allocated = torch.cuda.max_memory_allocated() / 10 ** 9
                logging.info(
                    f" Maximum allocated GPU memory: {max_gpu_allocated:.3f} GB")
        if self.optimizer_args.train_fn != "sft" and self.optimizer_args.sft_warmup_steps <= self.training_step:
            self.train_fn = self.optimizer_args.train_fn

    def save_model(self, output_dir):
        self.model.save_pretrained(
            output_dir, safe_serialization=self.args.save_safetensors
        )
        torch.save(self.optimizer, os.path.join(output_dir, "optimizer.pt"))