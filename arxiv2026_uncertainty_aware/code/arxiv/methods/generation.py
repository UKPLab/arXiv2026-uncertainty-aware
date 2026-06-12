import dataclasses
import math

import numpy as np
import torch

from peft import AutoPeftModelForCausalLM, AutoPeftModelForSeq2SeqLM, \
 get_peft_config, get_peft_model, LoraConfig, TaskType
from transformers import PretrainedConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM, \
 DataCollatorForSeq2Seq, Trainer, DefaultDataCollator, Seq2SeqTrainer, DataCollatorForLanguageModeling, \
 AutoTokenizer, DataCollatorWithPadding, default_data_collator

from arxiv.methods.base import Method
from arxiv.methods.preprocessing.generation import CausalSeq2SeqPreprocessor, Seq2SeqPreprocessor, LanguageModelingPreprocessor, \
 LanguageModelingPreprocessorWithEOSToken, MathDialPreprocessor, ReviewDirectPreprocessor, ReviewScorePreprocessor, ReviewScoreConsistencyProcessor, \
 ReviewQualityProcessor, ReviewMBRPreprocessor, DialogActPreprocessor, MathDialQualityProcessor, MathDialMBRPreprocessor, ReviewScoreNoOraclePreprocessor
from arxiv.methods.utils import *
from arxiv.models.nanogpt_flash import GPT, GPTConfig
from arxiv.trainer.custom_trainer import SimpleTrainer

class Seq2SeqMethod(Method):
    name = "seq2seq"
    peft_task_type = "SEQ_2_SEQ_LM"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.metrics = [
        ]

    def _ngram_stats(self, data, N):
        from nltk import ngrams
        """Return basic ngram statistics, as well as a dict of all ngrams and their freqsuencies."""
        ngram_freqs = {}  # ngrams with frequencies
        ngram_len = 0  # total number of ngrams
        for inst in data:
            for ngram in ngrams(inst, N):
                ngram_freqs[ngram] = ngram_freqs.get(ngram, 0) + 1
                ngram_len += 1
        # number of unique ngrams
        uniq_ngrams = len([val for val in ngram_freqs.values() if val == 1])
        return ngram_freqs, uniq_ngrams, ngram_len

    def _entropy(self, ngram_freqs):
        """Shannon entropy over ngram frequencies"""
        total_freq = sum(ngram_freqs.values())
        return -sum(
            [
                freq / total_freq * np.log2(freq / total_freq)
                for freq in ngram_freqs.values()
            ]
        )

    def compute_metrics(self, predictions, eval_dataset, reward_model):
        if self.model_args.task_type == "classification":
            num_correct = 0
            for prediction in predictions:
                if prediction["sequence"] == prediction["reference"]:
                    num_correct += 1
            results = {"Accuracy": float(num_correct) / len(predictions)}

        elif self.model_args.task_type == "translation":
            import sacrebleu
            sequences = [prediction["sequence"] for prediction in predictions]
            labels = [prediction["reference"] for prediction in predictions]
            sources = [prediction["source"] for prediction in predictions]
            bleu = sacrebleu.corpus_bleu(sequences, [labels])
            comet_inputs = [{"src": source, "mt": sequence} for source, sequence in zip(sources, sequences)]
            scores = reward_model.predict(comet_inputs, batch_size=128, gpus=1, progress_bar=False).scores
            comet_inputs = [{"src": source, "mt": sequence, "ref": ref} for source, sequence, ref in zip(sources, sequences, labels)]

            results = {
                "Bleu": round(float(bleu.score), 4),
                # "Comet": round(float(np.mean(comet_scores)), 4),
                "Reward": round(float(np.mean(scores)), 4)
            }
        elif self.model_args.task_type == "gsm8k":
            num_correct = 0
            for example in predictions:
                try:
                    example["reference"] = int(example["reference"].split("### ")[-1])
                    example["sequence"] = int(example["sequence"].split("### ")[-1])
                except:
                    example["sequence"] = -1
                    example["reference"] = -2
                if example["reference"] == example["sequence"]:
                    num_correct += 1
                    results = {"Accuracy": float(num_correct) / len(predictions)}
        elif self.model_args.task_type == "summarization":
            rouge1s, rougels = [], []
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

            for example in predictions:
                prediction, reference = example["sequence"], example["reference"]
                score = scorer.score(prediction, reference)
                rouge1s.append(score["rouge1"].fmeasure)
                rougels.append(score["rougeL"].fmeasure)

            return {
                "rouge1": round(np.mean(rouge1s), 4),
                "rougeL": round(np.mean(rougels), 4)
            }
        else:
            raise NotImplementedError

        return results


    def get_trainer_class(self):
        return SimpleTrainer
        
    def get_data_collator(self):
        return DataCollatorForSeq2Seq(self.tokenizer)

    def get_model_class(self, config: PretrainedConfig):
        if self.optimizer_args.load_peft_model:
            return AutoPeftModelForSeq2SeqLM
        else:
            return AutoModelForSeq2SeqLM

    def preprocess_features(self, features, train=True):
        processor = Seq2SeqPreprocessor(self.config, self.data_args, self.model_args, self.tokenizer)
        input_ids, labels = processor.preprocess(features)

        return_dict = {
            "input_ids": input_ids,
        }

        return_dict["labels"] = labels

        return return_dict

    def postprocess_predictions(self, p, dataset, scores=None):
        out = []
        references = [sample["labels"] for sample in dataset]
        references = self.tokenizer.batch_decode(references, skip_special_tokens=True)
        decoded_predictions = []
        for batch in p:
            for prediction in batch:
                decoded_predictions.append(prediction)
        for prediction, reference in zip(decoded_predictions, references):
            out.append({
                "sequence": prediction,
                "reference": reference
            })

        return out

    def compute_test_metrics(self, out):
        return {}
        
class CausalSeq2SeqMethod(Seq2SeqMethod):
    name="causal_seq2seq"
    peft_task_type = "CAUSAL_LM"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.metrics = [
        ]

        if not self.data_args.is_training:
            self.tokenizer.padding_side = "left"
    
    def preprocess_features(self, features, train=True):
        processor = CausalSeq2SeqPreprocessor(self.config, self.data_args, self.model_args, self.optimizer_args, self.tokenizer)
        input_ids, labels = processor.preprocess(features, train=train)

        return_dict = {
            "input_ids": input_ids,
            "labels": labels
        }

        return return_dict
        
    def get_data_collator(self):
        return DataCollatorForSeq2Seq(self.tokenizer)

    def get_model_class(self, config: PretrainedConfig):
        return AutoModelForCausalLM

    def get_trainer_class(self):
        return SimpleTrainer

    def postprocess_predictions(self, predictions, dataset, input_ids=None, num_beams=1):
        out = []
        # None for RL-training but for eval it's the eval dataset
        if dataset is not None:
            reference_strings = self.tokenizer.batch_decode([sample["labels"] for sample in dataset], skip_special_tokens=True)
        model_class = self.get_model_class(self.config)

        if isinstance(predictions[0][0], str):
            decoded_predictions = []
            for batch in predictions:
                for prediction in batch:
                    decoded_predictions.append(prediction)
        else:
            decoded_predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        if dataset is not None:
            decoded_inputs = self.tokenizer.batch_decode(
                [sample["input_ids"] for sample in dataset], skip_special_tokens=True
            )
        else:
            assert input_ids is not None
            decoded_inputs = self.tokenizer.batch_decode(
                input_ids, skip_special_tokens=True
            )

        for idx, prediction in enumerate(decoded_predictions):
            input_idx = math.floor(idx / self.model_args.num_return_sequences)
            if dataset is not None:
                out.append({
                    "sequence": prediction[len(decoded_inputs[input_idx]):].strip(),
                    "reference": reference_strings[input_idx],
                    "source": decoded_inputs[input_idx]
                })
            else:
                input_idx = math.floor(idx / num_beams)
                out.append({
                    "sequence": prediction[len(decoded_inputs[input_idx]):].strip(),
                    "source": decoded_inputs[input_idx].replace("user\n", "").replace("\nmodel\n", "").replace("\nsystem\n", "").replace(self.model_args.prompt_prefix, "").replace("You are a helpful assistant. ", "") 
                })
        return out


class MathDialSampleMethod(CausalSeq2SeqMethod):
    name="mathdial_sample"
    peft_task_type = "CAUSAL_LM"

    def preprocess_features(self, features, train=True):
        processor = MathDialPreprocessor(self.config, self.data_args, self.model_args, self.optimizer_args, self.tokenizer)
        input_ids, labels = processor.preprocess(features)

        return_dict = {
            "input_ids": input_ids,
        }

        return_dict["labels"] = labels

        return return_dict

class ReviewGenerationDirectMethod(CausalSeq2SeqMethod):
    name="review_direct"
    peft_task_type = "CAUSAL_LM"

    def preprocess_features(self, features, train=True):
        processor = ReviewDirectPreprocessor(self.config, self.data_args, self.model_args, self.optimizer_args, self.tokenizer)
        input_ids, labels = processor.preprocess(features)

        return_dict = {
            "input_ids": input_ids,
        }

        return_dict["labels"] = labels

        return return_dict

class ReviewScorePredictionMethod(CausalSeq2SeqMethod):
    name="review_score_prediction"
    peft_task_type = "CAUSAL_LM"

    def preprocess_features(self, features, train=True):
        processor = ReviewScorePreprocessor(self.config, self.data_args, self.model_args, self.optimizer_args, self.tokenizer)
        input_ids, labels = processor.preprocess(features)

        return_dict = {
            "input_ids": input_ids,
        }

        return_dict["labels"] = labels

        return return_dict


class ReviewScorePredictionNoOracleMethod(CausalSeq2SeqMethod):
    name="review_score_prediction_no_oracle"
    peft_task_type = "CAUSAL_LM"

    def preprocess_features(self, features, train=True):
        processor = ReviewScoreNoOraclePreprocessor(self.config, self.data_args, self.model_args, self.optimizer_args, self.tokenizer)
        input_ids, labels = processor.preprocess(features)

        return_dict = {
            "input_ids": input_ids,
        }

        return_dict["labels"] = labels

        return return_dict

    def postprocess_scores(self, results):
        from datasets import load_dataset
        nbest = self.model_args.num_return_sequences
        final_scores = []
        for idx in range(int(len(results) // (10.*nbest))):
            scores = {
                i: 0.0 for i in range(10)
            }
            for result in results[idx * (10 * nbest): idx * (10 * nbest) + (10 * nbest)]:
                score = int(result["sequence"])
                scores[score] += result["probs"]
            formatted_scores = {}
            for k, v in scores.items():
                formatted_scores[str(k)] = v / nbest
            final_scores.append(formatted_scores)

        test_dataset = load_dataset(self.data_args.dataset_name, self.data_args.dataset_config_name, split=self.data_args.dataset_test_split)
        final_final_scores = []
        old_review_input = None

        for sample in test_dataset:
            for review in sample["reviews"]:
                if old_review_input is None:
                    idx = 0
                    old_review_input = sample["paper"]
                else:
                    if old_review_input != sample["paper"]:
                        idx += 1
                        old_review_input = sample["paper"]
                for k, v in final_scores[idx].items():
                    final_final_scores.append({"sequence": k, "tokens": k, "probs": v})

        return final_final_scores

class ReviewScoreConsistencyMethod(CausalSeq2SeqMethod):
    name="review_score_consistency"
    peft_task_type = "CAUSAL_LM"

    def preprocess_features(self, features, train=True):
        processor = ReviewScoreConsistencyProcessor(self.config, self.data_args, self.model_args, self.optimizer_args, self.tokenizer)
        input_ids, labels = processor.preprocess(features)

        return_dict = {
            "input_ids": input_ids,
        }

        return_dict["labels"] = labels

        return return_dict

class ReviewQualityMethod(CausalSeq2SeqMethod):
    name="review_quality_estimation"
    peft_task_type = "CAUSAL_LM"

    def preprocess_features(self, features, train=True):
        processor = ReviewQualityProcessor(self.config, self.data_args, self.model_args, self.optimizer_args, self.tokenizer)
        input_ids, labels = processor.preprocess(features)

        return_dict = {
            "input_ids": input_ids,
        }

        return_dict["labels"] = labels

        return return_dict

class ReviewMBRMethod(CausalSeq2SeqMethod):
    name="review_mbr"
    peft_task_type = "CAUSAL_LM"

    def preprocess_features(self, features, train=True):
        processor = ReviewMBRPreprocessor(self.config, self.data_args, self.model_args, self.optimizer_args, self.tokenizer)
        input_ids, labels = processor.preprocess(features)

        return_dict = {
            "input_ids": input_ids,
        }

        return_dict["labels"] = labels

        return return_dict

    def predict(self, test_dataset):
        import json
        import numpy as np
        from datasets import load_dataset
        from tqdm import tqdm

        with open(self.data_args.reviews_file, "r") as f:
            reviews = json.load(f)
        with open(self.data_args.quality_file, "r") as f:
            quality_scores = json.load(f)
        with open(self.data_args.scores_file, "r") as f:
            score_distribution = json.load(f)
        if self.data_args.threshold_file is not None:
            with open(self.data_args.threshold_file, "r") as f:
                threshold = json.load(f)["threshold"]
        else:
            threshold = None
        ground_truth = load_dataset(self.data_args.dataset_name, self.data_args.dataset_config_name, split=self.data_args.dataset_test_split)
        possible_scores = self.model_args.possible_scores.split(";")

        possible_quality_scores = self.model_args.possible_quality_scores.split(";") 
        num_return_sequences = self.model_args.num_return_sequences

        num_input_samples = len(reviews) // self.model_args.num_return_sequences
        score_idx = 0

        final_samples = []

        for idx, sample in tqdm(enumerate(ground_truth)):
            local_score_dist = []
            for review in sample["reviews"]:
                review_scores = []
                for score in possible_scores:
                    review_scores.append(score_distribution[score_idx]["probs"])
                    score_idx += 1
                local_score_dist.append(review_scores)
            local_score_dist = np.stack(local_score_dist).mean(0)

            local_reviews = reviews[idx*num_return_sequences: idx*num_return_sequences + num_return_sequences]
            local_quality_scores = quality_scores[idx*num_return_sequences*len(possible_scores)*len(possible_quality_scores): 
                                                  idx*num_return_sequences*len(possible_scores)*len(possible_quality_scores) + num_return_sequences*len(possible_scores)*len(possible_quality_scores)]

            local_utilities = []
            for jdx, review in enumerate(local_reviews):
                utility = 0.0
                # per review we have for each of the 10 scores 5 ratings!
                for kdx, score_prob in enumerate(local_score_dist):
                    if threshold is None or 1. - score_prob < threshold:
                        for ldx in range(len(possible_quality_scores)):
                            utility += local_quality_scores[jdx*len(possible_scores)*len(possible_quality_scores) + kdx*len(possible_quality_scores) + ldx]["probs"] * ldx * score_prob
                local_utilities.append(utility)
            best_review_idx = np.argmax(local_utilities)
            best_utility = np.max(local_utilities)
            best_review = local_reviews[best_review_idx]
            mean_score = float(np.sum([prob * (idx + 1) for idx, prob in enumerate(local_score_dist)]))

            score_variance = np.sum([prob * ((idx+1) - mean_score)**2 for idx, prob in enumerate(local_score_dist)])
            final_samples.append({
                "review": best_review,
                "utility": best_utility,
                "expected_score": mean_score,
                "score_variance": score_variance
            })

        return final_samples

class EvalReviewsMethod(CausalSeq2SeqMethod):
    name="eval_reviews"
    peft_task_type = "CAUSAL_LM"

    def preprocess_features(self, features, train=True):
        processor = ReviewMBRPreprocessor(self.config, self.data_args, self.model_args, self.optimizer_args, self.tokenizer)
        input_ids, labels = processor.preprocess(features)

        return_dict = {
            "input_ids": input_ids,
        }

        return_dict["labels"] = labels

        return return_dict
    
    def _self_bleu(self, reviews):
        self_bleus = []
        import sacrebleu
        for idx, review in enumerate(reviews):
            refs = reviews[:idx] + reviews[idx+1:]
            self_bleus.append(sacrebleu.corpus_bleu([review["sequence"]], [[ref["sequence"]] for ref in refs]).score)
        return np.mean(self_bleus)

    def predict(self, test_dataset):
        import json
        import re
        import numpy as np
        from datasets import load_dataset
        from sacrebleu.metrics import BLEU
        from vllm import LLM, SamplingParams
        from vllm.inputs.data import TokensPrompt
        bleu = BLEU()
        test_dataset = load_dataset(self.data_args.dataset_name, self.data_args.dataset_config_name, split=self.data_args.dataset_test_split)

        with open(self.data_args.mbr_file, "r") as f:
            mbr_sequences = json.load(f)
        with open(self.data_args.reviews_file, "r") as f:
            reviews = json.load(f)
        num_return_sequences = self.model_args.num_return_sequences

        ground_truth_reviews = []
        mbr_reviews = []
        sampled_reviews = []
        self_bleus = []

        for idx, (ground_truth, mbr_sample) in enumerate(zip(test_dataset, mbr_sequences)):
            all_review_samples = reviews[idx*num_return_sequences: idx*num_return_sequences + num_return_sequences]
            # self_bleus.append(self._self_bleu(all_review_samples))
            mbr_review = mbr_sample["review"]["sequence"]
            review_probs = []
            for review in all_review_samples:
                review_probs.append(np.mean([np.exp(prob) for prob in review["logprobs"]]))
            if self.model_args.use_probs_for_eval:
                most_likely_review_idx = np.argmax(review_probs)
            else:
                most_likely_review_idx = 0 #np.argmax(review_probs)
            sampled_review = all_review_samples[most_likely_review_idx]["sequence"]

            for review in ground_truth["reviews"]:
                ground_truth_reviews.append(review["text"])
                mbr_reviews.append(mbr_review)
                sampled_reviews.append(sampled_review)

        assert len(sampled_reviews) == len(ground_truth_reviews) == len(mbr_reviews)

        mbr_bleu = bleu.corpus_score(mbr_reviews, [ground_truth_reviews])
        sampled_bleu = bleu.corpus_score(sampled_reviews, [ground_truth_reviews])

        ###########################
        ### Reward-model-based eval
        ###########################

        mbr_reviews = []
        sampled_reviews = []

        for idx, (ground_truth, mbr_sample) in enumerate(zip(test_dataset, mbr_sequences)):
            all_review_samples = reviews[idx*num_return_sequences: idx*num_return_sequences + num_return_sequences]
            mbr_review = mbr_sample["review"]["sequence"]
            review_probs = []
            for review in all_review_samples:
                review_probs.append(np.mean([np.exp(prob) for prob in review["logprobs"]]))
            most_likely_review_idx = 0 #np.argmax(review_probs)
            sampled_review = all_review_samples[most_likely_review_idx]["sequence"]
            mbr_reviews.append(mbr_review)
            sampled_reviews.append(sampled_review)

        model_path = "/path/to/SciRM-Ref-Qwen2.5-7B-Instruct/"

        model = LLM(model=model_path, dtype=torch.bfloat16, max_model_len=12288, trust_remote_code=True)

        sampling_params = model.get_default_sampling_params()
        sampling_params.max_tokens = 2048
        sampling_params.temperature = 0.0
        sampling_params.top_p = 0.95

        mbr_chats = {
            "actionable": [],
            "verifiable": [],
            "helpful": [],
            "grounded": []
        }
        mbr_scores = {}
        sampling_chats = {
            "actionable": [],
            "verifiable": [],
            "helpful": [],
            "grounded": []
        }
        sampling_scores = {}


        def get_scores(messages):
            scores = []
            completions = model.chat(messages, sampling_params)
            for completion in completions:
                output = completion.outputs[0].text
                try:
                    score = int(re.search(r"<score>\s*(\d+)\s*</score>", output).group(1))
                    scores.append(score)
                except:
                    print(output)
            return scores

        for review in mbr_reviews:
            review = review.split("\"weaknesses\": ")[-1].replace("}", "").strip()
            parts = re.split(r'(?=\d+\.)', review)
            parts = [p.strip() for p in parts if p.strip()]
            for sentence in parts:
                user_message = QUERY_SCIRM + CRITERIA_SCIRM_ACTIONABLE + EXAMPLES_SCIRM_ACTIONABLE + "[ANSWER]: " + sentence
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_SCIRM},
                    {"role": "user",   "content": user_message},
                ]
                mbr_chats["actionable"].append(messages)
                user_message = QUERY_SCIRM + CRITERIA_SCIRM_VERIFIABLE + EXAMPLES_SCIRM_VERIFIABLE + "[ANSWER]: " + sentence
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_SCIRM},
                    {"role": "user",   "content": user_message},
                ]
                mbr_chats["verifiable"].append(messages)
                user_message = QUERY_SCIRM + CRITERIA_SCIRM_HELPFULNESS + EXAMPLES_SCIRM_HELPFULNESS + "[ANSWER] :" + sentence
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_SCIRM},
                    {"role": "user",   "content": user_message},
                ]
                mbr_chats["helpful"].append(messages)
                user_message = QUERY_SCIRM + CRITERIA_SCIRM_GROUNDED + EXAMPLES_SCIRM_GROUNDED + "[ANSWER]: " + sentence
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_SCIRM},
                    {"role": "user",   "content": user_message},
                ]
                mbr_chats["grounded"].append(messages)

        for review in sampled_reviews:
            review = review.split("\"weaknesses\": ")[-1].replace("}", "").strip()
            parts = re.split(r'(?=\d+\.)', review)
            parts = [p.strip() for p in parts if p.strip()]
            for sentence in parts:
                user_message = QUERY_SCIRM + CRITERIA_SCIRM_ACTIONABLE + EXAMPLES_SCIRM_ACTIONABLE + "[ANSWER]: " + sentence
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_SCIRM},
                    {"role": "user",   "content": user_message},
                ]
                sampling_chats["actionable"].append(messages)
                user_message = QUERY_SCIRM + CRITERIA_SCIRM_VERIFIABLE + EXAMPLES_SCIRM_VERIFIABLE + "[ANSWER]: " + sentence
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_SCIRM},
                    {"role": "user",   "content": user_message},
                ]
                sampling_chats["verifiable"].append(messages)
                user_message = QUERY_SCIRM + CRITERIA_SCIRM_HELPFULNESS + EXAMPLES_SCIRM_HELPFULNESS + "[ANSWER] :" + sentence
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_SCIRM},
                    {"role": "user",   "content": user_message},
                ]
                sampling_chats["helpful"].append(messages)
                user_message = QUERY_SCIRM + CRITERIA_SCIRM_GROUNDED + EXAMPLES_SCIRM_GROUNDED + "[ANSWER]: " + sentence
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_SCIRM},
                    {"role": "user",   "content": user_message},
                ]
                sampling_chats["grounded"].append(messages)

        mbr_scores["actionable"] = float(np.mean(get_scores(mbr_chats["actionable"])))
        mbr_scores["verifiable"] = float(np.mean(get_scores(mbr_chats["verifiable"])))
        mbr_scores["helpful"] = float(np.mean(get_scores(mbr_chats["helpful"])))
        mbr_scores["grounded"] = float(np.mean(get_scores(mbr_chats["grounded"])))
        mbr_scores["bleu"] = float(mbr_bleu.score)

        sampling_scores["actionable"] = float(np.mean(get_scores(sampling_chats["actionable"])))
        sampling_scores["verifiable"] = float(np.mean(get_scores(sampling_chats["verifiable"])))
        sampling_scores["helpful"] = float(np.mean(get_scores(sampling_chats["helpful"])))
        sampling_scores["grounded"] = float(np.mean(get_scores(sampling_chats["grounded"])))
        sampling_scores["bleu"] = float(sampled_bleu.score)


        return {
            "mbr": mbr_scores,
            "sampling": sampling_scores,
        }



class DialogActPredictionMethod(CausalSeq2SeqMethod):
    name="dialog_act"
    peft_task_type = "CAUSAL_LM"

    def preprocess_features(self, features, train=True):
        processor = DialogActPreprocessor(self.config, self.data_args, self.model_args, self.optimizer_args, self.tokenizer)
        input_ids, labels = processor.preprocess(features)

        return_dict = {
            "input_ids": input_ids,
        }

        return_dict["labels"] = labels

        return return_dict


class MathdialQualityMethod(CausalSeq2SeqMethod):
    name="mathdial_quality"
    peft_task_type = "CAUSAL_LM"

    def preprocess_features(self, features, train=True):
        processor = MathDialQualityProcessor(self.config, self.data_args, self.model_args, self.optimizer_args, self.tokenizer)
        input_ids, labels = processor.preprocess(features)

        return_dict = {
            "input_ids": input_ids,
        }

        return_dict["labels"] = labels

        return return_dict



class MathDialCalibrationMethod(CausalSeq2SeqMethod):
    name="mathdial_calibration"
    peft_task_type = "CAUSAL_LM"

    def preprocess_features(self, features, train=True):
        processor = DialogActPreprocessor(self.config, self.data_args, self.model_args, self.optimizer_args, self.tokenizer)
        input_ids, labels = processor.preprocess(features)

        return_dict = {
            "input_ids": input_ids,
        }

        return_dict["labels"] = labels

        return return_dict

    def predict(self, test_dataset):
        import json
        import numpy as np
        from datasets import load_dataset

        with open(self.data_args.scores_file) as f:
            act_scores = json.load(f)
        ground_truth = load_dataset(self.data_args.dataset_name, self.data_args.dataset_config_name, split=self.data_args.dataset_test_split)
        acts2index = dict([(v, i) for i, v in enumerate(self.model_args.possible_acts.split(";"))])
        num_acts = len(self.model_args.possible_acts.split(";"))
        scores = []
        alpha = 0.05
        n = len(ground_truth)
        num_correct = 0

        for idx, sample in enumerate(ground_truth):
            ground_truth_act = acts2index[sample["dialog_act"]]
            ground_truth_prob = act_scores[idx*num_acts + ground_truth_act]["probs"]
            all_acts = act_scores[idx*num_acts: idx*num_acts+num_acts]
            best_act = np.argmax([act["probs"] for act in all_acts])
            num_correct += int(ground_truth_act == best_act)
            scores.append(1.-ground_truth_prob)

        threshold = np.quantile(scores, np.ceil((n + 1.) * (1. - alpha)) / n, method="higher")
        return {"threshold": threshold, "accuracy": num_correct / n}


class MathDialMBR(CausalSeq2SeqMethod):
    name="mathdial_mbr"
    peft_task_type = "CAUSAL_LM"

    def preprocess_features(self, features, train=True):
        processor = MathDialMBRPreprocessor(self.config, self.data_args, self.model_args, self.optimizer_args, self.tokenizer)
        input_ids, labels = processor.preprocess(features)

        return_dict = {
            "input_ids": input_ids,
        }

        return_dict["labels"] = labels

        return return_dict

    def predict(self, test_dataset):
        import json
        import numpy as np
        from datasets import load_dataset
        from tqdm import tqdm

        with open(self.data_args.reviews_file, "r") as f:
            responses = json.load(f)
        with open(self.data_args.quality_file, "r") as f:
            quality_scores = json.load(f)
        with open(self.data_args.scores_file, "r") as f:
            act_distribution = json.load(f)
        if self.data_args.threshold_file is not None:
            with open(self.data_args.threshold_file, "r") as f:
                threshold = json.load(f)["threshold"]
        else:
            threshold = None

        ground_truth = load_dataset(self.data_args.dataset_name, self.data_args.dataset_config_name, split=self.data_args.dataset_test_split)
        possible_acts = self.model_args.possible_acts.split(";")
        possible_quality_scores = self.model_args.possible_quality_scores.split(";") 
        num_return_sequences = self.model_args.num_return_sequences

        num_input_samples = len(responses) // self.model_args.num_return_sequences
        score_idx = 0

        final_samples = []

        for idx, sample in tqdm(enumerate(ground_truth)):
            local_score_dist = act_distribution[idx*len(possible_acts): idx*len(possible_acts) + len(possible_acts)]
            local_responses = responses[idx*num_return_sequences: idx*num_return_sequences + num_return_sequences]
            local_quality_scores = quality_scores[idx*num_return_sequences*len(possible_acts)*len(possible_quality_scores): 
                                                  idx*num_return_sequences*len(possible_acts)*len(possible_quality_scores) + num_return_sequences*len(possible_acts)*len(possible_quality_scores)]

            local_utilities = []
            local_conformal_acts = []
            for jdx, review in enumerate(local_responses):
                utility = 0.0
                for kdx, score_prob in enumerate(local_score_dist):
                    if threshold is None or 1. - score_prob["probs"] < threshold:
                        if jdx == 0:
                            local_conformal_acts.append(possible_acts[kdx])
                        for ldx in range(len(possible_quality_scores)):
                            utility += local_quality_scores[jdx*len(possible_acts)*len(possible_quality_scores) + kdx*len(possible_quality_scores) + ldx]["probs"] * ldx * score_prob["probs"]

            local_utilities.append(utility)

            best_response_idx = np.argmax(local_utilities)
            best_utility = np.max(local_utilities)
            best_response = local_responses[best_response_idx]
            best_act = np.argmax([act["probs"] for act in local_score_dist])
            best_act = possible_acts[best_act]

            final_samples.append({
                "review": best_response,
                "utility": best_utility,
                "dialog_act": best_act,
                "conformal_acts": local_conformal_acts
            })

        return final_samples




class MathDialMinMax(CausalSeq2SeqMethod):
    name="mathdial_minmax"
    peft_task_type = "CAUSAL_LM"

    def preprocess_features(self, features, train=True):
        processor = MathDialMBRPreprocessor(self.config, self.data_args, self.model_args, self.optimizer_args, self.tokenizer)
        input_ids, labels = processor.preprocess(features)

        return_dict = {
            "input_ids": input_ids,
        }

        return_dict["labels"] = labels

        return return_dict

    def predict(self, test_dataset):
        import json
        import numpy as np
        from datasets import load_dataset
        from tqdm import tqdm

        with open(self.data_args.reviews_file, "r") as f:
            responses = json.load(f)
        with open(self.data_args.quality_file, "r") as f:
            quality_scores = json.load(f)
        with open(self.data_args.scores_file, "r") as f:
            act_distribution = json.load(f)
        if self.data_args.threshold_file is not None:
            with open(self.data_args.threshold_file, "r") as f:
                threshold = json.load(f)["threshold"]
        else:
            threshold = None

        ground_truth = load_dataset(self.data_args.dataset_name, self.data_args.dataset_config_name, split=self.data_args.dataset_test_split)
        possible_acts = self.model_args.possible_acts.split(";")
        possible_quality_scores = self.model_args.possible_quality_scores.split(";") 
        num_return_sequences = self.model_args.num_return_sequences
        acts2index = dict([(v, i) for i, v in enumerate(self.model_args.possible_acts.split(";"))])

        num_input_samples = len(responses) // self.model_args.num_return_sequences
        score_idx = 0

        final_samples = []

        for idx, sample in tqdm(enumerate(ground_truth)):
            local_score_dist = act_distribution[idx*len(possible_acts): idx*len(possible_acts) + len(possible_acts)]
            local_responses = responses[idx*num_return_sequences: idx*num_return_sequences + num_return_sequences]
            local_quality_scores = quality_scores[idx*num_return_sequences*len(possible_acts)*len(possible_quality_scores): 
                                                  idx*num_return_sequences*len(possible_acts)*len(possible_quality_scores) + num_return_sequences*len(possible_acts)*len(possible_quality_scores)]

            local_utilities = []
            local_conformal_acts = []
            assert threshold is not None

            for jdx, review in enumerate(local_responses):
                local_utilities.append({idx: 1e10 for idx in range(len(possible_acts))})
                for kdx, score_prob in enumerate(local_score_dist):
                    if threshold is None or 1. - score_prob["probs"] < threshold:
                        if jdx == 0:
                            local_conformal_acts.append(possible_acts[kdx])
                        for ldx in range(len(possible_quality_scores)):
                            local_utilities[jdx][kdx] = local_quality_scores[jdx*len(possible_acts)*len(possible_quality_scores) + kdx*len(possible_quality_scores) + ldx]["probs"] * ldx * score_prob["probs"]

            local_utilities = [min(utilities.values()) for utilities in local_utilities]

            best_response_idx = np.argmax(local_utilities)
            best_utility = np.max(local_utilities)
            best_response = local_responses[best_response_idx]
            best_act = np.argmax([act["probs"] for act in local_score_dist])
            best_act = possible_acts[best_act]

            final_samples.append({
                "review": best_response,
                "utility": best_utility,
                "dialog_act": best_act,
                "conformal_acts": local_conformal_acts
            })

        return final_samples




class EvalMathDialMethod(CausalSeq2SeqMethod):
    name="eval_mathdial"
    peft_task_type = "CAUSAL_LM"

    def preprocess_features(self, features, train=True):
        processor = MathDialMBRPreprocessor(self.config, self.data_args, self.model_args, self.optimizer_args, self.tokenizer)
        input_ids, labels = processor.preprocess(features)

        return_dict = {
            "input_ids": input_ids,
        }

        return_dict["labels"] = labels

        return return_dict
    
    def _self_bleu(self, reviews):
        self_bleus = []
        import sacrebleu
        for idx, review in enumerate(reviews):
            refs = reviews[:idx] + reviews[idx+1:]
            self_bleus.append(sacrebleu.corpus_bleu([review["sequence"]], [[ref["sequence"]] for ref in refs]).score)
        return np.mean(self_bleus)

    def predict(self, test_dataset):
        import json
        import re
        import numpy as np
        from datasets import load_dataset
        from sacrebleu.metrics import BLEU
        from tqdm import tqdm
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.inputs.data import TokensPrompt
        bleu = BLEU()
        test_dataset = load_dataset(self.data_args.dataset_name, self.data_args.dataset_config_name, split=self.data_args.dataset_test_split)

        with open(self.data_args.mbr_file, "r") as f:
            mbr_sequences = json.load(f)
        with open(self.data_args.reviews_file, "r") as f:
            reviews = json.load(f)
        num_return_sequences = self.model_args.num_return_sequences

        ground_truth_reviews = []
        mbr_reviews = []
        sampled_reviews = []
        self_bleus = []

        for idx, (ground_truth, mbr_sample) in enumerate(zip(test_dataset, mbr_sequences)):
            all_review_samples = reviews[idx*num_return_sequences: idx*num_return_sequences + num_return_sequences]
            mbr_review = mbr_sample["review"]["sequence"]
            review_probs = []
            for review in all_review_samples:
                review_probs.append(np.mean([np.exp(prob) for prob in review["logprobs"]]))
            if self.model_args.use_probs_for_eval:
                most_likely_review_idx = np.argmax(review_probs)
            else:
                most_likely_review_idx = 0
            sampled_review = all_review_samples[most_likely_review_idx]["sequence"]
            ground_truth_reviews.append(ground_truth["output"])
            mbr_reviews.append(mbr_review)
            sampled_reviews.append(sampled_review)

        assert len(sampled_reviews) == len(ground_truth_reviews) == len(mbr_reviews)

        mbr_bleu = bleu.corpus_score(mbr_reviews, [ground_truth_reviews])
        sampled_bleu = bleu.corpus_score(sampled_reviews, [ground_truth_reviews])

        ###########################
        ### Reward-model-based eval
        ###########################

        SYSTEM_PROMPT_MATHDIAL = ("Judge the pedagogical quality of the responses provided by two teachers. Focus on the quality of the "
                                  "scaffolding guidance, correctness, and actionability of the feedback through nudges, questions "
                                  "and hints. Do not give high scores for revealing the full answer.")

        self.tokenizer = AutoTokenizer.from_pretrained("eth-nlped/Qwen2.5-1.5B-pedagogical-rewardmodel")

        mbr_chats = []
        mbr_scores = []
        sampling_chats = []
        sampling_scores = []

        for response, ground_truth in zip(mbr_reviews, test_dataset):
            conversation = []

            # Add system prompt
            system_prompt = {"role": "system",
                         "content": SYSTEM_PROMPT_MATHDIAL}
            conversation.append(system_prompt)
            conversation.append({"role": "user",
                                 "content": "Problem: " + ground_truth["problem"] + "\nReference Solution: " + ground_truth["ground_truth"]})

            # Add the dialog history
            for entry in ground_truth["turns"]:
                role = "assistant" if entry["role"] in ["teacher", "tutor"] else "user"
                conversation.append({"role": role, "content": entry["text"]})

            conversation.append({"role": "assistant", "content": response})
            tokens = self.tokenizer.apply_chat_template(conversation, tokenize=True)#, return_tensors="pt")
            tokens = TokensPrompt(prompt_token_ids=tokens)
            mbr_chats.append(tokens)

        for response, ground_truth in zip(sampled_reviews, test_dataset):
            conversation = []

            # Add system prompt
            system_prompt = {"role": "system",
                         "content": SYSTEM_PROMPT_MATHDIAL}
            conversation.append(system_prompt)
            conversation.append({"role": "user",
                                 "content": "Problem: " + ground_truth["problem"] + "\nReference Solution: " + ground_truth["ground_truth"]})

            # Add the dialog history
            for entry in ground_truth["turns"]:
                role = "assistant" if entry["role"] in ["teacher", "tutor"] else "user"
                conversation.append({"role": role, "content": entry["text"]})

            conversation.append({"role": "assistant", "content": response})
            tokens = self.tokenizer.apply_chat_template(conversation, tokenize=True)#, return_tensors="pt")
            tokens = TokensPrompt(prompt_token_ids=tokens)
            sampling_chats.append(tokens)

        ground_truth_chats = []

        for ground_truth in test_dataset:
            conversation = []

            # Add system prompt
            system_prompt = {"role": "system",
                         "content": SYSTEM_PROMPT_MATHDIAL}
            conversation.append(system_prompt)
            conversation.append({"role": "user",
                                 "content": "Problem: " + ground_truth["problem"] + "\nReference Solution: " + ground_truth["ground_truth"]})

            # Add the dialog history
            for entry in ground_truth["turns"]:
                role = "assistant" if entry["role"] in ["teacher", "tutor"] else "user"
                conversation.append({"role": role, "content": entry["text"]})

            conversation.append({"role": "assistant", "content": ground_truth["output"]})
            tokens = self.tokenizer.apply_chat_template(conversation, tokenize=True)#, return_tensors="pt")
            tokens = TokensPrompt(prompt_token_ids=tokens)
            ground_truth_chats.append(tokens)
            

        for response in sampled_reviews:
            pass

        model = LLM(model="eth-nlped/Qwen2.5-1.5B-pedagogical-rewardmodel", dtype=torch.bfloat16, max_model_len=12288, trust_remote_code=True)

        def get_scores(messages):
            scores = []

            completions = model.classify(messages)
            for completion in completions:
                output = completion.outputs.probs#.text
                scores.append(output[0])
            return scores

        mbr_scores = get_scores(mbr_chats)
        sampling_scores = get_scores(sampling_chats)
        ground_truth_scores = get_scores(ground_truth_chats)

        print(sum([i > j for i, j in zip(mbr_scores, sampling_scores)]))
        print(sum([i == j for i, j in zip(mbr_scores, sampling_scores)]))
        print(sum([i < j for i, j in zip(mbr_scores, sampling_scores)]))

        return {
            "mbr": round(float(np.mean(mbr_scores)), 6),
            "sampling": round(float(np.mean(sampling_scores)), 6),
            "ground_truth": round(float(np.mean(ground_truth_scores)), 6),
            "mbr_win_rate": sum([i > j for i, j in zip(mbr_scores, sampling_scores)]) / (len(mbr_scores) - sum([i == j for i, j in zip(mbr_scores, sampling_scores)])),
            "mbr_win_rate_gt": sum([i > j for i, j in zip(mbr_scores, ground_truth_scores)]) / (len(mbr_scores) - sum([i == j for i, j in zip(mbr_scores, ground_truth_scores)])),
            "sampling_win_rate_gt": sum([i > j for i, j in zip(sampling_scores, ground_truth_scores)]) / (len(mbr_scores) - sum([i == j for i, j in zip(sampling_scores, ground_truth_scores)])),
            "mbr_bleu": mbr_bleu.score,
            "sampling_bleu": sampled_bleu.score
        }



class ReviewCalibrationMethod(CausalSeq2SeqMethod):
    name="review_calibration"
    peft_task_type = "CAUSAL_LM"

    def preprocess_features(self, features, train=True):
        processor = ReviewScorePreprocessor(self.config, self.data_args, self.model_args, self.optimizer_args, self.tokenizer)
        input_ids, labels = processor.preprocess(features)

        return_dict = {
            "input_ids": input_ids,
        }

        return_dict["labels"] = labels

        return return_dict

    def predict(self, test_dataset):
        import json
        import numpy as np
        from datasets import load_dataset

        with open(self.data_args.scores_file) as f:
            review_scores = json.load(f)
        ground_truth = load_dataset(self.data_args.dataset_name, self.data_args.dataset_config_name, split=self.data_args.dataset_test_split)

        score2index = dict([(v, i) for i, v in enumerate(self.model_args.possible_scores.split(";"))])
        num_scores = len(self.model_args.possible_scores.split(";"))
        scores = []
        alpha = 0.05
        n = 0
        num_correct = 0
        for sample in ground_truth:
            for review in sample["reviews"]:
                ground_truth_score = score2index[review["score"]]
                ground_truth_prob = review_scores[n*num_scores + ground_truth_score]["probs"]
                all_scores = review_scores[n*num_scores: n*num_scores+num_scores]
                best_score = np.argmax([score["probs"] for score in all_scores])
                num_correct += int(ground_truth_score == best_score)
                scores.append(1.-ground_truth_prob)
                n+=1
        threshold = np.quantile(scores, np.ceil((n + 1.) * (1. - alpha)) / n, method="higher")
        return {"threshold": threshold, "accuracy": num_correct / n}


class ReviewMinMaxMethod(CausalSeq2SeqMethod):
    name="review_minmax"
    peft_task_type = "CAUSAL_LM"

    def preprocess_features(self, features, train=True):
        processor = ReviewMBRPreprocessor(self.config, self.data_args, self.model_args, self.optimizer_args, self.tokenizer)
        input_ids, labels = processor.preprocess(features)

        return_dict = {
            "input_ids": input_ids,
        }

        return_dict["labels"] = labels

        return return_dict

    def predict(self, test_dataset):
        import json
        import numpy as np
        from datasets import load_dataset
        from tqdm import tqdm

        with open(self.data_args.reviews_file, "r") as f:
            reviews = json.load(f)
        with open(self.data_args.quality_file, "r") as f:
            quality_scores = json.load(f)
        with open(self.data_args.scores_file, "r") as f:
            score_distribution = json.load(f)
        if self.data_args.threshold_file is not None:
            with open(self.data_args.threshold_file, "r") as f:
                threshold = json.load(f)["threshold"]
        else:
            threshold = None
        ground_truth = load_dataset(self.data_args.dataset_name, self.data_args.dataset_config_name, split=self.data_args.dataset_test_split)
        possible_scores = self.model_args.possible_scores.split(";")
        possible_quality_scores = self.model_args.possible_quality_scores.split(";") 
        num_return_sequences = self.model_args.num_return_sequences

        num_input_samples = len(reviews) // self.model_args.num_return_sequences
        score_idx = 0

        final_samples = []

        assert threshold is not None
        local_utilities = []
        local_conformal_scores = []


        # this is slow and could be vectorized but for now we're okay with it..
        for idx, sample in tqdm(enumerate(ground_truth)):
            local_score_dist = []
            for review in sample["reviews"]:
                review_scores = []
                for score in possible_scores:
                    review_scores.append(score_distribution[score_idx]["probs"])
                    score_idx += 1
                local_score_dist.append(review_scores)
            local_score_dist = np.stack(local_score_dist).mean(0)

            local_reviews = reviews[idx*num_return_sequences: idx*num_return_sequences + num_return_sequences]
            local_quality_scores = quality_scores[idx*num_return_sequences*len(possible_scores)*len(possible_quality_scores): 
                                                  idx*num_return_sequences*len(possible_scores)*len(possible_quality_scores) + num_return_sequences*len(possible_scores)*len(possible_quality_scores)]

            local_utilities = []
            local_conformal_scores = []
            for jdx, review in enumerate(local_reviews):
                local_utilities.append({idx: 1e10 for idx in range(len(possible_scores))})
                for kdx, score_prob in enumerate(local_score_dist):
                    if threshold is None or 1. - score_prob < threshold:
                        if jdx == 0:
                            local_conformal_scores.append(str(kdx+1.))
                        for ldx in range(len(possible_quality_scores)):
                            local_utilities[jdx][kdx] = local_quality_scores[jdx*len(possible_scores)*len(possible_quality_scores) + kdx*len(possible_quality_scores) + ldx]["probs"] * ldx * score_prob
            local_utilities = [min(utilities.values()) for utilities in local_utilities]

            best_review_idx = np.argmax(local_utilities)
            best_utility = np.max(local_utilities)
            best_review = local_reviews[best_review_idx]
            mean_score = float(np.sum([prob * (idx + 1) for idx, prob in enumerate(local_score_dist)]))

            score_variance = np.sum([prob * ((idx+1) - mean_score)**2 for idx, prob in enumerate(local_score_dist)])
            final_samples.append({
                "review": best_review,
                "utility": best_utility,
                "expected_score": mean_score,
                "score_variance": score_variance,
                "conformal_scores": local_conformal_scores
            })

        return final_samples