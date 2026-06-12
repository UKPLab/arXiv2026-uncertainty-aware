import numpy as np
import torch
from datasets import interleave_datasets, \
 load_dataset
from peft import AutoPeftModelForSequenceClassification
from scipy.special import log_softmax, softmax
from scipy.stats import spearmanr
from sklearn.metrics import matthews_corrcoef
import torch.nn.functional as F
from transformers import PretrainedConfig, Trainer, AutoTokenizer, \
 DataCollatorWithPadding, AutoModelForSequenceClassification

from arxiv.methods.base import Method
from arxiv.methods.preprocessing.classification import SequenceClassificationPreprocessor
from arxiv.trainer.custom_trainer import SimpleTrainer

class SequenceClassificationMethod(Method):

    name = "classification"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.metrics = [
        ]
        self.peft_task_type = "SEQ_CLS"

    def compute_test_metrics(self, p):
        return self.compute_metrics(p)

    def compute_metrics(self, p):
        predictions = np.argmax(p.predictions, axis=-1)
        confidence = np.exp(np.max(p.predictions, axis=-1))
        accuracy = sum(predictions == p.label_ids) / len(p.label_ids)

        results = {
            "accuracy": round(accuracy, 5)
        }

        return results

    def preprocess_features(self, features, train=True):
        processor = SequenceClassificationPreprocessor(self.config, self.data_args, self.model_args, self.tokenizer)
        input_ids, labels = processor.preprocess(features, train=train)

        return_dict = {
            "input_ids": input_ids,
        }

        return_dict["labels"] = labels

        return return_dict

    def postprocess_predictions(self, predictions, eval_dataset):
        predictions = [p.logits for p in predictions]
        logits = torch.cat(predictions, 0)
        probabilities = F.softmax(logits, dim=-1)
        classes = probabilities.argmax(dim=-1)
        return classes

    def compute_metrics(self, predictions, eval_dataset):
        is_correct = []
        for pred, sample in zip(predictions, eval_dataset):
            is_correct.append(pred.item() == sample["labels"])
        accuracy = float(np.array(is_correct).mean())
        return {
            "Accuracy": accuracy
        }


    def get_trainer_class(self):
        return SimpleTrainer
        
    def get_data_collator(self):
        return DataCollatorWithPadding(self.tokenizer)

    def get_model_class(self, config: PretrainedConfig):
        return AutoModelForSequenceClassification
