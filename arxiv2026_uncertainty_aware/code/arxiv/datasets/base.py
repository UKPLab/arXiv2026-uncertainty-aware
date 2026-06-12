# http://parl.ai/downloads/fits/fits_data_v0.1.tar.gz 

import abc

import datasets

_CITATION = ""
_DESCRIPTION = ""
_HOMEPAGE = ""

import json
import logging

import datasets
import numpy as np
import transformers
from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, DataCollatorWithPadding
from tqdm import tqdm


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class MathDialDataset(object):
    VERSION = datasets.Version("1.0.0")
    DEFAULT_CONFIG_NAME = "default"

    BUILDER_CONFIGS = [
        Seq2SeqDatasetConfig(
            name=name,
            version=datasets.Version("1.0.0"),
            description=""
        ) for name in ["seq2seq", "chat"]
    ]


    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "dataset_id": datasets.Value("string"),
                    "turns": [{
                        "text": datasets.Value("string"),
                        "role": datasets.Value("string"),
                        "dialog_act": datasets.Value("string")
                    }],
                    "output": datasets.Value("string"),
                    "dialog_act": datasets.Value("string"),
                    "problem": datasets.Value("string"),
                    "student_profile": datasets.Value("string"),
                    "incorrect_solution": datasets.Value("string"),
                    "ground_truth": datasets.Value("string")
                }
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

class ReviewDataset(object):
    VERSION = datasets.Version("1.0.0")
    DEFAULT_CONFIG_NAME = "default"

    BUILDER_CONFIGS = [
        Seq2SeqDatasetConfig(
            name=name,
            version=datasets.Version("1.0.0"),
            description=""
        ) for name in ["seq2seq", "chat"]
    ]


    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "paper": datasets.Value("string"),
                    "reviews": [{
                        "text": datasets.Value("string"),
                        "score": datasets.Value("string"),
                    }],
                },
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )


    def _split_generators(self, dl_manager):
        pass

    def _generate_examples(self, filepath):
        pass

    @abc.abstractmethod
    def _map_to_common_format(self, sample):
        pass

    def _download_files(self, urls, data_files, dl_manager):
        if data_files is not None:
            raise NotImplementedError()
        return dl_manager.download_and_extract(urls)
