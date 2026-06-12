import json
from typing import List

import datasets
from datasets import load_dataset
from tqdm import tqdm

from .base import ReviewDataset

class NLPEER(ReviewDataset, datasets.GeneratorBasedBuilder):

    def _map_to_common_format(self, sample):
        paper = sample["paper"]
        reviews = []
        for review in sample["reviews"]:
            score = str(review["scores"]["overall_assessment"])
            review = "\n".join(
                [review["report"][key] for key in ["paper_summary", "summary_of_strengths", "summary_of_weaknesses", "comments_suggestions_and_typos"]]
            )
            reviews.append({
                "text": review,
                "score": score
            })
        return {
            "dataset_id": "nlpeer",
            "paper": "\n".join([line[:-3] for line in paper.split("\n")]).encode('utf-8', 'ignore').decode('utf-8'),
            "reviews": reviews
        }

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        splits = ["test"]
        hf_splits = [datasets.Split.TEST]
        data = {split: [] for split in splits}

        with open("../../data/nlpeer_processed.json", "r") as f:
            dataset = json.load(f)["data"]

        for split in splits:
            for sample in dataset:
                data[split].append(self._map_to_common_format(sample))
        return [
            datasets.SplitGenerator(
                name=ds_split, gen_kwargs={
                    "data": data[split],
                })
            for ds_split, split in zip(hf_splits, splits)
        ]

    def _generate_examples(self, data):
        for idx, sample in tqdm(enumerate(data)):
            if not "id" in sample:
                sample["id"] = str(idx)
            yield idx, sample
