import json
from typing import List

import datasets
from datasets import load_dataset
from tqdm import tqdm

from .base import MathDialDataset

class MathDial(MathDialDataset, datasets.GeneratorBasedBuilder):

    def _map_to_common_format(self, sample):
        conversation = sample["conversation"].split("|EOM|")
        turns = []
        samples = []
        for turn in conversation:
            if "Teacher: " in turn:
                role = "teacher"
            else:
                role = "student" 
            turn = turn.replace("Student: ", "").replace("Teacher: ", "")
            for dialog_act in ["(probing)", "(generic)", "(telling)", "(focus)"]:
                if dialog_act in turn:
                    turn = turn.replace(dialog_act, "")
                    break
            turns.append({
                "role": role,
                "text": turn,
                "dialog_act": dialog_act.replace("(", "").replace(")", ""),
            })
        for idx, turn in enumerate(turns):
            if turn["role"] == "teacher" and idx > 0:
                new_sample = {
                    "dataset_id": "mathdial",
                    "turns": turns[:idx],
                    "output": turns[idx]["text"],
                    "dialog_act": turns[idx]["dialog_act"],
                    "problem": sample["question"],
                    "student_profile": sample["student_profile"],
                    "incorrect_solution": sample["student_incorrect_solution"],
                    "ground_truth": sample["ground_truth"]
                }
                samples.append(new_sample)
        return samples

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        splits = ["train", "validation", "test"]
        hf_splits = [datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST]
        data = {split: [] for split in splits}
        # dataset = {split: [] for split in splits}

        dataset = load_dataset("eth-nlped/mathdial")
        train_len = .8 * len(dataset["train"])
        val_len = .8 * len(dataset["train"])
        for split in ["train", "test"]:
            for idx, sample in enumerate(dataset[split]):
                if split == "test":
                    data[split].extend(self._map_to_common_format(sample))
                else:
                    if idx < train_len:
                        data["train"].extend(self._map_to_common_format(sample))
                    else:
                        data["validation"].extend(self._map_to_common_format(sample))
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