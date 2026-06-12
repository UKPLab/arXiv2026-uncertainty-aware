import json
import os

from nlpeer import DATASETS, PAPERFORMATS, PaperReviewDataset
from pypdf import PdfReader

data = PaperReviewDataset("/path/to/data/", "nlpeer", version=1, paper_format=PAPERFORMATS.ITG)

final_data = {"data": []}

for sample in data:
    if len(sample[-1]) == 1:
        path = os.path.join("/path/to/data/nlpeer/data", sample[0])
        cycle = os.path.join(path, "meta.json")
        with open(cycle, "r") as f:
            cycle = json.load(f)["cycle"]
        path = os.path.join(path, "v1", "paper.pdf")
        try:
            reader = PdfReader(path)
            paper = "\n".join([reader.pages[idx].extract_text() for idx in range(min(8, len(reader.pages)))])
            new_sample = {
                "reviews": sample[-1],
                "cycle": cycle,
                "paper": paper
            }
            final_data["data"].append(new_sample)
        except:
            pass

print(len(final_data["data"]))
