import datasets
from datasets.arrow_dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from dataclasses import dataclass

import numpy as np


def load_dataset(path):
    instances = datasets.load_dataset("json", data_files=path)

    return instances["train"]


@dataclass
class MetaphorDatasetStats:
    total_instances: int
    metaphor_instances: int
    literal_instances: int
    total_sentences: int
    metaphoric_sentences: int

    metaphors_in_metaphoric_sentences: int

    @classmethod
    def from_dataset(cls, dataset: Dataset):
        total_instances = 0
        metaphor_instances = 0
        literal_instances = 0

        total_sentences = len(dataset)

        metaphoric_sentences = 0

        metaphors_in_metaphoric_sentences = 0

        for entry in dataset:
            if any(i["label"] == 0 for i in entry["instances"]):
                metaphors_in_metaphoric_sentences += len(entry["instances"])
                metaphoric_sentences += 1

            for instance in entry["instances"]:
                if instance["label"] == 0:
                    literal_instances += 1
                else:
                    metaphor_instances += 1
            total_instances += len(entry["instances"])

        return cls(
            total_instances=total_instances,
            metaphor_instances=metaphor_instances,
            literal_instances=literal_instances,
            total_sentences=total_sentences,
            metaphoric_sentences=metaphoric_sentences,
            metaphors_in_metaphoric_sentences=metaphors_in_metaphoric_sentences
        )

    def __str__(self):
        return f"""
Num. Tokens:    {self.total_instances}
%Met:           {(self.metaphor_instances / self.total_instances) * 100:.1f}
Num. Sents:     {self.total_sentences}
%Met Sents:     {self.metaphoric_sentences / self.total_sentences:.1f}
%Met per MetS.: {self.metaphors_in_metaphoric_sentences / self.metaphoric_sentences:.1f}
        """.strip()


def load_and_process_dataset(path, tokenizer):
    ds = load_dataset(path)

    stats = MetaphorDatasetStats.from_dataset(ds)
    print(stats)

    return preprocess_dataset(ds, tokenizer)

def preprocess_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizer):
    def process(elem):
        tokenized = tokenizer(elem["sentence"], padding="max_length", truncation=True, return_tensors="np")
        labels = np.full(tokenized.input_ids.shape, -100, dtype=np.int32)
        #labels[tokenized.attention_mask != 0] = -100

        for instance in elem["instances"]:
            labels[0, tokenized.char_to_token(instance["char_range"][0])] = instance["label"]

        return {"input_ids": tokenized.input_ids[0], "attention_mask": tokenized.attention_mask[0], "labels": labels}

    ds = dataset.map(
        process,
        remove_columns=dataset.column_names
    )

    return ds
