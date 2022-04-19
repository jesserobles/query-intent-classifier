import json
import os
from pathlib import Path
from typing import Union

from datasets import Dataset, DatasetDict, ClassLabel, Sequence
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ..base import BaseParser
from ..conll import CoNLLParser


class JsonParser(CoNLLParser):
    def __init__(self, location: Union[Path, str]) -> None:
        super().__init__(location)
        # self.data = self.load()

    def load(self, location: Path=None):
        payload = {}
        seq_in = []
        seq_out = []
        labels = []
        tokens = []
        location = location or self.location
        if not location:
            return
        for file in location.glob('*.json'):
            intent = os.path.splitext(os.path.basename(file))[0]
            with open(file, encoding="utf-8") as f:
                for record in json.load(f)[intent]:
                    pl = self.convert_json_to_conll(record, intent)
                    tokens.append(pl["tokens"])
                    seq_in.append(" ".join(pl["tokens"]))
                    seq_out.append(pl["ner_tags"])
                    labels.append(pl["intent"])
                # records.extend([self.convert_json_to_conll(record, intent) for record in json.load(f)[intent]])
        payload['seq.in'] = seq_in
        payload['seq.out'] = seq_out
        payload['tokens'] = tokens
        payload['label'] = labels
        payload['ner_labels'] = self.generate_labels(payload['seq.out'])
        payload['ner_tags_names'] = [line.split() for line in payload['seq.out']]
        payload['ner_tags'] = [[payload['ner_labels']['label_encoding'][tag] for tag in line.split()] for line in payload['seq.out']]
        return payload

    def generate_labels(self, labels_seq) -> Dataset:
        """
        Create all labels from a list of the form ['B-ORG', 'B-LOC', 'I-LOC']
        so that it adds missing data.
        """
        labels = sorted(set(lbl for line in labels_seq for lbl in line.split()),key=lambda x: x.split('-')[-1], reverse=True)
        try:
            labels.remove('O')
        except ValueError:
            print(labels)
            pass
        cleaned_labels = ['O'] + labels
        label_encoding = {c: ix for ix, c in enumerate(cleaned_labels)}
        # label_encoding = self.seq_label_encoder.fit_transform(cleaned_labels)
        return {"names": cleaned_labels, "label_encoding": label_encoding}
    

    @staticmethod
    def convert_json_to_conll(record: dict, intent: str=None) -> dict:
        """
        Method to convert the json format into ConLL format.
        We need to account for the B-ENT and I-ENT formatting.
        The input records contain all tokens in an entity within
        the "text" field. We also have records with just space values.
        We need to ignore those for entity purposes.
        """
        data = record["data"]
        all_tokens = []
        labels = []
        for item in data:
            text = item["text"].strip()
            if not text:
                continue
            tokens = text.split()
            all_tokens.extend(tokens)
            label = item.get("entity", "O")
            if label == "O":
                labels.extend([label]*len(tokens))
            else:
                # There is an entity. We need to split tokens on whitespace, prepend B and I
                labels.extend([f"B-{label}"] + [f"I-{label}"]*(len(tokens) - 1))
        if len(all_tokens) != len(labels):
            raise ValueError("Length mismatch between tokens and labels")
        return {"tokens": all_tokens, "ner_tags": ' '.join(labels), "intent": intent}

