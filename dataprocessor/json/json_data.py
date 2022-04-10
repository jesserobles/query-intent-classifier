import json
import os
from pathlib import Path
from typing import Union

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ..base import BaseParser


class JsonParser(BaseParser):
    def __init__(self, location: Union[Path, str]) -> None:
        super().__init__(location)

    def load(self, location: Path=None):
        records = []
        location = location or self.location
        if not location:
            return
        for file in location.iterdir():
            intent = os.path.splitext(os.path.basename(file))[0]
            with open(file, encoding="utf-8") as f:
                records.extend([self.convert_json_to_conll(record, intent) for record in json.load(f)[intent]])
        return records
    
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
        return {"tokens": all_tokens, "ner_tags": labels, "intent": intent}

