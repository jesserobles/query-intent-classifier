"""
This module contains utilities for parsing the datasets that are in conll format for this project
"""
import os
import itertools
import logging
from pathlib import Path
from typing import Union

from datasets import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def get_all_tokens_and_ner_tags(directory):
    return pd.concat([get_tokens_and_ner_tags(os.path.join(directory, filename)) for filename in os.listdir(directory)]).reset_index().drop('index', axis=1)
    
def get_tokens_and_ner_tags(filename):
    with open(filename, 'r', encoding="utf8") as f:
        lines = f.readlines()
        split_list = [list(y) for x, y in itertools.groupby(lines, lambda z: z == '\n') if not x]
        tokens = [[x.split('\t')[0] for x in y] for y in split_list]
        entities = [[x.split('\t')[1][:-1] for x in y] for y in split_list] 
    return pd.DataFrame({'tokens': tokens, 'ner_tags': entities})
  
def get_un_token_dataset(train_directory, test_directory):
    train_df = get_all_tokens_and_ner_tags(train_directory)
    test_df = get_all_tokens_and_ner_tags(test_directory)
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    return (train_dataset, test_dataset)

class PairParser:
    """
    Class for parsing a text line and its associated CONLL information.
    TODO: Figure out the entity data that Huggingface requires for BERT
    """
    def __init__(self, text_line: str, conll_line: str) -> None:
        self.text_line: list[str] = text_line.strip().split()
        self.conll_line: list[str] = conll_line.strip().split()



class CONLLParser:
    """
    Class to parse a dataset (test, train). It should be able to output formats
    for both HuggingFace and Rasa.
    """
    EXPECTED_FILES = {'label', 'seq.in', 'seq.out'}
    def __init__(self, location: Union[Path, str]) -> None:
        if isinstance(location, str):
            location = Path(location)
        self.location: Path = location
        self.data = self.load()
        self.label_encoder = LabelEncoder().fit(self.data['label'])
    
    def load(self, location: Path=None):
        payload = {}
        location = location or self.location
        if not location:
            return
        for file in self.EXPECTED_FILES:
            file_location = location.joinpath(file)
            with open(file_location, encoding="utf-8") as f:
                data = [line.strip() for line in f.read().strip().splitlines()]
            payload[file] = data
        payload['label_list'] = self.generate_labels(payload['seq.out'])
        return payload
    
    def generate_labels(self, labels_seq):
        """
        Create all labels from a list of the form ['B-ORG', 'B-LOC', 'I-LOC']
        so that it adds missing data.
        """
        labels = sorted(set(lbl for line in labels_seq for lbl in line.split()),key=lambda x: x.split('-')[-1], reverse=True)
        labels.remove('O')
        cleaned_labels = ['O'] + labels
        label_encoding = {c: ix for ix, c in enumerate(cleaned_labels)}
        return {"labels_list": cleaned_labels, "label_encoding": label_encoding}


    def bert_ner_data(self):
        return pd.DataFrame({"tokens": [i.split() for i in self.data['seq.in']], "ner_tags": [o.split() for o in self.data["seq.out"]]})

    def bert_intent_data(self):
        label_dict = {label: ix for ix, label in enumerate(sorted(set(self.data['label'])))}
        return pd.DataFrame([{"label": self.label_encoder.transform([label])[0], "text": text} for label, text in zip(self.data['label'], self.data['seq.in'])])