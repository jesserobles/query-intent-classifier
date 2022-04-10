"""
This module contains utilities for parsing the datasets that are in conll format for this project
"""
import os
import itertools
import logging
from pathlib import Path
from typing import Union

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ..base import BaseParser


class PairParser:
    """
    Class for parsing a text line and its associated CONLL information.
    TODO: Figure out the entity data that Huggingface requires for BERT
    """
    def __init__(self, text_line: str, conll_line: str) -> None:
        self.text_line: list[str] = text_line.strip().split()
        self.conll_line: list[str] = conll_line.strip().split()



class CoNLLParser(BaseParser):
    """
    Class to parse a dataset (test, train). It should be able to output formats
    for both HuggingFace and Rasa.
    """
    EXPECTED_FILES = {'label', 'seq.in', 'seq.out'}
    def __init__(self, location: Union[Path, str]) -> None:
        # if isinstance(location, str):
        #     location = Path(location)
        # self.location: Path = location
        super.__init__(location)
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
        return pd.DataFrame([{"label": self.label_encoder.transform([label])[0], "text": text} for label, text in zip(self.data['label'], self.data['seq.in'])])

    @staticmethod
    def conll_to_rasa(tokens: list[str], ner_tags: list[str]) -> str:
        """
        Method to convert from ConLL format to a string that can be
        used in a rasa yaml file.
        tokens = ["i", "would", "like", "to", "find", "a", "flight", "from", "charlotte", "to", "las", "vegas", "that", "makes", "a", "stop", "in", "st.", "louis"]
        ner_tags = ["O", "O", "O", "O", "O", "O", "O", "O", "B-fromloc.city_name", "O", "B-toloc.city_name", "I-toloc.city_name", "O", "O", "O", "O", "O", "B-stoploc.city_name", "I-stoploc.city_name"]
        output = "i would like to find a flight from [charlotte](fromloc.city_name) to [las vegas](toloc.city_name) that makes a stop in [st. louis](stoploc.city_name)"
        """
        annotated_tokens = []
        current_entity_tokens = [] # We'll use this to keep track of entities
        current_entity_tag = None
        for token, tag in zip(tokens, ner_tags):
            if current_entity_tokens: # We are in the middle of an entity
                if tag == 'O' or tag.startswith('B'): # We are outside of the previous entity
                    annotated_token = f"[{' '.join(current_entity_tokens)}]({current_entity_tag})"
                    annotated_tokens.append(annotated_token)
                    current_entity_tokens = []
                    current_entity_tag = None
                else: # Just append the token
                    current_entity_tokens.append(token)
                    continue
            if tag != 'O': # Entity
                current_entity_tag = tag.split('-')[-1]
                current_entity_tokens.append(token)
            else: # Just append the token
                annotated_tokens.append(token)
        # Append any remaining tokens
        if current_entity_tokens:
            annotated_token = f"[{' '.join(current_entity_tokens)}]({current_entity_tag})"
            annotated_tokens.append(annotated_token)
        return ' '.join(annotated_tokens)
