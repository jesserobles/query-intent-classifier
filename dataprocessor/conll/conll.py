"""
This module contains utilities for parsing the datasets that are in conll format for this project
"""
from collections import defaultdict
import os
import itertools
import logging
from pathlib import Path
from typing import Union, List

from datasets import Dataset, DatasetDict, ClassLabel, Sequence
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ..base import BaseParser


class DatasetCombiner(BaseParser):
    """
    Class for parsing a text line and its associated CONLL information.
    """
    FOLDERS = {'train', 'test'}
    def __init__(self, location: Union[Path, str]) -> None:
        super().__init__(location)
        self.payload = {}
        self.datasets = {}
        for folder in self.FOLDERS:
            parser = CoNLLParser(self.location.joinpath(folder))
            self.datasets[folder] = parser
            self.payload[folder] = parser.to_bert_ner_data()
        self.dataset = DatasetDict(self.payload)


class CoNLLParser(BaseParser):
    """
    Class to parse a dataset (test, train). It should be able to output formats
    for both HuggingFace and Rasa.
    """
    EXPECTED_FILES = {'label', 'seq.in', 'seq.out'}
    def __init__(self, location: Union[Path, str]) -> None:
        super().__init__(location)
        self.seq_label_encoder = LabelEncoder()
        self.data = self.load()
        self.intent_label_encoder = LabelEncoder().fit(self.data['label'])
        
    def load(self, location: Path=None) -> dict:
        payload = {}
        location = location or self.location
        if not location:
            return
        for file in self.EXPECTED_FILES:
            file_location = location.joinpath(file)
            if not file_location.exists():
                continue
            with open(file_location, encoding="utf-8") as f:
                data = [line.strip() for line in f.read().strip().splitlines()]
            payload[file] = data
        payload['tokens'] = [text.split() for text in payload['seq.in']]
        if 'seq.out' in payload:
            payload['ner_labels'] = self.generate_labels(payload['seq.out'])
            payload['ner_tags_names'] = [line.split() for line in payload['seq.out']]
            payload['ner_tags'] = [[payload['ner_labels']['label_encoding'][tag] for tag in line.split()] for line in payload['seq.out']]
            if len(payload['label']) != len(payload['tokens']) != len(payload['ner_tags']):
                raise ValueError(f"Mismatched lengths of tokens and ner_tags: label={len(payload['label'])}, tokens={len(payload['tokens'])}, ner_tags={payload['ner_tags']}")
            # Validate that the tokens and labels are aligned
            for tokens, ner_tags in zip(payload['tokens'], payload['ner_tags']):
                if len(tokens) != len(ner_tags):
                    raise ValueError(f"Mismatched lengths of tokens and ner_tags: {len(tokens)} != {len(ner_tags)}")
        if len(payload['label']) != len(payload['tokens']) != len(payload['ner_tags']):
            raise ValueError(f"Mismatched lengths of tokens and ner_tags: label={len(payload['label'])}, tokens={len(payload['tokens'])}, ner_tags={payload['ner_tags']}")
        return payload
    
    def generate_labels(self, labels_seq) -> Dataset:
        """
        Create all labels from a list of the form ['B-ORG', 'B-LOC', 'I-LOC']
        so that it adds missing data.
        """
        labels = sorted(set(lbl for line in labels_seq for lbl in line.split()),key=lambda x: x.split('-')[-1], reverse=True)
        labels.remove('O')
        cleaned_labels = ['O'] + labels
        label_encoding = {c: ix for ix, c in enumerate(cleaned_labels)}
        # label_encoding = self.seq_label_encoder.fit_transform(cleaned_labels)
        return {"names": cleaned_labels, "label_encoding": label_encoding}

    def to_bert_ner_data(self):
        labels = self.data['ner_labels']
        tokens = self.data['tokens']
        ner_tags = ClassLabel(names=labels['names'])
        payload = {
            "id": range(len(self.data['seq.in'])),
            "tokens": tokens,
            "ner_tags": self.data['ner_tags'],
        }
        dataset = Dataset.from_dict(payload)
        dataset.features.update({"ner_tags": Sequence(feature=ner_tags)})
        return  dataset

    def bert_intent_data(self):
        return pd.DataFrame([{"label": self.intent_label_encoder.transform([label])[0], "text": text} for label, text in zip(self.data['label'], self.data['seq.in'])])

    def to_rasa_data(self) -> str:
        grouped_intents = defaultdict(list)
        intents = self.data['label']
        if not 'seq.out' in self.data:
            utterances = self.data['seq.in']
            payload = []
            for intent, utterance in zip(intents, utterances):
                grouped_intents[intent.replace('#', '+')].append(utterance)
            for intent, lines in grouped_intents.items():
                if len(lines) < 2:
                    continue
                examples = '\n    - '.join(lines)
                block = f'- intent: {intent}\n  examples: |\n    - {examples}\n'
                payload.append(block)
            return 'version: "3.1"\n\nnlu:\n' + ''.join(payload)
        tokens = self.data['tokens']
        ner_tags = self.data['ner_tags_names']
        for intent, tags, toks in zip(intents, ner_tags, tokens):
            grouped_intents[intent.replace('#', '+')].append(self.conll_to_rasa(toks, tags))
        payload = []
        for intent, lines in grouped_intents.items():
            if len(lines) < 2:
                continue
            examples = '\n    - '.join(lines)
            block = f'- intent: {intent}\n  examples: |\n    - {examples}\n'
            payload.append(block)
        return 'version: "3.1"\n\nnlu:\n' + ''.join(payload)

    @staticmethod
    def conll_to_rasa(tokens: List[str], ner_tags: List[str]) -> str:
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
