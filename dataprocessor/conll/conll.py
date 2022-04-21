"""
This module contains utilities for parsing the datasets that are in conll format for this project
"""
from cProfile import label
from collections import defaultdict
import os
import itertools
import logging
from pathlib import Path
from typing import Union, List, Dict

from datasets import Dataset, DatasetDict, ClassLabel, Sequence
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ..base import BaseParser


class DatasetCombiner(BaseParser):
    """
    Class for parsing a text line and its associated CONLL information.
    """
    FOLDERS = {'train', 'test'}
    def __init__(self, location: Union[Path, str], mode:str="ner") -> None:
        super().__init__(location)
        self.payload = {}
        self.datasets = {}
        self.label_encoder = LabelEncoder()
        self.ner_label_encoder = LabelEncoder()
        func = "to_bert_ner_data"
        if mode != "ner":
            func = "bert_intent_data"
        all_labels = []
        all_ner_labels = []
        for folder in self.FOLDERS:
            parser = CoNLLParser(self.location.joinpath(folder), self.label_encoder)
            all_labels.extend(parser.data['label'])
            all_ner_labels.extend(parser.data.get('raw_ner_labels', ''))
            self.datasets[folder] = parser
        # Fit label encoder on entire dataset to ensure consistency
        self.label_encoder.fit(all_labels)
        self.ner_label_encoder.fit(all_ner_labels)
        for folder in self.FOLDERS:
            parser = self.datasets[folder]
            parser.reset_labels(self.ner_label_encoder)
            self.payload[folder] = getattr(parser, func)()
        self.dataset = DatasetDict(self.payload)


class CoNLLParser(BaseParser):
    """
    Class to parse a dataset (test, train). It should be able to output formats
    for both HuggingFace and Rasa.
    """
    EXPECTED_FILES = {'label', 'seq.in', 'seq.out'}
    def __init__(self, location: Union[Path, str], intent_label_encoder: LabelEncoder=None) -> None:
        super().__init__(location)
        self.data = self.load()
        self.intent_label_encoder = intent_label_encoder
        
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
            payload['raw_ner_labels'] = [lbl for line in payload['seq.out'] for lbl in line.split()]
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
    
    def reset_labels(self, ner_label_encoder: LabelEncoder):
        tags = []
        if not 'ner_tags_names' in self.data:
            return
        for lbl in self.data['ner_tags_names']:
            try:
                encoded = ner_label_encoder.transform(lbl).tolist()
            except ValueError:
                lbl = [l if l in ner_label_encoder.classes_ else 'O' for l in lbl]
                encoded = ner_label_encoder.transform(lbl).tolist()
            tags.append(encoded)
        # self.data['ner_tags'] = [ner_label_encoder.transform(lbl).tolist() for lbl in self.data['ner_tags_names']]
        self.data['ner_tags'] = tags
        self.data['ner_labels']['names'] = list(ner_label_encoder.classes_)
        self.data['ner_labels']['label_encoding'] = {l: ner_label_encoder.transform([l])[0] for l in ner_label_encoder.classes_}

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
        return Dataset.from_pandas(pd.DataFrame([{"label": self.intent_label_encoder.transform([label])[0], "text": text} for label, text in zip(self.data['label'], self.data['seq.in'])]))

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
    
    @staticmethod
    def get_spans(text):
        tokens = text.split()
        spans = []
        ix = 0
        for token in tokens:
            start = ix
            end = start + len(token)
            ix = end + 1
            spans.append((start, end))
        return spans

    @staticmethod
    def rasa_to_IOB(parsed):
        spans = CoNLLParser.get_spans(parsed['text'])
        labels = ['O']*len(spans)
        for entity in parsed['entities']:
            start = entity['start']
            end = entity['end']
            ent = entity["entity"]
            first_found = False
            for ix, (s, e) in enumerate(spans):
                prefix = "I" if first_found else "B"
                if s == start or e == end: # This will just pick up the appropriate token
                    if e < end: # This is the start token span, and the entity spills over it
                        first_found = True
                    label = f'{prefix}-{ent}'
                    labels[ix] = label
        return labels

"""
fields = [
    "text",
    "intent.name",
    "entities"
]
{
 "text": "i would like to find a flight from charlotte to las vegas that makes a stop in st. louis",
 "intent": {
  "name": "atis_flight",
  "confidence": 1.0
 },
 "entities": [
  {
   "entity": "fromloc.city_name",
   "start": 35,
   "end": 44,
   "confidence_entity": 0.9999704360961914,
   "value": "charlotte",
   "extractor": "DIETClassifier"
  },
  {
   "entity": "toloc.city_name",
   "start": 48,
   "end": 57,
   "confidence_entity": 0.999308705329895,
   "value": "las vegas",
   "extractor": "DIETClassifier"
  },
  {
   "entity": "stoploc.city_name",
   "start": 79,
   "end": 81,
   "confidence_entity": 0.9966257810592651,
   "value": "st",
   "extractor": "DIETClassifier"
  },
  {
   "entity": "stoploc.city_name",
   "start": 83,
   "end": 88,
   "confidence_entity": 0.9859499335289001,
   "value": "louis",
   "extractor": "DIETClassifier"
  }
 ],
 "text_tokens": [
  [
   0,
   1
  ],
  [
   2,
   7
  ],
  [
   8,
   12
  ],
  [
   13,
   15
  ],
  [
   16,
   20
  ],
  [
   21,
   22
  ],
  [
   23,
   29
  ],
  [
   30,
   34
  ],
  [
   35,
   44
  ],
  [
   45,
   47
  ],
  [
   48,
   51
  ],
  [
   52,
   57
  ],
  [
   58,
   62
  ],
  [
   63,
   68
  ],
  [
   69,
   70
  ],
  [
   71,
   75
  ],
  [
   76,
   78
  ],
  [
   79,
   81
  ],
  [
   83,
   88
  ]
 ],
 "intent_ranking": [
  {
   "name": "atis_flight",
   "confidence": 1.0
  },
  {
   "name": "atis_flight+atis_airfare",
   "confidence": 1.590412373596495e-22
  },
  {
   "name": "atis_airline",
   "confidence": 9.077638899810575e-23
  },
  {
   "name": "atis_quantity",
   "confidence": 6.554135341792252e-23
  },
  {
   "name": "atis_ground_fare",
   "confidence": 4.008480569555323e-23
  },
  {
   "name": "atis_abbreviation",
   "confidence": 3.9862173370881756e-23
  },
  {
   "name": "atis_flight_no",
   "confidence": 3.805358668336137e-23
  },
  {
   "name": "atis_capacity",
   "confidence": 3.243134030197139e-23
  },
  {
   "name": "atis_airport",
   "confidence": 3.0513414337543754e-23
  },
  {
   "name": "atis_ground_service",
   "confidence": 2.9761579407770195e-23
  }
 ],
 "response_selector": {
  "all_retrieval_intents": [],
  "default": {
   "response": {
    "responses": null,
    "confidence": 0.0,
    "intent_response_key": null,
    "utter_action": "utter_None"
   },
   "ranking": []
  }
 }
}
"""