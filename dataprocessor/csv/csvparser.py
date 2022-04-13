from collections import defaultdict
from csv import DictReader
from pathlib import Path
from typing import Union

from ..base import BaseParser

class CsvParser(BaseParser):
    def __init__(self, location: Union[Path, str]) -> None:
        super().__init__(location)
        self.data = self.load()
    
    def load(self) -> dict:
        queries = defaultdict(set)
        labels = []
        samples = []
        with open(self.location, encoding='utf-8') as csvfile:
            reader = DictReader(csvfile)
            for row in reader:
                queries[row['query']].add(row['category']) 
        for query, categories in queries.items():
            intent = '+'.join(sorted(categories))
            labels.append(intent)
            samples.append(query)

        return {
            'label': labels,
            'seq.in': samples,
        }
    
    def to_rasa_data(self) -> str:
        grouped_intents = defaultdict(list)
        intents = self.data['label']
        utterances = self.data['seq.in']
        payload = []
        for intent, utterance in zip(intents, utterances):
            grouped_intents[intent].append(utterance)
        for intent, lines in grouped_intents.items():
            if len(lines) < 2:
                continue
            examples = '\n    - '.join(lines)
            block = f'- intent: {intent}\n  examples: |\n    - {examples}\n'
            payload.append(block)
        return 'version: "3.1"\n\nnlu:\n' + ''.join(payload)

    
