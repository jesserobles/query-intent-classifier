import json
import os
from pathlib import Path
import unittest

from dataprocessor import CoNLLParser, CsvParser, JsonParser


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

class TestDataProcessors(unittest.TestCase):
    def test_conll_to_rasa_line(self):
        folder = os.path.join('datasets', "ATIS", "test")
        with open(os.path.join(folder, "seq.in"), "r") as file:
            tokens = next(file).split()
        with open(os.path.join(folder, "seq.out"), "r") as file:
            ner_tags = next(file).split()
        expected_output = "i would like to find a flight from [charlotte](fromloc.city_name) to [las vegas](toloc.city_name) that makes a stop in [st. louis](stoploc.city_name)"
        output = CoNLLParser.conll_to_rasa(tokens, ner_tags)
        self.assertEqual(output, expected_output)
    
    def test_conll_to_rasa(self):
        dataset_name = "ATIS"
        data_folder = os.path.join('datasets', dataset_name, 'train')
        parser = CoNLLParser(data_folder)
        data = parser.to_rasa_data()
        prefix = 'version: "3.1"\n\nnlu:\n'
        self.assertTrue(data.startswith(prefix))
        data = data.replace(prefix, '').strip()
        self.assertTrue(data.startswith('- intent:'))

    def test_json_processor(self):
        with open(os.path.join('datasets', 'benchmarking_data', 'Train', 'AddToPlaylist.json'), "r") as file:
            records = json.load(file)
        record = records['AddToPlaylist'][0]
        expected_output_tokens = ["Add", "another", "song", "to", "the", "Cita", "Romantica", "playlist."]
        expected_ner_tags = ["O", "O", "B-music_item", "O", "O", "B-playlist", "I-playlist", "O"]
        parsed = JsonParser.convert_json_to_conll(record)
        self.assertEqual(expected_output_tokens, parsed["tokens"])
        self.assertEqual(expected_ner_tags, parsed["ner_tags"])

    def test_csv_to_rasa(self):
        datafile = os.path.join('datasets', 'search_trends', 'US_l1_vaccination_trending_searches.csv')
        parser = CsvParser(datafile)
        data = parser.to_rasa_data()
        prefix = 'version: "3.1"\n\nnlu:\n'
        self.assertTrue(data.startswith(prefix))
        data = data.replace(prefix, '').strip()
        self.assertTrue(data.startswith('- intent:'))
    
    def test_rasa_to_IOB(self):
        data_path = os.path.join("datasets", "atis", "test")
        with open(os.path.join(data_path, "test.json"), "r") as file:
            payload = json.load(file)['rasa_nlu_data']['common_examples']
        example = payload[0]
        labels = CoNLLParser.rasa_to_IOB(example)
        file = open(os.path.join(data_path, "seq.out"))
        expected_labels = next(file).split()
        self.assertEqual(labels, expected_labels)

        example = payload[1]
        labels = CoNLLParser.rasa_to_IOB(example)
        expected_labels = next(file).split()
        self.assertEqual(labels, expected_labels)

        example = payload[640]
        labels = CoNLLParser.rasa_to_IOB(example)
        file.close()
        expected_labels = open(os.path.join(data_path, "seq.out")).read().split('\n')[640].split()
        self.assertEqual(labels, expected_labels)
