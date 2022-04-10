import json
import os
import unittest

from dataprocessor.conll import CoNLLParser
from dataprocessor.json import JsonParser


class TestDataProcessors(unittest.TestCase):
    def test_conll_to_rasa(self):
        folder = os.path.join('datasets', "ATIS", "test")
        with open(os.path.join(folder, "seq.in"), "r") as file:
            tokens = next(file).split()
        with open(os.path.join(folder, "seq.out"), "r") as file:
            ner_tags = next(file).split()
        expected_output = "i would like to find a flight from [charlotte](fromloc.city_name) to [las vegas](toloc.city_name) that makes a stop in [st. louis](stoploc.city_name)"
        output = CoNLLParser.conll_to_rasa(tokens, ner_tags)
        self.assertEqual(output, expected_output)

    def test_json_processor(self):
        with open(os.path.join('datasets', 'benchmarking_data', 'Train', 'AddToPlaylist.json'), "r") as file:
            records = json.load(file)
        record = records['AddToPlaylist'][0]
        expected_output_tokens = ["Add", "another", "song", "to", "the", "Cita", "Romantica", "playlist."]
        expected_ner_tags = ["O", "O", "B-music_item", "O", "O", "B-playlist", "I-playlist", "O"]
        parsed = JsonParser.convert_json_to_conll(record)
        self.assertEqual(expected_output_tokens, parsed["tokens"])
        self.assertEqual(expected_ner_tags, parsed["ner_tags"])