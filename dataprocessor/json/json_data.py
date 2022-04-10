
from pathlib import Path
from typing import Union

from datasets import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder


"""
{
  "AddToPlaylist": [
    {
      "data": [
        {
          "text": "Add another "
        },
        {
          "text": "song",
          "entity": "music_item"
        },
        {
          "text": " to the "
        },
        {
          "text": "Cita Romantica",
          "entity": "playlist"
        },
        {
          "text": " playlist. "
        }
      ]
    }
  ]
}
"""


class JsonParser:
    def __init__(self, ) -> None:
        pass

    def convert_json_to_conll(self, record: dict) -> dict:
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
        return {"tokens": all_tokens, "ner_tags": labels}

