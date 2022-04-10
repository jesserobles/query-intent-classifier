import os
import itertools
import logging
from pathlib import Path
from typing import Union


class BaseParser:
    """
    Class to parse a dataset (test, train). It should be able to output formats
    for both HuggingFace and Rasa.
    """
    EXPECTED_FILES = None
    def __init__(self, location: Union[Path, str]) -> None:
        if isinstance(location, str):
            location = Path(location)
        self.location: Path = location