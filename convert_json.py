import os
from pathlib import Path

from dataprocessor import JsonParser

dataset_name = "benchmarking_data"
data_folder = os.path.join('datasets', dataset_name, 'Train')
dest_folder = Path(os.path.join('rasa-models', dataset_name.lower()))
dest_folder.mkdir(parents=True, exist_ok=True)
dest_file = dest_folder.joinpath(f'{dataset_name.lower()}.yml')

jparser = JsonParser(data_folder)
data = jparser.to_rasa_data()

with open(dest_file, 'w', encoding='utf-8') as file:
    file.write(data)
