import os
from pathlib import Path

from dataprocessor import JsonParser

dataset_name = "BENCHMARKING_DATA"
train_data_folder = os.path.join('datasets', dataset_name, 'train')
test_data_folder = os.path.join('datasets', dataset_name, 'test')
valid_data_folder = os.path.join('datasets', dataset_name, 'valid')
# First convert to the CoNLL format
train_parser = JsonParser(train_data_folder)
test_parser = JsonParser(test_data_folder)
valid_parser = JsonParser(valid_data_folder)

dest_train_data_folder = Path(os.path.join('datasets', dataset_name, 'train'))
dest_train_data_folder.mkdir(parents=True, exist_ok=True)
dest_test_data_folder = Path(os.path.join('datasets', dataset_name, 'test'))
dest_test_data_folder.mkdir(parents=True, exist_ok=True)
dest_valid_data_folder = Path(os.path.join('datasets', dataset_name, 'valid'))
dest_valid_data_folder.mkdir(parents=True, exist_ok=True)

# Training set
with open(dest_train_data_folder.joinpath("seq.in"), "w", encoding='utf-8') as file:
    file.write('\n'.join(train_parser.data['seq.in']))

with open(dest_train_data_folder.joinpath("label"), "w", encoding='utf-8') as file:
    file.write('\n'.join(train_parser.data['label']))

with open(dest_train_data_folder.joinpath("seq.out"), "w", encoding='utf-8') as file:
    file.write('\n'.join(train_parser.data['seq.out']))

# Test set
with open(dest_test_data_folder.joinpath("seq.in"), "w", encoding='utf-8') as file:
    file.write('\n'.join(test_parser.data['seq.in']))

with open(dest_test_data_folder.joinpath("label"), "w", encoding='utf-8') as file:
    file.write('\n'.join(test_parser.data['label']))

with open(dest_test_data_folder.joinpath("seq.out"), "w", encoding='utf-8') as file:
    file.write('\n'.join(test_parser.data['seq.out']))

# Valid set
with open(dest_valid_data_folder.joinpath("seq.in"), "w", encoding='utf-8') as file:
    file.write('\n'.join(test_parser.data['seq.in']))

with open(dest_valid_data_folder.joinpath("label"), "w", encoding='utf-8') as file:
    file.write('\n'.join(test_parser.data['label']))

with open(dest_valid_data_folder.joinpath("seq.out"), "w", encoding='utf-8') as file:
    file.write('\n'.join(test_parser.data['seq.out']))

data = train_parser.to_rasa_data()

dest_folder = Path(os.path.join('rasa-models', dataset_name.lower()))
dest_folder.mkdir(parents=True, exist_ok=True)
dest_file = dest_folder.joinpath(f'{dataset_name.lower()}.yml')

with open(dest_file, 'w', encoding='utf-8') as file:
    file.write(data)