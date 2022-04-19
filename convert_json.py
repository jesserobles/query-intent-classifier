import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from dataprocessor import CoNLLParser, JsonParser

dataset_name = "benchmarking_data"
train_data_folder = os.path.join('datasets', dataset_name, 'train')
test_data_folder = os.path.join('datasets', dataset_name, 'valid')

# First convert to the CoNLL format
train_parser = JsonParser(train_data_folder)
test_parser = JsonParser(test_data_folder)

dest_train_data_folder = Path(os.path.join('datasets', dataset_name, 'train'))
dest_train_data_folder.mkdir(parents=True, exist_ok=True)
dest_test_data_folder = Path(os.path.join('datasets', dataset_name, 'test'))
dest_test_data_folder.mkdir(parents=True, exist_ok=True)
dest_valid_data_folder = Path(os.path.join('datasets', dataset_name, 'valid'))
dest_valid_data_folder.mkdir(parents=True, exist_ok=True)

# Combine train and valid sets, then split into train, test, valid with 0.7, 0.15, 0.15
df = pd.DataFrame({
    "seq.in": train_parser.data['seq.in'] + test_parser.data['seq.in'],
    "seq.out": train_parser.data['seq.out'] + test_parser.data['seq.out'],
    "label": train_parser.data['label'] + test_parser.data['label'],
}).sample(frac=1).reset_index(drop=True)

# In the first step we will split the data in training and remaining dataset
X_train, X_rem, y_train, y_rem = train_test_split(df,df['label'], train_size=0.8, stratify=df['label'])

# Now since we want the valid and test size to be equal (10% each of overall data). 
# we have to define valid_size=0.5 (that is 50% of remaining data)
test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5, stratify=X_rem['label'])

# Training set
with open(dest_train_data_folder.joinpath("seq.in"), "w", encoding='utf-8') as file:
    file.write('\n'.join(X_train['seq.in']))

with open(dest_train_data_folder.joinpath("label"), "w", encoding='utf-8') as file:
    file.write('\n'.join(X_train['label']))

with open(dest_train_data_folder.joinpath("seq.out"), "w", encoding='utf-8') as file:
    file.write('\n'.join(X_train['seq.out']))

# Test set
with open(dest_test_data_folder.joinpath("seq.in"), "w", encoding='utf-8') as file:
    file.write('\n'.join(X_test['seq.in']))

with open(dest_test_data_folder.joinpath("label"), "w", encoding='utf-8') as file:
    file.write('\n'.join(X_test['label']))

with open(dest_test_data_folder.joinpath("seq.out"), "w", encoding='utf-8') as file:
    file.write('\n'.join(X_test['seq.out']))

# Valid set
with open(dest_valid_data_folder.joinpath("seq.in"), "w", encoding='utf-8') as file:
    file.write('\n'.join(X_valid['seq.in']))

with open(dest_valid_data_folder.joinpath("label"), "w", encoding='utf-8') as file:
    file.write('\n'.join(X_valid['label']))

with open(dest_valid_data_folder.joinpath("seq.out"), "w", encoding='utf-8') as file:
    file.write('\n'.join(X_valid['seq.out']))

# Now read using CoNLL
train_parser = CoNLLParser(dest_test_data_folder)
data = train_parser.to_rasa_data()

dest_folder = Path(os.path.join('rasa-models', dataset_name.lower()))
dest_folder.mkdir(parents=True, exist_ok=True)
dest_file = dest_folder.joinpath(f'{dataset_name.lower()}.yml')

with open(dest_file, 'w', encoding='utf-8') as file:
    file.write(data)