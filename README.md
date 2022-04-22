# Query Intent and Entity Classifier Benchmarking

# Datasets
- ATIS - Airline travel information systems: Contains intents and entities. The dataset is also available in the rasa format
- BANKING77 - Banking Intents (no entities)
- BANKING77-OOS - Same as previous but with out of scope and out of domain intents.
- benchmarking_data - [kaggle dataset](https://www.kaggle.com/datasets/joydeb28/nlp-benchmarking-data-for-intent-and-entity) for - benchmarking intents and entities
- CLINC150 - A dataset with 10 domains and 150 intents (no entities)
- CLINC-Single-Domain-OOS - CLINC150 reduced to a single domain (no entities)
- HWU64 - Personal assistant data with 64 intents and several domains (no entities)
- SNIPS - Personal assistant data with 7 intents. Includes entities. Need to parse similar to the ATIS format, which is in [CONLL](https://nlpforge.com/2021/07/13/data-annotation-for-named-entity-recognition-part-1/) format
- search_trends - [Google search trends](https://storage.googleapis.com/covid19-open-data/covid19-vaccination-search-insights/top_queries/US_l1_vaccination_trending_searches.csv) for covid vaccines and some intents.

The datasets are available in the datasets.zip file. Once unzipped, all of the folder should contain three folders: `test`, `train`, `valid`. Rename any folders to match that case. Some of the data will be generated and placed in the folders as you run the scripts.

## Train rasa model
```bash
rasa train
rasa run --enable-api
```

## NOTE
I modified the rasa WhitespaceTokenizer to simply split on whitespace. The original implementation uses a regular expression to handle special characters, but in order to align with the datasets it's necessary to split on whitespace.

## Interact via rasa API
```python
>>> import requests
>>> r = requests.post('http://localhost:5005/model/parse', json={"text": "hi there"})
```


## Interact via python
```python
import os
from rasa.core.agent import Agent
from rasa.core.channels.channel import UserMessage

model_path = model_path = os.path.join('rasa-models', 'models', '20220409-202441-old-falloff.tar.gz')

agent = Agent()
agent.load_model(model_path=model_path)

message = UserMessage("hello")
agent.processor._parse_message_with_graph(message)
```

## Train a model with a fixed name (doesn't work on windows)
```bash
rasa train --data atis/ -c config.yml -d domain.yml --out out/ --fixed-model-name foo nlu

rasa train --data nlu_data/ -c config.yml -d domain.yml --fixed-model-name foo nlu
```

## Generate Rasa Dataset from CoNLL Format
```python
import os
from pathlib import Path

from dataprocessor import DatasetCombiner

dataset_name = "SNIPS"
data_folder = os.path.join('datasets', dataset_name)
dest_folder = Path(os.path.join('rasa-models', dataset_name.lower()))
dest_folder.mkdir(parents=True, exist_ok=True)
dest_file = dest_folder.joinpath(f'{dataset_name.lower()}.yml')

parser = DatasetCombiner(data_folder, mode=mode)
data = parser.to_rasa_data()

with open(dest_file, 'w', encoding='utf-8') as file:
    file.write(data)

```

## Generate Rasa Dataset from JSON Format
Note that you should run this to convert the json data to CoNLL format before converting to rasa.
```python
import os
from pathlib import Path

from dataprocessor import JsonParser

dataset_name = "benchmarking_data"
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
```
## Generate Rasa Dataset from CSV Format
```python
import os
from pathlib import Path

from dataprocessor import CsvParser

dataset_name = "search_trends"
data_folder = os.path.join('datasets', 'search_trends', 'US_l1_vaccination_trending_searches.csv')
train_dest_folder = Path(os.path.join('datasets', 'search_trends', 'train'))
test_dest_folder = Path(os.path.join('datasets', 'search_trends', 'test'))

train_dest_folder.mkdir(parents=True, exist_ok=True)
test_dest_folder.mkdir(parents=True, exist_ok=True)

dest_folder = Path(os.path.join('rasa-models', dataset_name.lower()))
dest_file = dest_folder.joinpath(f'{dataset_name.lower()}.yml')
dest_folder.mkdir(parents=True, exist_ok=True)

parser = CsvParser(data_folder)
data = parser.to_rasa_data()

with open(dest_file, 'w', encoding='utf-8') as file:
    file.write(data)
```

```python
import os
from rasa.core.agent import Agent
from rasa.core.channels.channel import UserMessage

model = 'atis'
model_path = model_path = os.path.join('rasa-models', 'models', f'{model}.tar.gz')

agent = Agent()
agent.load_model(model_path=model_path)

text = "i would like to find a flight from charlotte to las vegas that makes a stop in st. louis"
message = UserMessage(text)
parsed = agent.processor._parse_message_with_graph(message)

```

## Rasa model training times
ATIS: 15:00 (estimate with enough GPU memory), 3:41:50 on CPU
BANKING77: 5:55
benchmarking_data: 9:47
CLINC150: 5:20
HWU64: 2:42
SNIPS: 10:05