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


```bash
rasa train
rasa run --enable-api

python
>>> import requests
>>> r = requests.post('http://localhost:5005/model/parse', json={"text": "hi there"})
```