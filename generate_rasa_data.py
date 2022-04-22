import os
from pathlib import Path

from tqdm import tqdm

from dataprocessor import CoNLLParser, DatasetCombiner
sets = ["ATIS", "BANKING77", "benchmarking_data", "CLINC150", "HWU64", "SNIPS"]
NO_NER = set(['BANKING77', 'CLINC150', 'HWU64'])
modes = {False: "ner", True: "intent"}
for dataset_name in tqdm(sets):
    mode = modes[dataset_name in NO_NER]
    data_folder = os.path.join('datasets', dataset_name)
    dest_folder = Path(os.path.join('rasa-models', dataset_name.lower()))
    dest_folder.mkdir(parents=True, exist_ok=True)
    dest_file = dest_folder.joinpath(f'{dataset_name.lower()}.yml')

    parser = DatasetCombiner(data_folder, mode=mode)
    data = parser.to_rasa_data()

    with open(dest_file, 'w', encoding='utf-8') as file:
        file.write(data)
