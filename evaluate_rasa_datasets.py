import os
import time

from datasets import load_metric
import pandas as pd
from sklearn.metrics import classification_report


ner_metric = load_metric("seqeval")

sets = ["ATIS", "BANKING77", "benchmarking_data", "CLINC150", "HWU64", "SNIPS"]
NO_NER = set(['BANKING77', 'CLINC150', 'HWU64'])
start = time.time()
for dataset_name in sets:
    rasa_path = os.path.join("results", "rasa")
    data_path = os.path.join("datasets", dataset_name, "valid")

    if dataset_name not in NO_NER:
        print("Running NER")
        # First the NER data
        
        with open(os.path.join(rasa_path, f"{dataset_name.lower()}.out"), "r") as file:
            y_pred = [line.strip().split() for line in file]

        
        with open(os.path.join(data_path, "seq.out"), "r") as file:
            y_true = [line.strip().split() for line in file]


        ner_results = pd.DataFrame(ner_metric.compute(predictions=y_pred, references=y_true))
        ner_results.to_csv(os.path.join("results", "rasa", f"{dataset_name.lower()}-ner.csv"))

    # Now the classification data
    with open(os.path.join(rasa_path, f"{dataset_name.lower()}.label"), "r") as file:
        y_pred = [line.strip().split() for line in file]

    with open(os.path.join(data_path, "label"), "r") as file:
        y_true = [line.strip().split() for line in file]

    report = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose()
    report.to_csv(os.path.join("results", "rasa", f"{dataset_name.lower()}-clf.csv"))

print(f"Ellapsed: {time.time() - start}")