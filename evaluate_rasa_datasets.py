import os

from datasets import load_metric
from sklearn.metrics import classification_report


ner_metric = load_metric("seqeval")

sets = ["ATIS", "BANKING77", "benchmarking_data", "CLINC150", "HWU64", "SNIPS"]
NO_NER = set(['BANKING77', 'CLINC150', 'HWU64'])

dataset_name = "ATIS"
# dataset_name = 'SNIPS'

rasa_path = os.path.join("results", "rasa")
data_path = os.path.join("datasets", dataset_name, "valid")

if dataset_name not in NO_NER:
    print("Running NER")
    # First the NER data
    
    with open(os.path.join(rasa_path, f"{dataset_name.lower()}.out"), "r") as file:
        y_pred = [line.strip().split() for line in file]

    
    with open(os.path.join(data_path, "seq.out"), "r") as file:
        y_true = [line.strip().split() for line in file]


    ner_results = ner_metric.compute(predictions=y_pred, references=y_true)

# Now the classification data
with open(os.path.join(rasa_path, f"{dataset_name.lower()}.label"), "r") as file:
    y_pred = [line.strip().split() for line in file]

with open(os.path.join(data_path, "label"), "r") as file:
    y_true = [line.strip().split() for line in file]

print(classification_report(y_true, y_pred))

with open(os.path.join(rasa_path, f"{dataset_name.lower()}.out"), "r") as file:
    y_pred = [line.strip().split() for line in file]


with open(os.path.join(data_path, "seq.out"), "r") as file:
    y_true = [line.strip().split() for line in file]

for ix, (yt, yp) in enumerate(zip(y_true, y_pred)):
    if len(yt) != len(yp):
        print(ix)