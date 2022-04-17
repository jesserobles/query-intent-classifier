import os

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

data_path = os.path.join("datasets", "ATIS", "test")
with open(os.path.join("results", "atis.out"), "r") as file:
    y_pred = [line.strip().split() for line in file]

with open(os.path.join(data_path, "seq.out"), "r") as file:
    y_true = [line.strip().split() for line in file]

f1_score(y_true, y_pred)