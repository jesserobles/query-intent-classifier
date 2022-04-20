import os

from datasets import load_metric
# from seqeval.metrics import accuracy_score
# from seqeval.metrics import classification_report
# from seqeval.metrics import f1_score

data_path = os.path.join("datasets", "ATIS", "test")
with open(os.path.join("results", "atis.out"), "r") as file:
    y_pred = [line.strip().split() for line in file]

with open(os.path.join(data_path, "seq.out"), "r") as file:
    y_true = [line.strip().split() for line in file]

# f1_score(y_true, y_pred)

metric = load_metric("seqeval")
results = metric.compute(predictions=y_pred, references=y_true)
print(
{
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    })