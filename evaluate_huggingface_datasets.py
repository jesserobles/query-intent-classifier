import os
import time

from datasets import Dataset, load_dataset, load_metric
from datasets.dataset_dict import DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from dataprocessor.conll.conll import CoNLLParser, DatasetCombiner

def get_metrics(y_true, y_pred):
    return {
        "eval_precision": precision_score(y_true, y_pred),
        "eval_recall": recall_score(y_true, y_pred),
        "eval_f1": f1_score(y_true, y_pred),
        "eval_accuracy":  accuracy_score(y_true, y_pred)
    }


sets = ["ATIS", "BANKING77", "benchmarking_data", "CLINC150", "HWU64", "SNIPS"]
NO_NER = set(['BANKING77', 'CLINC150', 'HWU64'])
start = time.time()
dataset_name = "ATIS"
for dataset_name in sets:

    # Now get the intent inferences
    dataset_combiner = DatasetCombiner(os.path.join('datasets', dataset_name), mode="intent")
    # Load the validation dataset
    valid_dataset = CoNLLParser(os.path.join('datasets', dataset_name, 'valid'), intent_label_encoder=dataset_combiner.label_encoder).bert_intent_data()
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(os.path.join("huggingface-models", f"{dataset_name.lower()}-clf.model"))
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join("huggingface-models", f"{dataset_name.lower()}-clf.model"))
    # Tokenize the inputs
    inputs = tokenizer(valid_dataset['text'], padding=True, truncation=True, return_tensors="pt")
    # Make the inferences
    outputs = model(**inputs)
    # Convert logits to class labels
    logits = outputs['logits'].detach().numpy()
    y_pred = np.argmax(logits, axis=-1)
    # Get classification report
    y_true = valid_dataset['label']
    report = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose()
    report.to_csv(os.path.join("results", "bert", f"{dataset_name.lower()}-clf-full.csv"))

print(f"Ellapsed: {time.time() - start}")