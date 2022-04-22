import os

from datasets import Dataset, load_dataset, load_metric
from datasets.dataset_dict import DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from dataprocessor.conll.conll import CoNLLParser, DatasetCombiner


sets = ["ATIS", "BANKING77", "benchmarking_data", "CLINC150", "HWU64", "SNIPS"]

dataset_name = "ATIS"
dataset_combiner = DatasetCombiner(os.path.join('datasets', dataset_name), mode="intent")

valid_dataset = CoNLLParser(os.path.join('datasets', dataset_name, 'valid'), intent_label_encoder=dataset_combiner.label_encoder).bert_intent_data()

tokenizer = AutoTokenizer.from_pretrained(os.path.join("huggingface-models", f"{dataset_name.lower()}-clf.model"))
model = AutoModelForSequenceClassification.from_pretrained(os.path.join("huggingface-models", f"{dataset_name.lower()}-clf.model"))

inputs = tokenizer(valid_dataset['text'], padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)
logits = outputs['logits'].detach().numpy()
predictions = np.argmax(logits, axis=-1)