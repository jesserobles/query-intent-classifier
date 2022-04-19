import os

from datasets import Dataset, load_dataset, load_metric
from datasets.dataset_dict import DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np

from dataprocessor.conll.conll import CoNLLParser


metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

dataset_name = "ATIS"

train = CoNLLParser(os.path.join('datasets', dataset_name, 'train'))
test = CoNLLParser(os.path.join('datasets', dataset_name, 'test'))
train_dataset = Dataset.from_pandas(train.bert_intent_data())
test_dataset = Dataset.from_pandas(train.bert_intent_data())

dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=len(train.label_encoder.classes_))

training_args = TrainingArguments(output_dir="test_trainer")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.evaluate()
trainer.save_model(os.path.join('huggingface-models', f'{dataset_name.lower()}-clf.model'))