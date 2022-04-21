import os

from datasets import Dataset, load_dataset, load_metric
from datasets.dataset_dict import DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from dataprocessor.conll.conll import CoNLLParser, DatasetCombiner


metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

sets = ["ATIS", "BANKING77", "benchmarking_data", "CLINC150", ]
sets = ["HWU64", "SNIPS"]

for dataset_name in sets:
    dataset_combiner = DatasetCombiner(os.path.join('datasets', dataset_name), mode="intent")

    valid_dataset = CoNLLParser(os.path.join('datasets', dataset_name, 'valid'), intent_label_encoder=dataset_combiner.label_encoder).bert_intent_data()

    dataset = dataset_combiner.dataset

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)


    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(dataset_combiner.label_encoder.classes_))

    training_args = TrainingArguments(
        output_dir=os.path.join("huggingface-models", "results", dataset_name),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    test_metrics = trainer.evaluate()
        
    ev_df = pd.DataFrame({"metric": k, "value": v} for k, v in test_metrics.items())
    ev_df.to_csv(os.path.join("results", "bert", f"{dataset_name}-clf-test.csv"), index=False)

    # Save the model
    trainer.save_model(os.path.join('huggingface-models', f'{dataset_name.lower()}-clf.model'))

    # Evaluate on the validaton dataset
    tokenized_validate_dataset = valid_dataset.map(tokenize_function, batched=True)
    valid_metrics = trainer.evaluate(tokenized_validate_dataset)

    ev_df = pd.DataFrame({"metric": k, "value": v} for k, v in valid_metrics.items())
    ev_df.to_csv(os.path.join("results", "bert", f"{dataset_name}-clf-valid.csv"), index=False)