import os
import time

from datasets import load_dataset, load_metric
import numpy as np
import pandas as pd
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments

from dataprocessor.conll import CoNLLParser, DatasetCombiner

def tokenize_and_align_labels(examples, tokenizer, task="ner", label_all_tokens=True):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

metric = load_metric("seqeval")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

datasets = ["ATIS", "benchmarking_data", "SNIPS"]
for dataset_name in datasets:
    print(f"Training on dataset {dataset_name}")
    dataset_combiner = DatasetCombiner(os.path.join("datasets", dataset_name))
    dataset = dataset_combiner.dataset
    label_list = dataset["train"].features[f"ner_tags"].feature.names

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer})

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_list))

    training_args = TrainingArguments(
        output_dir=os.path.join("huggingface-models", "results", dataset_name),
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        
    )
    start = time.time()
    trainer.train()
    ellapsed = time.time() - start
    test_metrics = trainer.evaluate()
    ev_df = pd.DataFrame({"metric": k, "value": v} for k, v in test_metrics.items())
    ev_df = pd.concat([ev_df, pd.DataFrame({"metric": ["training_time"], "value": [ellapsed]})])
    ev_df.to_csv(os.path.join("results", "bert", f"{dataset_name}-ner-test.csv"), index=False)

    # Save the model
    trainer.save_model(os.path.join('huggingface-models', f'{dataset_name.lower()}-ner.model'))

    # Process validation dataset to evaluate
    validate_dataset = CoNLLParser(os.path.join("datasets", dataset_name, "valid"))
    validate_dataset.reset_labels(dataset_combiner.ner_label_encoder)
    validate_dataset = validate_dataset.to_bert_ner_data()

    # Evaluate on the validaton dataset
    tokenized_validate_dataset = validate_dataset.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer})
    valid_metrics = trainer.evaluate(tokenized_validate_dataset)

    ev_df = pd.DataFrame({"metric": k, "value": v} for k, v in valid_metrics.items())
    ev_df.to_csv(os.path.join("results", "bert", f"{dataset_name}-ner-valid.csv"), index=False)