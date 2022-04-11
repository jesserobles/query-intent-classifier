import os

from datasets import load_dataset, load_metric
import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments

from dataprocessor.conll import DatasetCombiner

# def tokenize_and_align_labels(examples, tokenizer, key='tokens'):
#     is_split_into_words = False
#     if key == 'tokens':
#         is_split_into_words = True
#     tokenized_inputs = tokenizer(examples[key], truncation=True, is_split_into_words=is_split_into_words)
#     labels = []
#     for i, label in enumerate(examples[f"ner_tags"]):
#         word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
#         previous_word_idx = None
#         label_ids = []
#         for word_idx in word_ids:  # Set the special tokens to -100.
#             if word_idx is None:
#                 label_ids.append(-100)
#             elif word_idx != previous_word_idx:  # Only label the first token of a given word.
#                 label_ids.append(label[word_idx])
#             else:
#                 label_ids.append(-100)
#             previous_word_idx = word_idx
#         labels.append(label_ids)
#     tokenized_inputs["labels"] = labels
#     return tokenized_inputs

def tokenize_and_align_labels(examples, tokenizer, label_encoding_dict, key="tokens"):
    is_split_into_words = False
    if key == 'tokens':
        is_split_into_words = True
    label_all_tokens = True
    tokenized_inputs = tokenizer(list(examples[key]), truncation=True, is_split_into_words=is_split_into_words)
    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == '0':
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoding_dict[label[word_idx]])
            else:
                label_ids.append(label_encoding_dict[label[word_idx]] if label_all_tokens else -100)
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

# dataset = load_dataset("wnut_17")
dataset_combiner = DatasetCombiner(os.path.join("datasets", "ATIS"))
dataset = dataset_combiner.dataset
label_list = dataset["train"].features[f"ner_tags"].feature.names

label_encoding_dict = dataset_combiner.datasets['train'].data['ner_labels']['label_encoding']

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer, "key": "text", "label_encoding_dict": label_encoding_dict})

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_list))

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
trainer.train()