import os
from rasa.core.agent import Agent
from rasa.core.channels.channel import UserMessage
from tqdm import tqdm

from dataprocessor import CoNLLParser


model = 'atis'
model_path = model_path = os.path.join('rasa-models', 'models', f'{model}.tar.gz')

agent = Agent()
agent.load_model(model_path=model_path)

data_path = os.path.join("datasets", "ATIS", "test")
with open(os.path.join(data_path, "seq.in"), "r", encoding="utf-8") as file:
    texts = file.read().strip().split('\n')

all_labels = []
all_intents = []
for text in tqdm(texts):
    message = UserMessage(text)
    parsed = agent.processor._parse_message_with_graph(message)
    labels = CoNLLParser.rasa_to_IOB(parsed)
    all_labels.append(' '.join(labels))
    intent = parsed['intent']['name']
    all_intents.append(intent)

with open(os.path.join("results", "atis.out"), "w") as file:
    file.write("\n".join(all_labels))

with open(os.path.join("results", "atis.label"), "w") as file:
    file.write("\n".join(all_intents))


import os

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

data_path = os.path.join("datasets", "ATIS", "test")
with open(os.path.join("results", "atis.out"), "r") as file:
    y_pred = [line.strip().split() for line in file]

with open(os.path.join(data_path, "seq.out"), "r") as file:
    y_true = [line.strip().split() for line in file]

# y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
# y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
# f1_score(y_true, y_pred)