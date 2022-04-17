import os
import logging

from rasa.core.agent import Agent
from rasa.core.channels.channel import UserMessage
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dataprocessor import CoNLLParser


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

models = [
    'atis',
    'banking77',
    'clinc150',
    'hwu64',
    'snips'
]
with logging_redirect_tqdm():
    for model in models:
        model_path = model_path = os.path.join('rasa-models', 'models', f'{model}.tar.gz')

        agent = Agent()
        logger.info(f"Loading model {model}")
        agent.load_model(model_path=model_path)
        logger.info("Finished loading model.")
        data_path = os.path.join("datasets", f"{model.upper()}", "test")
        logger.info("Loading data")
        with open(os.path.join(data_path, "seq.in"), "r", encoding="utf-8") as file:
            texts = file.read().strip().split('\n')

        logger.info("Running inference")
        all_labels = []
        all_intents = []
        for text in tqdm(texts):
            message = UserMessage(text)
            parsed = agent.processor._parse_message_with_graph(message)
            labels = CoNLLParser.rasa_to_IOB(parsed)
            all_labels.append(' '.join(labels))
            intent = parsed['intent']['name']
            all_intents.append(intent)

        logger.info("Saving results")
        with open(os.path.join("results", f"{model}.out"), "w") as file:
            file.write("\n".join(all_labels))

        with open(os.path.join("results", f"{model}.label"), "w") as file:
            file.write("\n".join(all_intents))

# Now run on the json dataset (benchmarking_data)
# model = "benchmarking_data"
# model_path = model_path = os.path.join('rasa-models', 'models', f'{model}.tar.gz')

# agent = Agent()
# logger.info(f"Loading model {model}")
# agent.load_model(model_path=model_path)
# logger.info("Finished loading model.")
# data_path = os.path.join("datasets", f"{model.upper()}", "test")
# logger.info("Loading data")
# with open(os.path.join(data_path, "seq.in"), "r", encoding="utf-8") as file:
#     texts = file.read().strip().split('\n')

# logger.info("Running inference")
# all_labels = []
# all_intents = []
# for text in tqdm(texts):
#     message = UserMessage(text)
#     parsed = agent.processor._parse_message_with_graph(message)
#     labels = CoNLLParser.rasa_to_IOB(parsed)
#     all_labels.append(' '.join(labels))
#     intent = parsed['intent']['name']
#     all_intents.append(intent)

# logger.info("Saving results")
# with open(os.path.join("results", f"{model}.out"), "w") as file:
#     file.write("\n".join(all_labels))

# with open(os.path.join("results", f"{model}.label"), "w") as file:
#     file.write("\n".join(all_intents))