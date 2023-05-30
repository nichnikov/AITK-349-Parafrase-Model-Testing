import os
from src.texts_processing import TextsTokenizer
from src.config import (stopwords,
                        parameters,
                        logger,
                        PROJECT_ROOT_DIR)
from src.classifiers import FastAnswerClassifier
from sentence_transformers import SentenceTransformer

print("PROJECT_ROOT_DIR", PROJECT_ROOT_DIR)

model_name = "fast_answers85000"
# model_name = "expbot_paraphrase.transformers"
model = SentenceTransformer(os.path.join("models", model_name))
tokenizer = TextsTokenizer()
tokenizer.add_stopwords(stopwords)
classifier = FastAnswerClassifier(tokenizer, parameters, model)
logger.info("service started...")
