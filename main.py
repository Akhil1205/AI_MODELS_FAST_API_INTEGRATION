from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, BertForSequenceClassification, RobertaForSequenceClassification, BertForTokenClassification, BertForQuestionAnswering, DistilBertForQuestionAnswering, BartForConditionalGeneration, T5ForConditionalGeneration
from functools import lru_cache
# from fastapi.responses import JSONResponse
# import logging
# import secrets
# from fastapi.security import HTTPBasic, HTTPBasicCredentials
# from fastapi_limiter import FastAPILimiter
# from fastapi_limiter.depends import RateLimiter
# import aioredis

app = FastAPI()

# Load models
models = {
    "bert-base-uncased": pipeline("text-classification", model=BertForSequenceClassification.from_pretrained("bert-base-uncased"), tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")),
    "roberta-base": pipeline("text-classification", model=RobertaForSequenceClassification.from_pretrained("roberta-base"), tokenizer=AutoTokenizer.from_pretrained("roberta-base")),
    "bert-base-cased-ner-conll03": pipeline("ner", model=BertForTokenClassification.from_pretrained("dbmdz/bert-base-cased-finetuned-conll03-english"), tokenizer=AutoTokenizer.from_pretrained("dbmdz/bert-base-cased-finetuned-conll03-english")),
    "bert-base-ner": pipeline("ner", model=BertForTokenClassification.from_pretrained("dslim/bert-base-NER"), tokenizer=AutoTokenizer.from_pretrained("dslim/bert-base-NER")),
    # Question Answering Models
    "bert-large-uncased-whole-word-masking-finetuned-squad": pipeline("question-answering", model=BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad"), tokenizer=AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")),
    "distilbert-base-cased-distilled-squad": pipeline("question-answering", model=DistilBertForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad"), tokenizer=AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")),

    # Summarization Models
    "t5-small": pipeline("summarization", model=T5ForConditionalGeneration.from_pretrained("t5-small"), tokenizer=AutoTokenizer.from_pretrained("t5-small")),
    "bart-base": pipeline("summarization", model=BartForConditionalGeneration.from_pretrained("facebook/bart-base"), tokenizer=AutoTokenizer.from_pretrained("facebook/bart-base")),
}


class InputText(BaseModel):
    text: str

class InputQuestion(BaseModel):
    context: str
    question: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the NLP Model Comparison API"}

@app.get("/health")
def health_check():
    return {"status": "OK"}

@app.get("/models")
def get_models():
    return {"models": list(models.keys())}

@app.post("/classify")
def classify_text(input_text: InputText):
    results = {}
    for model_name, model in models.items():
        if "text-classification" in model.task:
            results[model_name] = model(input_text.text)
    return results

import numpy as np

def convert_to_python_type(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_type(i) for i in obj]
    else:
        return obj

@app.post("/ner")
def recognize_entities(input_text: InputText):
    results = {}
    for model_name, model in models.items():
        if "ner" in model.task:
            try:
                entities = model(input_text.text)
                results[model_name] = convert_to_python_type([{"entity": entity['entity'], "score": entity['score'], "index": entity.get('index', None), "start": entity['start'], "end": entity['end'], "word": entity['word']} for entity in entities])
            except Exception as e:
                results[model_name] = {"error": str(e)}
    return results

@app.post("/answer")
def answer_question(input_question: InputQuestion):
    results = {}
    for model_name, model in models.items():
        if "question-answering" in model.task:
            results[model_name] = model(question=input_question.question, context=input_question.context)
    return results

@app.post("/summarize")
def summarize_text(input_text: InputText):
    results = {}
    for model_name, model in models.items():
        if "summarization" in model.task:
            results[model_name] = model(input_text.text)
    return results

@lru_cache(maxsize=100)
def get_cached_model_output(task: str, text: str):
    if task in "classify":
        return classify_text(InputText(text=text))
    elif task in "summarize":
        return summarize_text(InputText(text=text))
    elif task in "ner":
        return recognize_entities(InputText(text=text))
    elif task in "answer":
        return answer_question(InputQuestion(context=text, question="What is the question?"))
    else:
        raise HTTPException(status_code=400, detail="Invalid task type")

@app.post("/model_output")
def cached_model_output(task: str, input_text: InputText):
    return get_cached_model_output(task, input_text.text)

@app.post("/benchmark")
def benchmark_models(dataset: list[InputText]):
    results = {}
    for model_name, model in models.items():
        predictions = [model(text.text)[0]['label'] for text in dataset]
        labels = [text.label for text in dataset]  # Assuming dataset contains labeled data
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        results[model_name] = {"accuracy": accuracy, "f1_score": f1}
    return results
