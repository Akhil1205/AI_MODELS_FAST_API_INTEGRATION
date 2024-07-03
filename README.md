# NLP Model Comparison API

Welcome to the NLP Model Comparison API! This API allows you to compare various NLP models across different tasks such as text classification, named entity recognition (NER), question answering (QA), and text summarization. 

## Features

- **Text Classification**: Classify text using various pretrained models.
- **Named Entity Recognition (NER)**: Recognize entities in text using different models.
- **Question Answering (QA)**: Answer questions based on given contexts using QA models.
- **Text Summarization**: Summarize text using summarization models.
- **Benchmarking**: Benchmark the performance of different models on custom datasets.
- **Rate Limiting**: Prevent abuse by rate limiting API usage.

## Installation

To install and run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/nlp-model-comparison-api.git
    cd nlp-model-comparison-api
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Start the FastAPI server:
    ```bash
    uvicorn main:app --reload
    ```

## API Endpoints

### Health Check

- **Endpoint**: `/health`
- **Method**: `GET`
- **Description**: Check the health status of the API.
- **Response**:
    ```json
    {
        "status": "OK"
    }
    ```

### List Models

- **Endpoint**: `/models`
- **Method**: `GET`
- **Description**: Get a list of available models.
- **Response**:
    ```json
    {
        "models": ["bert-base-uncased", "roberta-base", ...]
    }
    ```

### Text Classification

- **Endpoint**: `/classify`
- **Method**: `POST`
- **Description**: Classify text using available text classification models.
- **Request Body**:
    ```json
    {
        "text": "Your text to classify"
    }
    ```
- **Response**: Dictionary of classification results from different models.

### Named Entity Recognition (NER)

- **Endpoint**: `/ner`
- **Method**: `POST`
- **Description**: Recognize entities in text using available NER models.
- **Request Body**:
    ```json
    {
        "text": "Your text to recognize entities"
    }
    ```
- **Response**: Dictionary of recognized entities from different models.

### Question Answering (QA)

- **Endpoint**: `/answer`
- **Method**: `POST`
- **Description**: Answer questions based on given context using available QA models.
- **Request Body**:
    ```json
    {
        "context": "Context text",
        "question": "Your question"
    }
    ```
- **Response**: Dictionary of answers from different models.

### Text Summarization

- **Endpoint**: `/summarize`
- **Method**: `POST`
- **Description**: Summarize text using available summarization models.
- **Request Body**:
    ```json
    {
        "text": "Your text to summarize"
    }
    ```
- **Response**: Dictionary of summaries from different models.

### Benchmark Models

- **Endpoint**: `/benchmark`
- **Method**: `POST`
- **Description**: Benchmark the performance of models on custom datasets.
- **Request Body**: List of texts and labels for classification or NER tasks.
- **Response**: Dictionary of benchmark results (accuracy and F1 score) from different models.

### Cached Model Output

- **Endpoint**: `/cached_output`
- **Method**: `POST`
- **Description**: Get cached model output to improve performance and limit rate.
- **Request Body**:
    ```json
    {
        "task": "classification/ner/qa/summarization",
        "text": "Your text"
    }
    ```
- **Response**: Cached results of the specified task.

## Authentication

Some endpoints require basic authentication. Use the following credentials:
- **Username**: `user`
- **Password**: `pass`

## Rate Limiting

To prevent abuse, some endpoints are rate-limited. The default limit is 10 requests per 60 seconds.

## Logging

Requests and responses are logged for monitoring purposes.

## Error Handling

- **401 Unauthorized**: Invalid username or password.
- **429 Too Many Requests**: Rate limit exceeded.

## Acknowledgments

This project uses various pretrained models from the Hugging Face Transformers library.

