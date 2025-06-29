# Retrieval-Augmented Generation (RAG) with Qdrant and OpenAI

This project demonstrates how to perform Retrieval-Augmented Generation (RAG) using Qdrant for vector storage and OpenAI for language generation. The setup involves loading documents, generating embeddings, storing them in Qdrant, and querying them to generate responses using OpenAI.

## Table of Contents

- [Installation](#installation)
- [Setup](#setup)
- [Ingesting Documents](#ingesting-documents)
- [Running the RAG Script](#running-the-rag-script)
- [Configuration](#configuration)
- [License](#license)

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/rag-with-qdrant.git
    cd rag-with-qdrant
    ```

2. **Create a virtual environment** (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. **Install the dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Setup

1. **Ensure you have a running instance of Qdrant**. You can use Docker to run Qdrant:

    ```bash
    docker compose up -d
    ```

2. **Copy `.env.example` to `.env`** and fill in your credentials:

    ```bash
    cp .env.example .env
    ```

    Edit `.env` to include your OpenAI and Qdrant keys.

## Ingesting Documents

Use the `ingest.py` script to load and process documents, generate embeddings, and store them in Qdrant.

```bash
python ingest.py
```

## Running the RAG Script
Use the `rag.py` script to query the vector database and generate responses using OpenAI.
```bash
python rag.py
```

## Configuration
- Qdrant: Ensure Qdrant is running on http://localhost:6333 or set `QDRANT_URL` in your `.env` file.
- Environment Variables: Use a `.env` file based on `.env.example` to store your API keys.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
