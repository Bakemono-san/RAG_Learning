# Local RAG Demo

A small retrieval-augmented generation example that indexes a local document with
`sentence-transformers`, retrieves the most relevant chunks, and asks a local
Ollama model to answer from that context.

## Requirements

- Python 3.10+
- Ollama with the configured model available locally
- A working build environment for `sentence-transformers` and PyTorch

## Setup

```bash
cd demo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull llama3.2:3b
```

The embedding model is downloaded automatically the first time the application
runs.

## Usage

Run the default football question:

```bash
python main.py
```

Ask a different question:

```bash
python main.py "How many points is a field goal worth?"
```

The document, embedding model, Ollama model, and number of retrieved chunks can
be changed with `--document`, `--embedding-model`, `--llm-model`, and `--top-k`.

## Project structure

```text
main.py                 RAG pipeline and command-line entry point
docs/football.txt       Source document used by the demo
requirements.txt        Runtime dependencies
```