from argparse import ArgumentParser
from pathlib import Path

import ollama
from sentence_transformers import SentenceTransformer, util


DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM = "llama3.2:3b"
DEFAULT_DOCUMENT = Path(__file__).parent / "docs" / "football.txt"


def chunk_text(text: str, chunk_size: int = 30, overlap: int = 5) -> list[str]:
    """Split text into overlapping word-based chunks."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller")

    words = text.split()
    step = chunk_size - overlap
    return [
        " ".join(words[start : start + chunk_size])
        for start in range(0, len(words), step)
    ]

def call_llm(prompt: str, model_name: str = DEFAULT_LLM) -> str:
    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response["message"]["content"]


def build_vector_store(document: Path, embedding_model: SentenceTransformer) -> list[dict]:
    """Read a document, chunk it, and create its in-memory vector store."""
    chunks = chunk_text(document.read_text(encoding="utf-8"))
    embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
    return [
        {"id": index, "text": chunk, "embedding": embeddings[index]}
        for index, chunk in enumerate(chunks)
    ]


def retrieve(query: str, vector_store: list[dict], embedding_model: SentenceTransformer, top_k: int = 3) -> list[dict]:
    """Return the most relevant chunks for a query."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    results = [
        {
            "text": item["text"],
            "score": util.cos_sim(query_embedding, item["embedding"]).item(),
        }
        for item in vector_store
    ]
    return sorted(results, key=lambda result: result["score"], reverse=True)[:top_k]


def rag(
    query: str,
    vector_store: list[dict],
    embedding_model: SentenceTransformer,
    top_k: int = 3,
    llm_model: str = DEFAULT_LLM,
) -> str:
    """Answer a query using only retrieved document context."""
    results = retrieve(
        query,
        vector_store,
        embedding_model,
        top_k=top_k
    )
    context = "\n\n".join(
        result["text"]
        for result in results
    )
    prompt = f"""
Answer the question using only the context below.

If the context does not contain enough information
to answer the question, say that you don't have
enough information.

Context:
{context}

Question:
{query}

Answer:
"""
    return call_llm(prompt, model_name=llm_model)


def main() -> None:
    parser = ArgumentParser(description="Answer questions from a local document with RAG.")
    parser.add_argument("query", nargs="?", default="How many points is a touchdown worth?")
    parser.add_argument("--document", type=Path, default=DEFAULT_DOCUMENT)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    embedding_model = SentenceTransformer(args.embedding_model)
    vector_store = build_vector_store(args.document, embedding_model)
    print(rag(args.query, vector_store, embedding_model, args.top_k, args.llm_model))


if __name__ == "__main__":
    main()