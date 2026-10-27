from argparse import ArgumentParser
from pathlib import Path

import ollama
from sentence_transformers import SentenceTransformer, util
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams , PointStruct

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM = "llama3.2:3b"
DEFAULT_DOCUMENT = Path(__file__).parent / "docs" / "football.txt"
client = QdrantClient(
    url="http://localhost:6333"
)


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
    
def chunk_text_by_paragraph(text: str) -> list[str]:
    """Split text into paragraphs."""
    paragraphs = text.split("\n\n")
    return [p.strip() for p in paragraphs if p.strip()]

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


def build_vector_store(document: Path, embedding_model: SentenceTransformer) :
    """Read a document, chunk it, and create its in-memory vector store."""
    chunks = chunk_text_by_paragraph(document.read_text(encoding="utf-8"))
    embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
    # return [
    #     {"id": index, "text": chunk, "embedding": embeddings[index]}
    #     for index, chunk in enumerate(chunks)
    # ]
    
    # we will replace it with qdrant store
    if client.collection_exists("football_docs"):
        client.delete_collection("football_docs")

    client.create_collection(
        collection_name="football_docs",
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
    )
    
    points = []
    
    for i,(embedding,chunk) in enumerate(zip(embeddings,chunks)):
        print(f"\n--- CHUNK {i} ---")
        print(chunk)
        points.append(
            PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={
                    "text": chunk,
                    "source": 'football.txt'
                }
            )
        )
    
    client.upsert(
        collection_name="football_docs",
        points=points,
    )


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
        

def retrieve_with_qdrant(query: str, embedding_model: SentenceTransformer, top_k: int = 3) -> list[dict]:
    """Return the most relevant chunks for a query."""

    query_embedding = embedding_model.encode(
            query,
            convert_to_tensor=False,
        )

    results = client.query_points(
        collection_name="football_docs",
        query=query_embedding,
        limit=top_k,
    )
        
    retrieved = []

    for result in results.points:
        item = {
            "text": result.payload["text"],
            "source": result.payload["source"],
            "score": result.score,
        }

        retrieved.append(item)

        print(f"Score: {result.score:.4f}")
        print(result.payload["text"])
        print("---")

    return retrieved
    


def rag(
    query: str,
    top_k: int = 3,
    llm_model: str = DEFAULT_LLM,
) -> str:
    """Answer a query using only retrieved document context."""
    results = []
    results = retrieve_with_qdrant(
        query,
        embedding_model=SentenceTransformer(DEFAULT_MODEL),
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
    parser.add_argument("query", nargs="?", default=" What is a quarterback's favorite food?")
    parser.add_argument("--document", type=Path, default=DEFAULT_DOCUMENT)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    embedding_model = SentenceTransformer(args.embedding_model)
    vector_store = build_vector_store(args.document, embedding_model)
    print(rag(args.query, args.top_k, args.llm_model))
    
    # print(rag(args.query, vector_store, embedding_model, args.top_k, args.llm_model))
    
#     queries = [
#     "How many points is a touchdown worth?",
#     "Who throws the ball?",
#     "What position passes the ball?",
#     "What does a wide receiver do?",
#     "Who is the president of Senegal?",
# ]

    # for query in queries:
    #     print("\n" + "=" * 70)
    #     print(f"QUERY: {query}")

    #     results = retrieve(
    #         query,
    #         vector_store,
    #         embedding_model,
    #         top_k=3
    #     )

    #     for result in results:
    #         print(
    #             f"\nScore: {result['score']:.4f}"
    #         )
    #         print(result["text"])


if __name__ == "__main__":
    main()