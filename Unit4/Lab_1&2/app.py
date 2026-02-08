import os
import math
import datetime
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# Load environment variables
load_dotenv()

def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two vectors
    Cosine similarity = (A · B) / (||A|| * ||B||)
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimensions")
    
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def search_sentences(vector_store, query: str, k: int = 3):
    """
    Search the `vector_store` for `query` and return top `k` results with scores.
    Prints ranked results with score (4 decimal places) and the sentence text.
    """
    results = vector_store.similarity_search_with_score(query, k=k)
    top_k = results[:k]
    output = []
    for rank, item in enumerate(top_k, start=1):
        # item is expected to be a (doc, score) tuple
        try:
            doc, score = item
        except Exception:
            # Fallback if structure is unexpected
            doc = item[0]
            score = item[1] if len(item) > 1 else 0.0

        text = getattr(doc, "page_content", None)
        if text is None:
            # doc may be a plain string or have a `text` attribute
            text = getattr(doc, "text", None) or (doc if isinstance(doc, str) else str(doc))

        print(f"Rank {rank}: Score {score:.4f} - {text}")
        output.append((text, score))

    return output

def main():
    print("🤖 Python LangChain Agent Starting...\n")

    # Check for GitHub token
    if not os.getenv("GITHUB_TOKEN"):
        print("❌ Error: GITHUB_TOKEN not found in environment variables.")
        print("Please create a .env file with your GitHub token:")
        print("GITHUB_TOKEN=your-github-token-here")
        print("\nGet your token from: https://github.com/settings/tokens")
        print("Or use GitHub Models: https://github.com/marketplace/models")
        return
    
    # Create OpenAIEmbeddings instance for GitHub Models API
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        base_url="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN"),
        check_embedding_ctx_length=False,
    )

    print("✅ OpenAIEmbeddings instance created (GitHub Models API).")
    # Create an in-memory vector store using the embeddings instance
    vector_store = InMemoryVectorStore(embeddings)
    print("=== Embedding Inspector Lab ===")
    print("Generating embeddings for multiple sentences...")

    test_sentences = [
        # Animals and pets
        "The canine barked loudly.",
        "A tabby cat purred on the windowsill.",
        # Science and physics
        "The electron spins rapidly.",
        "Gravity causes objects to fall toward Earth.",
        # Food and cooking
        "Fresh basil elevates the flavor of tomato sauce.",
        "She baked a chocolate cake for the party.",
        "The chef chopped onions finely before sautéing.",
        # Sports and activities
        "He scored the winning goal in the final minute.",
        "They practiced yoga at sunrise in the park.",
        # Weather and nature
        "Thunderstorms are expected later this afternoon.",
        "Autumn leaves created a colorful carpet on the path.",
        # Technology and programming
        "The developer fixed a critical bug in the API.",
        "Machine learning models require careful tuning.",
    ]

    # Prepare metadata for each sentence and add all texts to the vector store
    metadatas = []
    for idx, _ in enumerate(test_sentences):
        metadatas.append({
            "created_at": datetime.datetime.now().isoformat(),
            "index": idx,
        })

    vector_store.add_texts(test_sentences, metadatas=metadatas)
    print(f"✅ Stored {len(test_sentences)} sentences in the vector store.")
    for idx, s in enumerate(test_sentences, start=1):
        print(f"Stored Sentence {idx}: {s}")

    # Interactive semantic search loop
    print("=== Semantic Search ===")
    while True:
        query = input("Enter a search query (or 'quit' to exit): ")
        if query.lower() in ("quit", "exit"):
            break
        if not query.strip():
            continue

        # Perform search and display results
        search_sentences(vector_store, query)
        print()

    # Generate embeddings for each test sentence (for similarity checks)
    embedding_vectors = []
    for idx, sentence in enumerate(test_sentences, start=1):
        vec = embeddings.embed_query(sentence)
        embedding_vectors.append(vec)

    print("✅ Generated embeddings for all sentences.")

    # Compute and display cosine similarities between sentence pairs
    comparisons = [
        (0, 1),  # Sentence 1 vs Sentence 2
        (1, 2),  # Sentence 2 vs Sentence 3
        (2, 0),  # Sentence 3 vs Sentence 1
    ]

    for a, b in comparisons:
        sim = cosine_similarity(embedding_vectors[a], embedding_vectors[b])
        print(
            f"Cosine similarity between Sentence {a+1} and Sentence {b+1} "
            f"('{test_sentences[a]}' vs '{test_sentences[b]}'): {sim:.4f}"
        )

if __name__ == "__main__":
    main()