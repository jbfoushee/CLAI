import os
import math
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.tools import tool

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


def load_document(vector_store, file_path: str):
    """
    Load a document from `file_path`, create a LangChain `Document`, add it
    to `vector_store`, and return the created document ID (if available).
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None
    except Exception as e:
        err_str = str(e).lower()
        if "maximum context length" in err_str or "token" in err_str:
            print("⚠️ This document is too large to embed as a single chunk.")
            print("Token limit exceeded. The embedding model can only process up to 8,191 tokens at once.")
            print("Solution: The document needs to be split into smaller chunks.")
        else:
            print(f"❌ Error reading file {file_path}: {e}")
        return None

    metadata = {
        "fileName": os.path.basename(file_path),
        "createdAt": datetime.now().isoformat(),
    }

    document = Document(page_content=text, metadata=metadata)

    try:
        result = vector_store.add_documents([document])
        doc_id = None
        if isinstance(result, list) and result:
            # Many vector stores return a list of inserted IDs
            doc_id = result[0]

        print(f"✅ Added document '{metadata['fileName']}' ({len(text)} chars) to vector store.")
        return doc_id
    except Exception as e:
        print(f"❌ Error adding document to vector store: {e}")
        return None


def load_document_with_chunks(vector_store, file_path: str, chunks: list):
    """
    Add pre-split LangChain `Document` chunks to the `vector_store`.

    Each chunk's metadata will be updated with fileName (including chunk X/Total),
    createdAt timestamp, and chunkIndex. Prints progress and returns total stored.
    """
    total = len(chunks)
    file_base = os.path.basename(file_path)
    stored = 0

    for idx, chunk in enumerate(chunks, start=1):
        try:
            meta = getattr(chunk, "metadata", None)
            if meta is None:
                meta = {}
                chunk.metadata = meta

            meta.update({
                "fileName": f"{file_base} (Chunk {idx}/{total})",
                "createdAt": datetime.now().isoformat(),
                "chunkIndex": idx,
            })

            vector_store.add_documents([chunk])
            stored += 1
            print(f"Stored chunk {idx}/{total} for '{file_base}'.")
        except FileNotFoundError:
            print(f"❌ File not found when storing chunk {idx}/{total}: {file_path}")
        except Exception as e:
            err_str = str(e).lower()
            if "maximum context length" in err_str or "token" in err_str:
                print("⚠️ Chunk too large to embed. Split it into smaller parts.")
            else:
                print(f"❌ Error storing chunk {idx}/{total}: {e}")

    return stored


def load_with_fixed_size_chunking(vector_store, file_path: str):
    """
    Read `file_path`, split into fixed-size chunks using `CharacterTextSplitter`,
    create Documents, and add them to `vector_store` via
    `load_document_with_chunks`. Prints chunk statistics and returns number stored.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return 0
    except Exception as e:
        print(f"❌ Error reading file {file_path}: {e}")
        return 0

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator=" ")
    # create_documents expects a list of texts
    docs = splitter.create_documents([text])

    num_chunks = len(docs)
    avg_size = sum(len(d.page_content) for d in docs) / num_chunks if num_chunks else 0

    print(f"Created {num_chunks} chunks (avg size: {avg_size:.1f} chars) for '{os.path.basename(file_path)}'.")

    stored = load_document_with_chunks(vector_store, file_path, docs)
    return stored


def load_with_paragraph_chunking(vector_store, file_path: str):
    """
    Read `file_path`, split into paragraph-preserving chunks using
    `RecursiveCharacterTextSplitter`, create Documents, compare to fixed-size
    chunking, add paragraph chunks to the vector store, and return number stored.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return 0
    except Exception as e:
        print(f"❌ Error reading file {file_path}: {e}")
        return 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0, separators=["\n\n", "\n", " ", ""])
    docs = splitter.create_documents([text])

    num_chunks = len(docs)
    sizes = [len(d.page_content) for d in docs]
    smallest = min(sizes) if sizes else 0
    largest = max(sizes) if sizes else 0
    starts_with_newline = sum(1 for d in docs if d.page_content.startswith("\n"))

    # Compare to fixed-size chunking (without storing)
    fixed_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator=" ")
    fixed_docs = fixed_splitter.create_documents([text])
    fixed_sizes = [len(d.page_content) for d in fixed_docs]

    print("=== Paragraph Chunking vs Fixed-Size Chunking ===")
    print(f"Paragraph chunking: total={num_chunks}, smallest={smallest}, largest={largest}, starts_with_newline={starts_with_newline}")
    print(f"Fixed-size chunking: total={len(fixed_docs)}, smallest={min(fixed_sizes) if fixed_sizes else 0}, largest={max(fixed_sizes) if fixed_sizes else 0}")

    # Store the paragraph chunks
    stored = load_document_with_chunks(vector_store, file_path, docs)
    return stored


def load_with_markdown_header_chunking(vector_store, file_path: str):
    """
    Split the markdown file by headers using `MarkdownHeaderTextSplitter`, then
    apply `RecursiveCharacterTextSplitter` with overlap to each section to
    produce chunks. Store chunks using `load_document_with_chunks` and return
    the number stored.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return 0
    except Exception as e:
        print(f"❌ Error reading file {file_path}: {e}")
        return 0

    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")])
    sections = header_splitter.split_text(text)

    chunker = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs = []
    for sec in sections:
        sec_docs = chunker.create_documents([sec])
        docs.extend(sec_docs)

    sizes = [len(d.page_content) for d in docs]
    num_chunks = len(docs)
    smallest = min(sizes) if sizes else 0
    largest = max(sizes) if sizes else 0

    print("=== Markdown-Header Chunking ===")
    print(f"Total chunks created: {num_chunks}")
    print(f"Smallest chunk: {smallest} chars, Largest chunk: {largest} chars")

    stored = load_document_with_chunks(vector_store, file_path, docs)
    return stored


def create_search_tool(vector_store):
    """Create a LangChain tool that searches the vector store.

    Returns a tool-wrapped function that accepts a query string and returns
    top-3 search results formatted with their scores.
    """
    @tool
    def search_documents(query: str) -> str:
        """Searches the company document repository for relevant information based on the given query. Use this to find information about company policies, benefits, and procedures."""
        try:
            results = vector_store.similarity_search_with_score(query, k=3)
        except Exception as e:
            return f"Error during search: {e}"

        lines = []
        for idx, item in enumerate(results, start=1):
            try:
                doc, score = item
            except Exception:
                doc = item[0]
                score = item[1] if len(item) > 1 else 0.0

            content = getattr(doc, "page_content", None)
            if content is None:
                content = getattr(doc, "text", None) or (doc if isinstance(doc, str) else str(doc))

            lines.append(f"Result {idx} (Score: {score:.4f}): {content}")

        return "\n\n".join(lines)

    return search_documents


def create_agent_executor(vector_store, chat_model=None):
    """Create a ReAct agent executor that can use the search tool to answer queries.

    If `chat_model` is None, a default `ChatOpenAI` instance is created.
    """
    if chat_model is None:
        chat_model = ChatOpenAI(temperature=0)

    search_tool = create_search_tool(vector_store)

    system_msg = "You are a helpful assistant that answers questions about company policies, benefits, and procedures. Use the search_documents tool to find relevant information before answering. Always cite which document chunks you used in your answer."

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_msg),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])


    class SimpleAgentExecutor:
        def __init__(self, llm, tool):
            self.llm = llm
            self.tool = tool

        def invoke(self, inputs: dict):
            query = inputs.get("input", "")
            chat_history = inputs.get("chat_history", [])

            # Run the search tool to get supporting chunks
            try:
                search_results = self.tool(query)
            except Exception as e:
                search_results = f"(search failed: {e})"

            # Build a prompt for the LLM that includes the search results
            prompt = (
                "You are a helpful assistant that answers questions about company policies, "
                "benefits, and procedures. Use the search results below to answer the user's "
                "question and cite which document chunks you used.\n\n"
                f"Search results:\n{search_results}\n\n"
                f"User question: {query}\n\nAnswer:" 
            )

            # Try to call the chat model; if the API shape differs, fall back to returning search results
            try:
                llm_response = self.llm(prompt)
                # llm_response may be a string or an object with content
                response_text = llm_response if isinstance(llm_response, str) else getattr(llm_response, "content", str(llm_response))
            except Exception:
                response_text = f"Search results:\n{search_results}"

            return {"output": response_text}

    return SimpleAgentExecutor(chat_model, search_tool)

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
    print("Removed sample sentences, add_texts usage, and the Lab 2 search loop per request.")
    print("Vector store is initialized and ready for adding documents programmatically.")

    # Load a brochure document into the vector DB
    print("=== Loading Documents into Vector Database ===")
    brochure_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "HealthInsuranceBrochure.md"))
    doc_id = load_document(vector_store, brochure_path)
    if doc_id is not None:
        print(f"Loaded '{os.path.basename(brochure_path)}' successfully into vector store (id: {doc_id}).")
    else:
        print(f"Failed to load '{os.path.basename(brochure_path)}'.")
    
    # Also load the employee handbook
    emp_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "EmployeeHandbook.md"))
    # Load the employee handbook using markdown-header-aware chunking
    chunks_stored = load_with_markdown_header_chunking(vector_store, emp_path)
    if chunks_stored and chunks_stored > 0:
        print(f"Loaded '{os.path.basename(emp_path)}' successfully into vector store ({chunks_stored} chunks stored).")
    else:
        print(f"Failed to load '{os.path.basename(emp_path)}' into vector store.")

    # Create agent executor for conversational QA
    agent_executor = create_agent_executor(vector_store)

    # Chat loop
    chat_history = []
    print("=== Interactive Agent Chat ===")
    print("You can ask the agent about company policies, benefits, and procedures. Type 'quit' or 'exit' to stop.")
    while True:
        try:
            user_input = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break

        if user_input.lower().strip() in ("quit", "exit"):
            print("Goodbye.")
            break

        if not user_input.strip():
            continue

        # Invoke the agent executor with the user's input and chat history
        try:
            result = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history,
            })
        except Exception as e:
            print(f"Agent error: {e}")
            continue

        # Extract and display the agent response
        response = result.get("output") if isinstance(result, dict) else result
        print(f"Agent: {response}")

        # Update chat history
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

if __name__ == "__main__":
    main()