import os
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# 1. Component Initialization
chroma_client = chromadb.PersistentClient(path="./chroma_db_storage")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Fetch or isolate our collection table
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=sentence_transformer_ef,
    metadata={"hnsw:space": "cosine"}
)

# 2. Automated Parsing & Chunking Functions
def chunk_document_text(text: str, chunk_size: int = 25, chunk_overlap: int = 5):
    """Splits text into overlapping string blocks."""
    words = text.split()
    chunks = []
    start_index = 0
    while start_index < len(words):
        end_index = start_index + chunk_size
        chunk_words = words[start_index:end_index]
        chunks.append(" ".join(chunk_words))
        start_index += (chunk_size - chunk_overlap)
    return chunks

def ingest_file_to_vector_db(file_path: str):
    """Reads a local file, processes it, and upserts it into ChromaDB."""
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return
        
    print(f" Reading raw file: {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
        
    # Break down the raw document text using our chunking processor
    text_chunks = chunk_document_text(raw_text, chunk_size=30, chunk_overlap=8)
    print(f" Split document into {len(text_chunks)} overlapping chunks.")
    
    # Generate structural IDs and metadata sets dynamically
    ids = [f"chunk_{os.path.basename(file_path)}_{i}" for i in range(len(text_chunks))]
    metadatas = [{"source_file": os.path.basename(file_path), "chunk_index": i} for i in range(len(text_chunks))]
    
    # Write to local vector database storage
    collection.upsert(ids=ids, documents=text_chunks, metadatas=metadatas)
    print("Vector indexing and metadata attachment successful!")

# 3. Execution Query Framework
def query_ingested_knowledge(user_query: str):
    print(f"\n Processing Search Query: '{user_query}'")
    
    # Query ChromaDB for the closest matching chunk context
    results = collection.query(query_texts=[user_query], n_results=1)
    
    if not results['documents'][0]:
        print(" No matching context located.")
        return
        
    context = results['documents'][0][0]
    source = results['metadatas'][0][0]['source_file']
    print(f" Best Context Found in [{source}]: \"{context}\"")
    
    # Fire structured prompt payload to Groq for factual answers
    client = Groq()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": f"Answer concisely using ONLY this context:\n{context}"},
            {"role": "user", "content": user_query}
        ],
        temperature=0.0
    )
    print(f" [AI ANSWER]: {response.choices[0].message.content}")

# --- Pipeline Run Verification ---
if __name__ == "__main__":
    # Step 1: Run the automated data ingestion factory
    target_file = "knowledge_source.txt"
    ingest_file_to_vector_db(target_file)
    
    print("\n" + "="*50)
    
    # Step 2: Test the retrieval against our newly ingested files
    query_ingested_knowledge("What is the purpose of the Bharni phase?")
    query_ingested_knowledge("How do you design a high retention video hook?")