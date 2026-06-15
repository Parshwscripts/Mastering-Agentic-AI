import os
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv

# 1. Bootstrap environment variables
load_dotenv()

# Verify API configuration before launching pipeline
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("CRITICAL: GROQ_API_KEY is missing from your root .env file!")

# 2. Initialize ChromaDB Persistent Engine (Using the storage from Day 7)
chroma_client = chromadb.PersistentClient(path="./chroma_db_storage")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Fetch our existing collection from yesterday
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=sentence_transformer_ef,
    metadata={"hnsw:space": "cosine"}
)

# 3. Define the RAG Query Execution Function
def execute_rag_pipeline(user_query: str):
    print(f"\n[STEP 1: RETRIEVAL] Querying ChromaDB for: '{user_query}'")
    
    # Query ChromaDB for the absolute best single context match (n_results=1)
    retrieval_results = collection.query(
        query_texts=[user_query],
        n_results=1
    )
    
    # Extract the text and distance score safely
    if retrieval_results['documents'][0]:
        fetched_context = retrieval_results['documents'][0][0]
        distance_score = retrieval_results['distances'][0][0]
        print(f"-> Found Match (Distance: {distance_score:.4f}): '{fetched_context}'")
    else:
        print("-> No relevant context found in database. Aborting inference safety.")
        return None

    print("\n[STEP 2: AUGMENTATION] Building grounded system prompt payload...")
    
    # Construct an enterprise-grade prompt framework that forbids hallucinations
    system_prompt = f"""
    You are a precise, fact-based engineering assistant. 
    Your core instruction is to answer the user's question using ONLY the provided verified context below.
    
    CRITICAL GUARDRAIL: If the answer cannot be fully derived from the verified context, explicitly state 
    "Information not available in the database." Do not use your pre-trained background knowledge to guess.
    
    [VERIFIED CONTEXT BASELINE]
    {fetched_context}
    """

    print("[STEP 3: GENERATION] Dispatching payload to Groq LPU engine...")
    
    # Initialize our fast Groq inference engine
    groq_client = Groq()
    
    # Fire the completion tracking call
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.0 # Force absolute deterministic accuracy
    )
    
    return response.choices[0].message.content

# --- Test Executions ---

# Test Case A: A question whose answer lives directly inside our database
query_a = "When should I apply my basal fertilizer dosage for sugarcane?"
answer_a = execute_rag_pipeline(query_a)
print("\n [AI GROUNDED RESPONSE]:")
print(answer_a)

print("\n" + "="*60 + "\n")

# Test Case B: A question that the LLM definitely knows from training, but is NOT in our database
query_b = "Who directed the movie Inception?"
answer_b = execute_rag_pipeline(query_b)
print("\n [AI GROUNDED RESPONSE]:")
print(answer_b)