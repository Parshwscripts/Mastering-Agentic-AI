import chromadb
from chromadb.utils import embedding_functions

print("Initializing local persistent storage engine...")
# 1. Instantiate a persistent on-disk database client (creates a local folder)
chroma_client = chromadb.PersistentClient(path="./chroma_db_storage")

# ChromaDB handles text-to-vector transformations automatically using this wrapper
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

print("Creating or fetching 'knowledge_base' collection...")
# Collections in Vector DBs are equivalent to 'Tables' in Relational DBs
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=sentence_transformer_ef,
    metadata={"hnsw:space": "cosine"} # Explicitly lock metric to Cosine Similarity
)

# 3. Define structured documents, unique primary IDs, and metadata tags
documents_pool = [
    "Dune Part Two features mind-blowing visuals and deep desert sci-fi political lore.",
    "Applying an explicit split dosage of NPK fertilizer maximizes sugarcane tonnage during the Bharni phase.",
    "Thalapathy Vijay's screen presence and high-octane action choreography drove massive box office numbers.",
    "Basal fertilizer application should be done immediately during the planting phase of sugarcane roots."
]

document_ids = ["doc_01", "doc_02", "doc_03", "doc_04"]

document_metadata = [
    {"category": "cinema", "sub_genre": "sci-fi"},
    {"category": "agriculture", "crop_type": "sugarcane"},
    {"category": "cinema", "sub_genre": "action"},
    {"category": "agriculture", "crop_type": "sugarcane"}
]

print("📥 Injecting documents into the vector space index...")
# The upsert method adds new documents or updates existing IDs seamlessly
collection.upsert(
    ids=document_ids,
    documents=documents_pool,
    metadatas=document_metadata
)
print("Vector indexing pipeline complete.")

# 4. Execute a production semantic search query
user_query = "What steps should I take to improve my crop yield?"
print(f"\nIncoming Search Query: '{user_query}'")

# Query the database for the top 2 closest contextual matches
search_results = collection.query(
    query_texts=[user_query],
    n_results=2
)

print("\n--- Top Semantic Query Matches ---")
for i in range(len(search_results['ids'][0])):
    print(f"\nRank #{i+1}:")
    print(f"ID: {search_results['ids'][0][i]}")
    print(f"Document Text: {search_results['documents'][0][i]}")
    print(f"Metadata Tag: {search_results['metadatas'][0][i]}")
    print(f"Distance Score: {search_results['distances'][0][i]:.4f}")

# 5. Advanced Feature: Executing Metadata Filtering (Pre-filtering the search space)
print(f"\n🔧 Executing strict filtered search: Looking ONLY for 'cinema' category...")
filtered_results = collection.query(
    query_texts=["action movie"],
    n_results=1,
    where={"category": "cinema"} # Strict metadata filter constraint
)
print(f"Filtered Match: '{filtered_results['documents'][0][0]}'")
