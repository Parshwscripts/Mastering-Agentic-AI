import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Initialize an industry-standard, lightweight embedding model
# 'all-MiniLM-L6-v2' maps text into a 384-dimensional vector space
print("Loading local embedding model pipeline...")
embedding_engine = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_cosine_similarity(vector_a, vector_b):
    """
    Computes the cosine similarity (angle) between two geometric vectors.
    Returns a score between -1.0 (opposite) and 1.0 (identical meaning).
    """
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)

# 2. Define a database of sample unstructured phrases
corpus = [
    "Denis Villeneuve directed a stunning sci-fi cinematic masterpiece.",
    "The agricultural yield for sugarcane in Maharashtra increased this year.",
    "A brilliant performance by the lead actor kept the audience hooked.",
    "Applying the right NPK fertilizer dosage improves crop health drastically."
]

print("\nGenerating high-dimensional vector coordinates for our data corpus...")
# This converts our text list into a mathematical matrix of coordinates
corpus_embeddings = embedding_engine.encode(corpus)

# Let's inspect the structural shape of what the machine sees
print(f"Total phrases embedded: {len(corpus_embeddings)}")
print(f"Dimensions per text vector: {len(corpus_embeddings[0])}") 
# Each sentence is represented precisely by 384 floating-point coordinates!

# 3. Simulate an incoming user search query
user_search = "I want to watch an incredible film tonight"
print(f"\nUser Search Query: '{user_search}'")

# Convert the user query into the exact same vector space coordinate map
query_embedding = embedding_engine.encode(user_search)

print("\n--- Calculating Semantic Search Matches ---")
# Loop through our database vectors and compare them to the user query vector
for idx, text_sentence in enumerate(corpus):
    similarity_score = calculate_cosine_similarity(query_embedding, corpus_embeddings[idx])
    print(f"Similarity Score: {similarity_score:.4f} | Text: '{text_sentence}'")