import pprint

# A mock dense text block representing an engineering document layout
dense_document_text = (
    "Phase 1 of the deployment requires initializing the root environment parameters. "
    "Engineers must ensure that the secret security keys are never exposed to public repositories. "
    "Moving into Phase 2, the team will transition focus toward data architecture structures. "
    "This includes configuring local high-performance vector index storage systems like ChromaDB. "
    "Phase 3 focuses heavily on testing pipeline accuracy boundaries under peak inference stress. "
    "Hyperparameters such as temperature must be locked to zero to maintain deterministic predictability. "
    "Finally, Phase 4 dictates pushing the optimized web interface dashboard to cloud hosting resources."
)

def chunk_text_by_words(text: str, chunk_size: int, chunk_overlap: int):
    """
    Splits a continuous string into distinct blocks based on word count
    while maintaining a sliding window token overlap.
    """
    words = text.split()
    chunks = []
    
    # Validation guardrail to prevent infinite loops
    if chunk_overlap >= chunk_size:
        raise ValueError("CRITICAL: Overlap size must be strictly smaller than the chunk size!")
        
    start_index = 0
    while start_index < len(words):
        # Determine the end boundary of the current chunk window
        end_index = start_index + chunk_size
        
        # Pull the words within our current window slice
        chunk_words = words[start_index:end_index]
        
        # Join the words back into a readable string block
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        
        # Slide the window forward by subtracting the overlap from the chunk size
        start_index += (chunk_size - chunk_overlap)
        
    return chunks

# --- Execution Tests ---

# Configuration Parameters
SIZE = 15     # Each chunk will contain exactly 15 words
OVERLAP = 5   # The subsequent chunk will pull the last 5 words of the previous chunk

print(f" Original Text Length: {len(dense_document_text.split())} words.\n")
print(f"Running Sliding Window Chunker (Size: {SIZE}, Overlap: {OVERLAP})...\n")

processed_chunks = chunk_text_by_words(dense_document_text, chunk_size=SIZE, chunk_overlap=OVERLAP)

# Beautifully print the list of strings to see the text boundaries clearly
pp = pprint.PrettyPrinter(indent=2)
print("---  Generated Context Chunks ---")
for index, chunk in enumerate(processed_chunks):
    print(f"\n[CHUNK #{index + 1}]:")
    print(f"Content: \"{chunk}\"")
    print(f"Word Count: {len(chunk.split())}")