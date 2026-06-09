import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
client = Groq()

def generate_social_hook(movie_name, creative_level):
    """
    Generates a cinematic, high-engagement social media hook for a movie.
    creative_level: Float between 0.0 (factual/rigid) and 1.0 (creative/wild)
    """
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are an elite Hollywood scriptwriter and social media growth hacker. Your job is to write a single, viral, jaw-dropping hook for a movie provided by the user. Do not include intro text, just give the hook."
            },
            {
                "role": "user",
                "content": f"Write a hook for the movie: {movie_name}"
            }
        ],
        # Handling the hyperparameters 
        temperature=creative_level, 
        max_tokens=100,
        top_p=1.0
    )
    
    return response.choices[0].message.content

# ---- Experimentation Zone ----
movie = "Inception"

print("--- FACTUAL / DETERMINISTIC (Temperature = 0.0) ---")
print(generate_social_hook(movie, creative_level=0.0))

print("\n--- CREATIVE / DYNAMIC (Temperature = 0.9) ---")
print(generate_social_hook(movie, creative_level=0.9))
