import os
import json
from groq import Groq
from dotenv import load_dotenv

# Load environmental configurations securely
load_dotenv()

# Initialize the Groq client pulling directly from shell context
client = Groq()

def extract_movie_analytics(raw_review_text):
    """
    Parses unstructured text and outputs a strictly typed JSON schema.
    """
    
    # 1. Embed the strict structural layout constraint directly into the prompt
    prompt = f"""
    Analyze the following unstructured movie review and parse out the critical analytics.
    You must output a single, valid JSON object matching this exact schema format without variance:
    {{
        "movie_title": "string",
        "tonal_verdict": "Positive" or "Negative" or "Neutral",
        "numerical_rating": float,
        "primary_cinematic_strength": "string",
        "one_line_viral_hook": "string"
    }}
    
    Unstructured Review Data:
    "{raw_review_text}"
    """

    # 2. Invoke the API using the premium 70B model configuration
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a specialized automated parsing backend. You only output minified, syntactically correct JSON. Do not include introductory text or markdown wrappers."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        # CRITICAL PARAMETER: Forces the model to use constrained decoding 
        response_format={"type": "json_object"},
        temperature=0.0  # Keep it zero for absolute deterministic predictability
    )
    
    return response.choices[0].message.content

# ---- Testing Execution Zone ----
if __name__ == "__main__":
    raw_critique = "Honestly, Denis Villeneuve's Dune: Part Two is a cinematic masterpiece. The sound design shook the theater, and Chalamet gave a career-best performance, though the 3-hour runtime felt a bit taxing. Easily a solid 4.8 out of 5 for me."

    print("Sending text to Llama-3.3-70b via Groq...")
    raw_json_string = extract_movie_analytics(raw_critique)
    
    print("\n--- Raw String Output from API ---")
    print(raw_json_string)

    print("\n--- Verifying Safe Python Dictionary Parsing ---")
    # This function would instantly throw an error if the model returned conversational text
    parsed_dictionary = json.loads(raw_json_string)
    
    print(f"Parsed Title: {parsed_dictionary['movie_title']}")
    print(f"Parsed Rating: {parsed_dictionary['numerical_rating']}/5")
    print(f"Parsed Hook:  {parsed_dictionary['one_line_viral_hook']}")