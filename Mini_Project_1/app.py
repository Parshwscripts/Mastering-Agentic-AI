import streamlit as st
import json
import os
from groq import Groq
from dotenv import load_dotenv

# Load environmental variables from your local .env file
load_dotenv()

# 1. Page Configuration and Layout
st.set_page_config(page_title="AI Cinematic Content Studio", page_icon="🎬", layout="wide")

st.title("🎬 AI Cinematic Content Hook Studio")
st.subheader("Transform raw cinematic descriptions into structured, viral social media marketing assets.")
st.write("---")

# 2. Sidebar Configuration for Parameters
st.sidebar.header("🛠️ Model Configuration")

# Verify if API key exists in environment
api_key_exists = os.getenv("GROQ_API_KEY") is not None

if api_key_exists:
    st.sidebar.success("🔒 API Key loaded securely from .env")
else:
    st.sidebar.error("❌ No API Key found in .env file!")

# Interactive parameter slider mapped directly to the LLM
creativity_slider = st.sidebar.slider(
    "Adjust Temperature (Creativity):", 
    min_value=0.0, max_value=1.0, value=0.7, step=0.1
)

st.sidebar.markdown("""
**Parameter Guide:**
- **Temp 0.0**: Strict, rigid structural analytics.
- **Temp 0.9**: High-variance, creative marketing copy.
""")

# 3. Core Input Dashboard Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📝 Input Movie Context")
    movie_name = st.text_input("Movie Title:", placeholder="e.g., Inception, RRR, Dune: Part Two")
    raw_context = st.text_area(
        "Enter raw themes, key scenes, or character vibes:", 
        placeholder="e.g., Intense sci-fi action, sandstorms, power dynamics, visually stunning cinematography...",
        height=150
    )
    submit_btn = st.button("🚀 Generate Optimized Assets")

# 4. Backend Processing and Schema Enforcement Phase
if submit_btn:
    if not movie_name or not raw_context:
        st.error("⚠️ Missing Data: Please enter both a movie title and context text.")
    elif not api_key_exists:
        st.error("⚠️ Environment Error: Ensure GROQ_API_KEY is properly set inside your .env file.")
    else:
        with col2:
            st.markdown("### 📊 Live AI Engine Output")
            with st.spinner("Executing structural inference via Llama-3.3-70b..."):
                try:
                    # Initialize client using environment credentials
                    client = Groq()
                    
                    # Formulate clear instructions with an absolute JSON structural layout constraint
                    prompt = f"""
                    Analyze the movie '{movie_name}' using this raw context data: '{raw_context}'.
                    Perform logical analytics and creative copywriting to return a single, valid JSON object.
                    
                    You must precisely follow this schema format:
                    {{
                        "core_genre_profile": "string",
                        "target_audience_demographic": "string",
                        "viral_marketing_hooks": ["hook1", "hook2", "hook3"],
                        "seo_meta_tags": ["tag1", "tag2", "tag3"]
                    }}
                    """

                    # Fire the payload to Groq
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": "You are an automated entertainment marketing backend. Output ONLY minified, valid JSON. No conversational intro text."},
                            {"role": "user", "content": prompt}
                        ],
                        response_format={"type": "json_object"},
                        temperature=creativity_slider
                    )
                    
                    # Parse the raw API string into a native Python dictionary
                    output_data = json.loads(response.choices[0].message.content)
                    
                    st.success("⚡ Analysis Generation Successful!")
                    
                    # Map JSON fields cleanly to professional UI containers
                    st.markdown("#### **📈 Metadata Profile**")
                    st.info(f"**Genre Identity:** {output_data.get('core_genre_profile')}")
                    st.warning(f"**Target Audience Profile:** {output_data.get('target_audience_demographic')}")
                    
                    st.markdown("#### **🔥 AI-Generated Marketing Hooks**")
                    for idx, hook in enumerate(output_data.get('viral_marketing_hooks', []), start=1):
                        st.code(f"Hook Option #{idx}: {hook}", language="text")
                        
                    st.markdown("#### **🏷️ Suggested Meta Tags**")
                    st.write(" | ".join([f"`#{tag}`" for tag in output_data.get('seo_meta_tags', [])]))
                    
                    # Render the pure structural JSON block inside an expander drawer
                    with st.expander("👁️ View Raw Parsed JSON Object"):
                        st.json(output_data)
                        
                except Exception as e:
                    st.error(f"Inference Failure: {str(e)}")