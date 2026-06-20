import os
from typing import Annotated, Dict
from typing_extensions import TypedDict
from groq import Groq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# 1. Bootstrap and Verify Environment API configuration
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("CRITICAL: GROQ_API_KEY is missing from your .env file!")

# Initialize our ultra-fast Groq Client
groq_client = Groq()

# 2. Define the State Structure with a Message List Reducer
class MultiAgentState(TypedDict):
    """
    The shared memory of our network. Using 'add_messages' ensures
    that updates returned by nodes are appended to the list, rather than overwriting it.
    """
    messages: Annotated[list, add_messages]

# 3. Define Agent Nodes using Groq Models

def creative_writer_agent(state: MultiAgentState) -> Dict:
    """Node 1: Specializes in aggressive, cinematic creative hooks."""
    print("🎬 [Node: Creative Writer] Engineering cinematic hooks...")
    
    formatted_messages = []
    for msg in state["messages"]:
        # Extract the role string safely
        role = msg.type if hasattr(msg, 'type') else msg.get('role', 'user')
        
        # Exact API Alignment Mapping
        if role == "human": role = "user"
        if role == "ai": role = "assistant"  # Fixes the downstream crash!
        
        formatted_messages.append({
            "role": role,
            "content": msg.content if hasattr(msg, 'content') else msg.get('content', '')
        })
    
    system_instruction = (
        "You are an expert South Indian cinema reviewer and Instagram creator. "
        "Your job is to take a raw movie concept from the user and generate a high-energy, "
        "cinematic, 1-sentence opening hook script that stops users from scrolling. "
        "Make it punchy, intense, and dramatic! Output ONLY the hook sentence."
    )
    
    payload = [{"role": "system", "content": system_instruction}] + formatted_messages
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=payload,
        temperature=0.85
    )
    
    ai_message = response.choices[0].message.content
    print(f"-> Writer Output: \"{ai_message}\"")
    
    return {"messages": [{"role": "assistant", "content": ai_message}]}


def localized_editor_agent(state: MultiAgentState) -> Dict:
    """Node 2: Evaluates the writer's work, structures it, and adds social tags."""
    print("✍️ [Node: Localized Editor] Refining text formatting and optimizing distribution tags...")
    
    formatted_messages = []
    for msg in state["messages"]:
        role = msg.type if hasattr(msg, 'type') else msg.get('role', 'user')
        
        # Exact API Alignment Mapping
        if role == "human": role = "user"
        if role == "ai": role = "assistant"  # Fixes the downstream crash!
        
        formatted_messages.append({
            "role": role,
            "content": msg.content if hasattr(msg, 'content') else msg.get('content', '')
        })
    
    system_instruction = (
        "You are a professional social media editor. Review the last cinematic hook generated in the conversation history. "
        "Your job is to polish it for a regional audience by making it incredibly neat, "
        "ensuring correct grammar, and adding 3 highly relevant viral movie hashtags. "
        "Output ONLY the final polished text with its hashtags."
    )
    
    payload = [{"role": "system", "content": system_instruction}] + formatted_messages
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=payload,
        temperature=0.2
    )
    
    ai_message = response.choices[0].message.content
    print(f"-> Editor Output: \"{ai_message}\"")
    
    return {"messages": [{"role": "assistant", "content": ai_message}]}
# 4. Assemble the Collaborative Graph Topology
builder = StateGraph(MultiAgentState)

# Register our agents into the processing layout
builder.add_node("writer", creative_writer_agent)
builder.add_node("editor", localized_editor_agent)

# Map the directional assembly highway
builder.set_entry_point("writer")  # Input goes to the writer first
builder.add_edge("writer", "editor") # Writer output passes automatically to the editor
builder.add_edge("editor", END)     # Editor wraps up the workflow

# Compile the graph into an active executable application
multi_agent_app = builder.compile()

# --- Execution Test Run ---
if __name__ == "__main__":
    print("🚀 Initializing Multi-Agent Conversation Network...")
    
    # The initial state contains a single user request message block
    initial_input = {
        "messages": [
            {"role": "user", "content": "Write a teaser review hook for a high-octane action film starring Jr NTR."}
        ]
    }
    
    # Run the compiled multi-agent application graph
    final_state = multi_agent_app.invoke(initial_input)
    
    print("\n" + "="*60)
    print("🏁 FINAL COMPILED PIPELINE RESULT:")
    print("="*60)
    # The last message in our reduced list is the editor's finalized polish
    print(final_state["messages"][-1].content)