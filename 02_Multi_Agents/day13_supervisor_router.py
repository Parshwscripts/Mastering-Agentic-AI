import os
from typing import Annotated, Dict
from typing_extensions import TypedDict
from groq import Groq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# 1. Bootstrap and Verify Environment
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("CRITICAL: GROQ_API_KEY is missing from your .env file!")

groq_client = Groq()

# 2. Define the State Layout
class RouterState(TypedDict):
    messages: Annotated[list, add_messages]
    next_node: str  # The Supervisor writes the routing decision here

# Helper function to parse LangGraph message objects into strict Groq JSON format
def parse_messages_for_groq(message_list):
    formatted = []
    for msg in message_list:
        role = msg.type if hasattr(msg, 'type') else msg.get('role', 'user')
        if role == "human": role = "user"
        if role == "ai": role = "assistant"
        formatted.append({
            "role": role,
            "content": msg.content if hasattr(msg, 'content') else msg.get('content', '')
        })
    return formatted

# 3. Define the Supervisor Node (The Router)
def supervisor_router_node(state: RouterState) -> Dict:
    print("\n [Node: Supervisor Router] Evaluating user intent...")
    history = parse_messages_for_groq(state["messages"])
    
    system_instruction = (
        "You are an elite triage supervisor manager. Your job is to analyze the user's latest query "
        "and route it to the single best specialist worker on your team.\n\n"
        "Your team members are:\n"
        "1. 'cinema_worker': Handles anything related to movies, movie reviews, script hooks, or actors.\n"
        "2. 'agri_worker': Handles anything related to farming, crops, soil, sugarcane, or fertilizers.\n\n"
        "CRITICAL REQUIREMENT: You must respond with exactly one word—the name of the node to route to. "
        "Output ONLY either 'cinema_worker' or 'agri_worker'. Do not write any other text, punctuation, or analysis."
    )
    
    payload = [{"role": "system", "content": system_instruction}] + history
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=payload,
        temperature=0.0 # Force absolute deterministic classification accuracy
    )
    
    routing_decision = response.choices[0].message.content.strip()
    print(f"-> Supervisor Routing Decision: Going to [{routing_decision}]")
    
    return {"next_node": routing_decision}

# 4. Define Specialist Worker Nodes
def cinema_worker_node(state: RouterState) -> Dict:
    print(" [Node: Cinema Worker] Processing media query...")
    history = parse_messages_for_groq(state["messages"])
    
    system_instruction = "You are a professional film reviewer. Answer the query with a sharp, high-energy cinematic insight."
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_instruction}] + history,
        temperature=0.7
    )
    return {"messages": [{"role": "assistant", "content": response.choices[0].message.content}]}

def agri_worker_node(state: RouterState) -> Dict:
    print(" [Node: Agri Worker] Processing agricultural calculations...")
    history = parse_messages_for_groq(state["messages"])
    
    system_instruction = "You are an expert agronomist specializing in crop yields. Provide a mathematically precise farming answer."
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_instruction}] + history,
        temperature=0.1
    )
    return {"messages": [{"role": "assistant", "content": response.choices[0].message.content}]}

# 5. Build Edge Routing Rule
def route_next(state: RouterState) -> str:
    """Reads the state's routing key and shifts data flow instantly."""
    return state["next_node"]

# 6. Assemble Graph Configuration
builder = StateGraph(RouterState)

# Register our workers
builder.add_node("supervisor", supervisor_router_node)
builder.add_node("cinema_worker", cinema_worker_node)
builder.add_node("agri_worker", agri_worker_node)

# Set global sequence paths
builder.set_entry_point("supervisor")

# Configure the Supervisor's dynamic conditional highway outward
builder.add_conditional_edges(
    "supervisor",
    route_next,
    {
        "cinema_worker": "cinema_worker",
        "agri_worker": "agri_worker"
    }
)

# Connect specialist exits cleanly to the END of the graph lifecycle
builder.add_edge("cinema_worker", END)
builder.add_edge("agri_worker", END)

app = builder.compile()

# --- Execution Test Phase ---
if __name__ == "__main__":
    # Test Case A: Agriculture Domain
    print("=== RUNNING TEST A ===")
    test_a = {"messages": [{"role": "user", "content": "What NPK ratio should I apply to sugarcane during the earthing up phase?"}], "next_node": ""}
    output_a = app.invoke(test_a)
    print(f" Final Answer A:\n{output_a['messages'][-1].content}\n")
    
    print("="*50 + "\n")
    
    # Test Case B: Cinema Domain
    print("=== RUNNING TEST B ===")
    test_b = {"messages": [{"role": "user", "content": "Give me a high-octane review line for Thalapathy Vijay's screen presence."}], "next_node": ""}
    output_b = app.invoke(test_b)
    print(f" Final Answer B:\n{output_b['messages'][-1].content}")