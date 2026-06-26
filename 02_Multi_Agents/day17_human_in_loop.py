import os
from typing import Annotated, Dict
from typing_extensions import TypedDict
from groq import Groq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# 1. Bootstrap and Verify Environment
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("CRITICAL: GROQ_API_KEY is missing from your .env file!")

groq_client = Groq()

# 2. Define State Layout
class HITLState(TypedDict):
    messages: Annotated[list, add_messages]
    deployment_ready: bool

# Helper function to align framework objects with raw Groq JSON standards
def parse_stream_for_api(message_stream):
    formatted = []
    for msg in message_stream:
        role = msg.type if hasattr(msg, 'type') else msg.get('role', 'user')
        if role == "human": role = "user"
        if role == "ai": role = "assistant"
        formatted.append({
            "role": role,
            "content": msg.content if hasattr(msg, 'content') else msg.get('content', '')
        })
    return formatted

# 3. Define the Processing Nodes
def content_generation_node(state: HITLState) -> Dict:
    print("✍️ [Node 1: Content Creator] Generating professional copy asset...")
    history = parse_stream_for_api(state["messages"])
    
    system_instruction = (
        "You are an expert copywriter. Create a ultra-short, high-impact marketing teaser line "
        "based on the user input. Output ONLY the teaser line text."
    )
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_instruction}] + history,
        temperature=0.7
    )
    return {"messages": [{"role": "assistant", "content": response.choices[0].message.content.strip()}]}


def production_deployment_node(state: HITLState) -> Dict:
    """The critical 'protected' node that should never execute without human review."""
    print("🚀 [Node 2: Production Deployment] CRITICAL ACTION EXECUTING!")
    print(f"-> Publishing to Cloud Live Vector: \"{state['messages'][-1].content}\"")
    return {"deployment_ready": True}


# 4. Construct the Graph and Enforce Compiling Breakpoints
builder = StateGraph(HITLState)

builder.add_node("creator", content_generation_node)
builder.add_node("deployment", production_deployment_node)

builder.set_entry_point("creator")
builder.add_edge("creator", "deployment")
builder.add_edge("deployment", END)

# Initialize persistence memory backing layer
persistence_checkpointer = MemorySaver()

# CRITICAL COMPILING STEP: Inject an interrupt constraint directly before deployment!
app = builder.compile(
    checkpointer=persistence_checkpointer,
    interrupt_before=["deployment"] # This locks the gate before entering Node 2
)

# --- Execution Gate Driver Loop ---
if __name__ == "__main__":
    print("🚀 Booting Human-in-the-Loop Verification Pipeline...")
    
    # Establish a unique session identifier configuration
    session_config = {"configurable": {"thread_id": "secure_release_999"}}
    
    initial_input = {"messages": [{"role": "user", "content": "Write a teaser review hook for a new sci-fi movie."}]}
    
    # ---- PHASE 1: Run graph up to the breakpoint ----
    print("\n--- 🟢 Executing Phase 1: Launching Generation Pipeline ---")
    current_state = app.invoke(initial_input, session_config)
    
    # Let's inspect where the graph state currently sits
    graph_snapshot = app.get_state(session_config)
    print(f"\n⏸️ Current Graph Execution Paused At: {graph_snapshot.next}")
    print(f"Proposed Asset Pending Release: \"{graph_snapshot.values['messages'][-1].content}\"")
    
    # ---- PHASE 2: Human Interaction Intervention Block ----
    print("\n--- 👥 Human-in-the-Loop Intervention Gate ---")
    user_input = input("⚠️ CRITICAL OVERRIDE ACTION REQUIRED. Type 'approve' to release to production, or any other key to abort: ")
    
    if user_input.strip().lower() == "approve":
        print("\n--- 🔵 Executing Phase 2: Resume Signal Dispatched ---")
        # To resume a paused checkpoint loop, pass None as the input payload while keeping the thread config intact
        final_state = app.invoke(None, session_config)
        print(f"\n🏁 Pipeline Complete. Deployment Status Verified: {final_state.get('deployment_ready')}")
    else:
        print("\n❌ Release Aborted by Human Supervisor! Purging processing thread.")