from typing import Dict, TypedDict
from langgraph.graph import StateGraph, END

# 1. Define the Global State Structure
class AgentState(TypedDict):
    """
    This dictionary represents the shared memory of our graph.
    Every node reads from and writes to this structure.
    """
    raw_input: str
    processed_text: str
    word_count: int
    validation_passed: bool

# 2. Define the Nodes (The Workers)
def editor_agent_node(state: AgentState) -> Dict:
    """Reads the raw text and refines it into a cinematic, punchy hook."""
    print("🎬 [Node 1: Editor Agent] Processing text payload...")
    text = state["raw_input"]
    
    # Simulating an LLM editing adjustment (making it uppercase and adding a hook element)
    refined = f"💥 BREAKING RECON: {text.upper()}!! #MustWatch"
    
    # Return ONLY the keys we want to update in the global state
    return {
        "processed_text": refined,
        "word_count": len(refined.split())
    }

def validator_agent_node(state: AgentState) -> Dict:
    """Evaluates if the refined text meets our character length guardrails."""
    print("🔍 [Node 2: Validator Agent] Auditing structural constraints...")
    text_to_check = state["processed_text"]
    
    # Constraint: Content must be under 12 words to fit social media specs
    is_valid = len(text_to_check.split()) < 12
    
    return {"validation_passed": is_valid}

# 3. Define the Routing Logic (The Conditional Edge)
def route_based_on_validation(state: AgentState) -> str:
    """Evaluates the state and determines the next logical path execution."""
    if state["validation_passed"]:
        print("✅ Audit Passed! Routing workflow to compilation exit.")
        return "end_workflow"
    else:
        print("❌ Audit Failed! Flagging structural layout violation.")
        return "trigger_fallback"

def fallback_handler_node(state: AgentState) -> Dict:
    """Fixes the text if validation fails by forcing a hard truncation."""
    print("⚠️ [Node 3: Fallback Handler] Truncating text to force-compliance...")
    words = state["processed_text"].split()
    truncated_text = " ".join(words[:8]) + "..."
    return {"processed_text": truncated_text, "validation_passed": True}


# 4. Construct the Graph Topology
workflow = StateGraph(AgentState)

# Add our processing units to the grid configuration
workflow.add_node("editor", editor_agent_node)
workflow.add_node("validator", validator_agent_node)
workflow.add_node("fallback", fallback_handler_node)

# Set the Entry Point - Where the data flow begins
workflow.set_entry_point("editor")

# Draw the explicit processing paths (Edges)
workflow.add_edge("editor", "validator")

# Add a Conditional Routing Edge leaving the validator node
workflow.add_conditional_edges(
    "validator",
    route_based_on_validation,
    {
        "end_workflow": END,           # END is a special LangGraph constant to stop execution
        "trigger_fallback": "fallback"
    }
)

# Connect the fallback node back to the END pipeline
workflow.add_edge("fallback", END)

# 5. Compile the Graph into an executable application
app = workflow.compile()

# --- Execution Test Run ---
if __name__ == "__main__":
    print("🚀 Initializing LangGraph Engine...")
    
    # Seed the initial dictionary state payload
    initial_payload = {
        "raw_input": "this movie has an incredible plot twist at the very end that changes everything",
        "processed_text": "",
        "word_count": 0,
        "validation_passed": False
    }
    
    # Run the graph synchronously
    final_output = app.invoke(initial_payload)
    
    print("\n--- 🏁 Final State Compilation Results ---")
    print(f"Final Output Text: \"{final_output['processed_text']}\"")
    print(f"Validation Status Verified: {final_output['validation_passed']}")