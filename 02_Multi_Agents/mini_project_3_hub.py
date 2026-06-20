import os
from typing import Annotated, Dict
from typing_extensions import TypedDict
from groq import Groq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# 1. Bootstrap Environment & Fast Inference Engine
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("CRITICAL: GROQ_API_KEY is missing from your .env file!")

groq_client = Groq()

# 2. Define the Poly-Functional State Schema
class ProjectState(TypedDict):
    messages: Annotated[list, add_messages]
    next_node: str
    feedback_log: str  # Tracks verification errors to guide the writer's revision loop

# Helper utility to strip framework wrappers for the raw Groq API
def prepare_history_for_groq(message_stream):
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

# 3. Define the Supervisor Node (The Triage Hub)
def supervisor_router_node(state: ProjectState) -> Dict:
    print("\n🧠 [Node: Supervisor Manager] Analyzing pipeline vector state...")
    history = prepare_history_for_groq(state["messages"])
    
    # If the validator just flagged an error, immediately route back to the writer for fixing
    if state.get("feedback_log"):
        print("-> Alert: Active error logs detected. Routing payload back to Writer for revision.")
        return {"next_node": "writer"}
        
    system_instruction = (
        "You are an executive triage manager. Analyze the user request. "
        "If it requires writing content, reviews, hooks, or guides, route to 'writer'. "
        "Respond with exactly one word: 'writer'. Do not include punctuation."
    )
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_instruction}] + history,
        temperature=0.0
    )
    
    decision = response.choices[0].message.content.strip().lower()
    return {"next_node": decision}

# 4. Define the Writer Node (The Content Engine)
def content_writer_node(state: ProjectState) -> Dict:
    print("✍️ [Node: Content Writer] Compiling draft asset...")
    history = prepare_history_for_groq(state["messages"])
    feedback = state.get("feedback_log", "")
    
    base_instruction = (
        "You are an expert content script writer. Your job is to fulfill the user's core request "
        "with an intense, high-impact delivery. CRITICAL COMPLIANCE RULE: Do not wrap your answer in "
        "conversational chatter or introductory phrases. Provide ONLY the final output text."
    )
    
    # If this is a revision loop, inject the validator's critique directly into the system layer
    if feedback:
        base_instruction += (
            f"\n\n⚠️ CRITICAL REVISION NOTICE: Your previous draft was REJECTED by the Validator. "
            f"You must rewrite the content to completely fix this specific error: '{feedback}'"
        )
        
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": base_instruction}] + history,
        temperature=0.7
    )
    
    draft = response.choices[0].message.content.strip()
    print(f"-> Draft Compiled: \"{draft}\"")
    
    # Clear out the feedback log since we have generated a fresh revision attempt
    return {"messages": [{"role": "assistant", "content": draft}], "feedback_log": ""}

# 5. Define the Validator Node (The Quality Control Gate)
def quality_validator_node(state: ProjectState) -> Dict:
    print("🔍 [Node: Quality Validator] Executing compliance audit against strict guardrails...")
    
    # Grab the latest draft from the writer
    latest_draft = state["messages"][-1].content
    
    system_instruction = (
        "You are a strict, zero-tolerance quality assurance engineer. "
        "Your sole job is to audit the provided text against this EXACT rule:\n"
        "RULE: The text MUST contain at least one explicit viral hashtag (e.g., #Cinema, #Agri, #Farming).\n\n"
        "If the text successfully includes a hashtag, reply with exactly one word: 'PASSED'.\n"
        "If the text completely lacks a hashtag, write a brief, direct, one-sentence feedback message "
        "explaining how to fix the error. Do not write 'PASSED'."
    )
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": latest_draft}
        ],
        temperature=0.0 # Force objective assessment
    )
    
    audit_result = response.choices[0].message.content.strip()
    
    if "PASSED" in audit_result.upper():
        print("✅ Audit Verdict: 100% Compliant. Accessing production release gates.")
        return {"next_node": "finalize", "feedback_log": ""}
    else:
        print(f"❌ Audit Verdict: REJECTED! Reason: {audit_result}")
        return {"next_node": "writer", "feedback_log": audit_result}

# 6. Build the Routing Conditional Edge Switch
def evaluate_conditional_edge(state: ProjectState) -> str:
    return state["next_node"]

# 7. Assemble the Graph Structure
workflow_builder = StateGraph(ProjectState)

# Register our computing units
workflow_builder.add_node("supervisor", supervisor_router_node)
workflow_builder.add_node("writer", content_writer_node)
workflow_builder.add_node("validator", quality_validator_node)

# Map the network topology
workflow_builder.set_entry_point("supervisor")
workflow_builder.add_edge("writer", "validator")

# The Supervisor can choose where to dispatch initial traffic
workflow_builder.add_conditional_edges(
    "supervisor",
    evaluate_conditional_edge,
    {
        "writer": "writer"
    }
)

# The Validator determines whether the graph ends or loops backward
workflow_builder.add_conditional_edges(
    "validator",
    evaluate_conditional_edge,
    {
        "finalize": END,
        "writer": "writer"
    }
)

# Compile into an active state machine
app = workflow_builder.compile()

# --- Execution Test Flight ---
if __name__ == "__main__":
    print("🚀 Initializing Self-Correcting Critic Loop Network...")
    
    # Intentionally sending a prompt that will trigger a validation rejection on the first pass
    # because it doesn't mention hashtags, forcing the system to auto-correct itself!
    input_payload = {
        "messages": [
            {"role": "user", "content": "Write a 1-sentence teaser hook for a blockbuster movie starring Prabhas. Do not include hashtags yet."}
        ],
        "next_node": "",
        "feedback_log": ""
    }
    
    final_state_output = app.invoke(input_payload)
    
    print("\n" + "="*60)
    print("🏁 FINAL PRODUCTION RELEASE COMPILED:")
    print("="*60)
    print(final_state_output["messages"][-1].content)