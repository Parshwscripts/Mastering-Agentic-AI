import os
from typing import Annotated, Dict
from typing_extensions import TypedDict
from groq import Groq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# 1. Bootstrap Environment
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("CRITICAL: GROQ_API_KEY is missing from your .env file!")

groq_client = Groq()

# 2. Define State Schema
class TimeTravelState(TypedDict):
    messages: Annotated[list, add_messages]
    internal_flag: str  # A tracking metric we will alter using time-travel

# Helper utility to format messages cleanly for Groq
def parse_for_groq(stream):
    formatted = []
    for msg in stream:
        role = msg.type if hasattr(msg, 'type') else msg.get('role', 'user')
        if role == "human": role = "user"
        if role == "ai": role = "assistant"
        formatted.append({
            "role": role,
            "content": msg.content if hasattr(msg, 'content') else msg.get('content', '')
        })
    return formatted

# 3. Define the Core Node
def standard_agent_node(state: TimeTravelState) -> Dict:
    print("🤖 [Node: Agent Core] Processing active payload...")
    history = parse_for_groq(state["messages"])
    flag = state.get("internal_flag", "NORMAL_MODE")
    
    system_instruction = (
        f"You are a helpful assistant. Current system flag constraint: [{flag}]. "
        "If flag is CRITICAL_OVERRIDE, your tone must change to highly formal and alert. "
        "Otherwise, keep the tone warm and friendly."
    )
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_instruction}] + history,
        temperature=0.3
    )
    return {"messages": [{"role": "assistant", "content": response.choices[0].message.content}]}

# 4. Assemble and Compile Persistent Graph
builder = StateGraph(TimeTravelState)
builder.add_node("agent", standard_agent_node)
builder.set_entry_point("agent")
builder.add_edge("agent", END)

checkpointer_db = MemorySaver()
app = builder.compile(checkpointer=checkpointer_db)

# --- Time Travel Manipulation Phase ---
if __name__ == "__main__":
    print("🚀 Initializing Time-Travel Exploration Sandbox...")
    
    # Establish our execution tracking thread configuration
    active_thread_config = {"configurable": {"thread_id": "sandbox_session_99"}}
    
    # ---- TURN 1: Initialize baseline state ----
    print("\n--- ⏳ Step 1: Running Normal Execution Turn ---")
    initial_input = {
        "messages": [{"role": "user", "content": "Hello! Give me a quick greeting."}],
        "internal_flag": "NORMAL_MODE"
    }
    state_turn_1 = app.invoke(initial_input, active_thread_config)
    print(f"Agent Response 1: \"{state_turn_1['messages'][-1].content}\"")
    
    # ---- STEP 2: Inspect the Historical Ledger ----
    print("\n--- 🔍 Step 2: Querying the Checkpoint Database History Ledger ---")
    # Retrieve all saved snapshots for this thread id
    history_ledger = list(app.get_state_history(active_thread_config))
    
    print(f"Total historical checkpoints recorded for this session: {len(history_ledger)}")
    for record in history_ledger:
        print(f"-> Checkpoint ID: {record.config['configurable']['checkpoint_id']} | Next Node Destination: {record.next}")
    
    # Let's target the immediate past checkpoint configuration
    historical_snapshot = history_ledger[-1] 
    past_checkpoint_config = historical_snapshot.config
    
    # ---- STEP 3: Rewind & Modify the State values ----
    print("\n--- 🛠️ Step 3: Rewinding State and Injecting Data Overrides ---")
    # We update the state at that specific past checkpoint configuration path
    app.update_state(
        past_checkpoint_config,
        {"internal_flag": "CRITICAL_OVERRIDE"}, # Inject a brand new value manually
        as_node="agent" # Tell the state machine we are mimicking an update inside this node
    )
    
    # ---- STEP 4: Replay Time from the modified checkpoint ----
    print("\n--- 🔄 Step 4: Replaying Execution Path from the Altered Timeline ---")
    # Invoke the graph passing None as input, but supplying our modified historic config map
    forked_state_result = app.invoke(None, past_checkpoint_config)
    print(f"Agent Forked Response (New Timeline): \"{forked_state_result['messages'][-1].content}\"")