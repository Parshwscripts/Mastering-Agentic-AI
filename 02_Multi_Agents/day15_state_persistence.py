import os
from typing import Annotated, Dict
from typing_extensions import TypedDict
from groq import Groq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
# CRITICAL IMPORT: The state checkpoint module
from langgraph.checkpoint.memory import MemorySaver

# 1. Bootstrap and Verify Environment
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("CRITICAL: GROQ_API_KEY is missing from your .env file!")

groq_client = Groq()

# 2. Define State Schema
class PersistentState(TypedDict):
    messages: Annotated[list, add_messages]

# Helper function to align framework objects with raw Groq JSON standards
def clean_messages_for_api(message_stream):
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

# 3. Define Chat Node
def core_chat_agent(state: PersistentState) -> Dict:
    print(" [Node: Chat Agent] Generating response using active thread memory context...")
    history = clean_messages_for_api(state["messages"])
    
    system_instruction = (
        "You are an elite, helpful AI assistant. Answer the user's question concisely. "
        "Keep the tone professional and brief."
    )
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_instruction}] + history,
        temperature=0.4
    )
    
    output_text = response.choices[0].message.content.strip()
    return {"messages": [{"role": "assistant", "content": output_text}]}

# 4. Construct the Graph Topology
builder = StateGraph(PersistentState)
builder.add_node("agent", core_chat_agent)
builder.set_entry_point("agent")
builder.add_edge("agent", END)

# 5. Compile the Graph WITH a Checkpoint Engine
# This turns our transient state graph into a persistent state machine!
memory_checkpoint_db = MemorySaver()
app = builder.compile(checkpointer=memory_checkpoint_db)

# --- Dynamic Thread Verification Execution ---
if __name__ == "__main__":
    print(" Booting Persistent LangGraph Engine...")
    
    # Define two completely distinct user session tracking configurations
    config_user_alpha = {"configurable": {"thread_id": "user_session_alpha"}}
    config_user_beta = {"configurable": {"thread_id": "user_session_beta"}}
    
    # ---- STEP 1: Interact with User Alpha ----
    print("\n---  Turning Turn 1: User Alpha introduces themselves ---")
    input_1 = {"messages": [{"role": "user", "content": "Hi, my name is Alex and I manage a 2.5-acre plot."}]}
    # We pass the input AND the specific thread configuration database pointer
    res_1 = app.invoke(input_1, config_user_alpha)
    print(f"Agent to Alpha: \"{res_1['messages'][-1].content}\"")
    
    # ---- STEP 2: Interact with User Beta ----
    print("\n---  Turning Turn 2: User Beta asks an isolated question ---")
    input_2 = {"messages": [{"role": "user", "content": "What is the capital of Maharashtra?"}]}
    res_2 = app.invoke(input_2, config_user_beta)
    print(f"Agent to Beta: \"{res_2['messages'][-1].content}\"")
    
    # ---- STEP 3: Follow up with User Alpha (The Ultimate Memory Test) ----
    print("\n---  Turning Turn 3: User Alpha requests contextual follow-up ---")
    # Note: We are NOT sending the name 'Alex' or the '2.5-acre' string in this prompt!
    input_3 = {"messages": [{"role": "user", "content": "What was my name and how big is my farm plot again?"}]}
    res_3 = app.invoke(input_3, config_user_alpha) # Accessing alpha's memory thread
    print(f"Agent to Alpha: \"{res_3['messages'][-1].content}\"")