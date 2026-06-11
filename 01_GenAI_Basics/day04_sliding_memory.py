import os
from groq import Groq
from dotenv import load_dotenv

# Securely load system environmental variables
load_dotenv()

class SlidingWindowAgent:
    def __init__(self, system_instruction, max_turns=2):
        """
        max_turns: Maximum number of user-assistant pairs to retain in memory.
        """
        self.client = Groq()
        self.model = "llama-3.3-70b-versatile"
        self.system_instruction = system_instruction
        self.max_turns = max_turns
        # This list maintains the active state of the conversation
        self.history = []

    def send_message(self, user_query):
        # 1. Compile the active payload starting with the permanent system rule
        active_payload = [{"role": "system", "content": self.system_instruction}]
        
        # 2. Append the current sliding memory window
        active_payload.extend(self.history)
        
        # 3. Append the brand new incoming user query
        active_payload.append({"role": "user", "content": user_query})
        
        # 4. Execute the API call using the 70B model
        response = self.client.chat.completions.create(
            model=self.model,
            messages=active_payload,
            temperature=0.7
        )
        
        assistant_response = response.choices[0].message.content
        
        # 5. Commit this turn to our memory tracking list
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": assistant_response})
        
        # 6. SLIDING WINDOW PRUNING
        # Each complete turn consists of 2 messages (1 User, 1 Assistant)
        max_messages_allowed = self.max_turns * 2
        if len(self.history) > max_messages_allowed:
            print(f"\n[SYSTEM NOTICE] Memory threshold exceeded. Pruning oldest conversational exchange...")
            # Slice the array to keep only the most recent messages
            self.history = self.history[-max_messages_allowed:]
            
        return assistant_response

# ---- Simulation Execution ----
if __name__ == "__main__":
    # Initialize our agent with a strict 2-turn memory window
    bot = SlidingWindowAgent(
        system_instruction="You are a precise technical interviewer. Keep replies under one sentence.",
        max_turns=2
    )

    # Turn 1
    print("User: I specialize in Python and Machine Learning.")
    print(f"AI: {bot.send_message('I specialize in Python and Machine Learning.')}")

    # Turn 2
    print("\nUser: I want to build a career in Agentic AI systems.")
    print(f"AI: {bot.send_message('I want to build a career in Agentic AI systems.')}")

    # Turn 3 -> This will trigger the pruning condition for Turn 1
    print("\nUser: My current goal is passing upcoming college placement drives.")
    print(f"AI: {bot.send_message('My current goal is passing upcoming college placement drives.')}")

    print("\n--- TEST: TESTING MEMORY LIMITS ---")
    
    # Testing Turn 3 memory (Should know it instantly)
    print("\nUser: What is my primary short-term goal?")
    print(f"AI: {bot.send_message('What is my primary short-term goal?')}")

    # Testing Turn 1 memory (Should be completely forgotten due to pruning)
    print("\nUser: What are my core programming specializations?")
    print(f"AI: {bot.send_message('What are my core programming specializations?')}")