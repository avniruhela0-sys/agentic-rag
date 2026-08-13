"""
Day 8: Learn LangGraph Basics
Understanding State, Nodes, Edges, and Graph Flow
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import random

print("=" * 60)
print("🤖 DAY 8: LANGGRAPH BASICS")
print("=" * 60)

# ============================================
# Part 1: What is LangGraph?
# ============================================
print("\n📚 WHAT IS LANGGRAPH?")
print("-" * 40)
print("""
LangGraph is a framework for building stateful, multi-step agents.

Key Concepts:
1. State: A dictionary that holds data as it flows through the graph
2. Nodes: Functions that process the state
3. Edges: Connections between nodes (directed flow)
4. Conditional Edges: Decision points (if/else routing)

Think of it like a flowchart:
    Node A → Node B → Node C → End
              ↑         ↓
          (conditional)
""")

# ============================================
# Part 2: Define State
# ============================================
print("\n📦 PART 2: DEFINING STATE")
print("-" * 40)

# State is a TypedDict that defines what data flows through our graph
class SimpleState(TypedDict):
    count: int
    message: str
    history: List[str]

print("""
State = TypedDict:
    count: int          # A counter
    message: str        # A message
    history: List[str]  # Track what happened

State is passed between nodes and updated at each step.
""")

# ============================================
# Part 3: Define Nodes (Functions)
# ============================================
print("\n🎯 PART 3: DEFINING NODES")
print("-" * 40)

# Node 1: Add One
def add_one(state: SimpleState):
    """Node that increments the count by 1"""
    print(f"   🔢 add_one: count {state['count']} → {state['count'] + 1}")
    return {
        "count": state["count"] + 1,
        "history": state["history"] + ["added 1"]
    }

# Node 2: Double
def double(state: SimpleState):
    """Node that doubles the count"""
    print(f"   🔢 double: count {state['count']} → {state['count'] * 2}")
    return {
        "count": state["count"] * 2,
        "history": state["history"] + ["doubled"]
    }

# Node 3: Set Message
def set_message(state: SimpleState):
    """Node that sets a message based on count"""
    msg = f"Count is now {state['count']}"
    print(f"   💬 set_message: '{msg}'")
    return {
        "message": msg,
        "history": state["history"] + ["set message"]
    }

# Node 4: Check Even
def check_even(state: SimpleState):
    """Node that checks if count is even"""
    is_even = state["count"] % 2 == 0
    print(f"   🔍 check_even: {state['count']} is {'even' if is_even else 'odd'}")
    return {
        "message": f"{state['count']} is {'even' if is_even else 'odd'}",
        "history": state["history"] + ["checked even"]
    }

print("""
Nodes are Python functions that:
1. Take state as input
2. Process the state
3. Return updates to the state

Example Node:
def add_one(state: SimpleState):
    return {"count": state["count"] + 1}
""")

# ============================================
# Part 4: Build a Simple Graph
# ============================================
print("\n🏗️ PART 4: BUILDING A SIMPLE GRAPH")
print("-" * 40)

print("""
Graph Flow:
    Start → add_one → double → set_message → End
""")

# Create the graph with our state
simple_graph = StateGraph(SimpleState)

# Add nodes
simple_graph.add_node("add_one", add_one)
simple_graph.add_node("double", double)
simple_graph.add_node("set_message", set_message)

# Set entry point
simple_graph.set_entry_point("add_one")

# Add edges (connections)
simple_graph.add_edge("add_one", "double")
simple_graph.add_edge("double", "set_message")
simple_graph.add_edge("set_message", END)

# Compile the graph
app_simple = simple_graph.compile()

print("✅ Simple graph created!")
print("   Flow: add_one → double → set_message → END")

# ============================================
# Part 5: Run the Simple Graph
# ============================================
print("\n▶️ PART 5: RUNNING THE SIMPLE GRAPH")
print("-" * 40)

print("\nInitial state: {'count': 5, 'message': '', 'history': []}\n")

result = app_simple.invoke({
    "count": 5,
    "message": "",
    "history": []
})

print("\n📊 Final result:")
print(f"   Count: {result['count']}")
print(f"   Message: {result['message']}")
print(f"   History: {result['history']}")

# ============================================
# Part 6: Build a Graph with Conditional Edges
# ============================================
print("\n🔀 PART 6: CONDITIONAL EDGES (IF/ELSE)")
print("-" * 40)

print("""
Conditional edges let us route based on logic:
    
    Start → add_one → check_even
                        ↓
                  ┌─────┴─────┐
                  ↓           ↓
              (even)       (odd)
                  ↓           ↓
              double    set_message
                  ↓           ↓
                └──────┬──────┘
                       ↓
                     End
""")

# Define the routing function
def route_based_on_state(state: SimpleState):
    """Decide which node to go to next"""
    if state["count"] % 2 == 0:
        print(f"   🔀 Routing: count {state['count']} is even → double")
        return "double"
    else:
        print(f"   🔀 Routing: count {state['count']} is odd → set_message")
        return "set_message"

# Create graph with conditional edges
conditional_graph = StateGraph(SimpleState)

# Add nodes
conditional_graph.add_node("add_one", add_one)
conditional_graph.add_node("check_even", check_even)
conditional_graph.add_node("double", double)
conditional_graph.add_node("set_message", set_message)

# Set entry point
conditional_graph.set_entry_point("add_one")

# Add edges
conditional_graph.add_edge("add_one", "check_even")

# Add conditional edge from check_even
conditional_graph.add_conditional_edges(
    "check_even",
    route_based_on_state,  # Function that returns next node
    {
        "double": "double",        # If returns "double", go to double
        "set_message": "set_message"  # If returns "set_message", go to set_message
    }
)

conditional_graph.add_edge("double", END)
conditional_graph.add_edge("set_message", END)

# Compile
app_conditional = conditional_graph.compile()

print("✅ Conditional graph created!")

# ============================================
# Part 7: Run the Conditional Graph
# ============================================
print("\n▶️ PART 7: RUNNING THE CONDITIONAL GRAPH")
print("-" * 40)

print("\nTest 1: Starting with count = 5 (odd)")
print("-" * 20)

result1 = app_conditional.invoke({
    "count": 5,
    "message": "",
    "history": []
})

print(f"\nFinal: {result1}")

print("\nTest 2: Starting with count = 4 (even)")
print("-" * 20)

result2 = app_conditional.invoke({
    "count": 4,
    "message": "",
    "history": []
})

print(f"\nFinal: {result2}")

# ============================================
# Part 8: Understanding the Agent State Pattern
# ============================================
print("\n🧠 PART 8: AGENT STATE PATTERN")
print("-" * 40)

print("""
In our Agentic RAG project, we'll use this pattern:

class AgentState(TypedDict):
    question: str                    # The original question
    sub_questions: List[str]         # Planned sub-questions
    retrieved_docs: List[str]        # Retrieved documents
    draft_answer: str                # Generated answer
    judge_score: float               # Quality score
    judge_feedback: str              # Judge feedback
    correction_attempts: int         # Number of corrections
    final_answer: str                # Final answer

Nodes:
    1. Plan: Break question into sub-questions
    2. Retrieve: Get documents for each sub-question
    3. Generate: Write answer
    4. Judge: Evaluate answer quality
    5. Correct: Fix if needed
    6. Finalize: Return final answer

Conditional Edges:
    Judge → (score >= 7) → Finalize
    Judge → (score < 7) → Correct → Generate → Judge
""")

# ============================================
# Part 9: Summary
# ============================================
print("\n📊 DAY 8 SUMMARY")
print("-" * 40)
print("""
✅ Learned what LangGraph is and why we use it
✅ Understood State (data flow)
✅ Created Nodes (functions)
✅ Built Edges (connections)
✅ Used Conditional Edges (if/else routing)
✅ Ran graphs and saw results

🎯 Key Concepts:
   1. State: Data dictionary that flows through the graph
   2. Nodes: Functions that process state
   3. Edges: Connections between nodes
   4. Conditional Edges: Decision points

📋 Remember:
   - State is IMMUTABLE (we return new state, don't modify)
   - Nodes return UPDATES (only changed fields)
   - Conditional edges ROUTE based on state
   - Graphs are COMPILED before running
""")

print("\n🚀 Ready for Day 9! (Agent State)")
print("=" * 60)