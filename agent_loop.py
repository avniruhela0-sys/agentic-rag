"""
Day 13: Wire the Complete LangGraph Loop
Building the full agent with plan → retrieve → generate → judge → correct → finalize
"""

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import TypedDict, List, Dict, Any
from datetime import datetime
import json
import os

print("=" * 60)
print("🔄 DAY 13: WIRING THE COMPLETE AGENT LOOP")
print("=" * 60)

# ============================================
# Part 1: Define Agent State
# ============================================
print("\n📦 PART 1: DEFINING AGENT STATE")
print("-" * 40)

class AgentState(TypedDict):
    """The complete state of our agentic RAG system"""
    # Input
    question: str
    
    # Planning
    sub_questions: List[str]
    
    # Retrieval
    retrieved_docs: List[str]
    
    # Generation
    draft_answer: str
    
    # Evaluation
    judge_score: float
    judge_feedback: str
    judge_issue: str
    
    # Correction
    correction_attempts: int
    correction_history: List[Dict[str, Any]]
    
    # Final
    final_answer: str
    sources: List[str]
    
    # Metadata
    start_time: str
    node_history: List[str]

def create_initial_state(question: str) -> AgentState:
    """Create a fresh state for a new question."""
    return AgentState(
        question=question,
        sub_questions=[],
        retrieved_docs=[],
        draft_answer="",
        judge_score=0.0,
        judge_feedback="",
        judge_issue="none",
        correction_attempts=0,
        correction_history=[],
        final_answer="",
        sources=[],
        start_time=datetime.now().isoformat(),
        node_history=[]
    )

print("✅ AgentState defined!")

# ============================================
# Part 2: Load Vector Store and Initialize LLM
# ============================================
print("\n🔧 PART 2: LOADING COMPONENTS")
print("-" * 40)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Load vector store
vector_store_paths = ['chroma_db_hf', 'chroma_db', 'chroma_db_50']
vectorstore = None
loaded_path = None

for path in vector_store_paths:
    if os.path.exists(path):
        try:
            print(f"   Trying: {path}...")
            vectorstore = Chroma(
                persist_directory=path,
                embedding_function=embeddings
            )
            loaded_path = path
            print(f"✅ Loaded vector store from: {path}")
            break
        except Exception as e:
            print(f"   ❌ Could not load from {path}: {e}")

if vectorstore is None:
    print("❌ No vector store found!")
    print("💡 Please run Day 4 first (vector_store_hf.py)")
    exit(1)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print(f"✅ Retriever created! (k=4)")

# Initialize LLM
try:
    llm = ChatOllama(
        model="mistral",
        temperature=0,
    )
    print("✅ LLM initialized! (Ollama - mistral)")
except Exception as e:
    print(f"❌ Error initializing LLM: {e}")
    exit(1)

# ============================================
# Part 3: Define All Prompts
# ============================================
print("\n📝 PART 3: CREATING PROMPTS")
print("-" * 40)

# 1. Planner Prompt
PLANNER_PROMPT = """
You are a research planner. Break down this question into 1-3 focused sub-questions.

Rules:
1. Each sub-question should target ONE piece of information
2. Sub-questions should be specific and searchable
3. Return ONLY the sub-questions, one per line
4. Do NOT number them

Question: {question}

Sub-questions:
"""
planner_prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)

# 2. Generator Prompt (with Chain-of-Thought)
GENERATOR_PROMPT = """
You are a research assistant. Using the provided context, answer the question.

First, reason step by step in 2-3 sentences. Then give your final answer.

Context:
{context}

Question: {question}

Step-by-step reasoning:
1. What does the context tell us?
2. Is there enough information?
3. What is the most accurate answer?

Final Answer:
"""
generator_prompt = ChatPromptTemplate.from_template(GENERATOR_PROMPT)

# 3. Judge Prompt
JUDGE_PROMPT = """
You are a strict evaluator. Score the answer from 0-10.

- 0-4: Poor, wrong or hallucinated
- 5-6: Mediocre, partially correct
- 7-8: Good, mostly correct
- 9-10: Excellent, completely correct

Also identify any issues:
- hallucination: information not in context
- contradiction: contradicts context
- irrelevant: doesn't answer the question
- none: no issues

Context:
{context}

Question: {question}

Answer: {answer}

Respond exactly as:
SCORE: <number>
ISSUE: <hallucination/contradiction/irrelevant/none>
REASON: <one sentence>
"""
judge_prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)

print("✅ All prompts created!")

# ============================================
# Part 4: Define All Nodes
# ============================================
print("\n🔧 PART 4: DEFINING NODES")
print("-" * 40)

# Node 1: Planner
def plan_node(state: AgentState) -> Dict[str, Any]:
    """Break down the question into sub-questions."""
    print(f"\n   📝 PLANNING: {state['question'][:50]}...")
    
    try:
        response = (planner_prompt | llm).invoke({"question": state["question"]})
        
        sub_questions = []
        for line in response.content.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("Sub-questions:"):
                if line and line[0].isdigit() and len(line) > 2 and line[1] in ". )":
                    line = line[2:].strip()
                if line:
                    sub_questions.append(line)
        
        if not sub_questions:
            sub_questions = [state["question"]]
        
        print(f"      ✅ Generated {len(sub_questions)} sub-questions")
        for i, sq in enumerate(sub_questions, 1):
            print(f"         {i}. {sq}")
        
        return {
            "sub_questions": sub_questions,
            "node_history": state.get("node_history", []) + ["plan"]
        }
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return {
            "sub_questions": [state["question"]],
            "node_history": state.get("node_history", []) + ["plan"]
        }

# Node 2: Retriever
def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """Retrieve documents for each sub-question."""
    sub_questions = state.get("sub_questions", [state["question"]])
    print(f"\n   🔍 RETRIEVING for {len(sub_questions)} sub-question(s)...")
    
    all_docs = []
    all_sources = []
    
    for i, query in enumerate(sub_questions, 1):
        print(f"      Query {i}: {query[:40]}...")
        try:
            docs = retriever.invoke(query)
            for doc in docs:
                all_docs.append(doc.page_content)
                all_sources.append(doc.metadata.get('title', 'Unknown'))
        except Exception as e:
            print(f"         ❌ Error: {e}")
    
    # Remove duplicates
    seen = set()
    unique_docs = []
    unique_sources = []
    for doc, source in zip(all_docs, all_sources):
        if doc not in seen:
            seen.add(doc)
            unique_docs.append(doc)
            unique_sources.append(source)
    
    print(f"      ✅ Retrieved {len(unique_docs)} unique documents")
    
    return {
        "retrieved_docs": unique_docs,
        "sources": unique_sources,
        "node_history": state.get("node_history", []) + ["retrieve"]
    }

# Node 3: Generator
def generate_node(state: AgentState) -> Dict[str, Any]:
    """Generate answer with Chain-of-Thought."""
    print(f"\n   ✍️ GENERATING answer...")
    print(f"      Using {len(state.get('retrieved_docs', []))} documents")
    
    if not state.get("retrieved_docs"):
        print("      ⚠️ No documents retrieved!")
        return {
            "draft_answer": "I don't have enough information to answer this question.",
            "node_history": state.get("node_history", []) + ["generate"]
        }
    
    try:
        context = "\n\n".join(state["retrieved_docs"])
        if len(context) > 4000:
            context = context[:4000] + "..."
        
        response = (generator_prompt | llm).invoke({
            "context": context,
            "question": state["question"]
        })
        
        print(f"      ✅ Generated answer ({len(response.content)} characters)")
        
        return {
            "draft_answer": response.content,
            "node_history": state.get("node_history", []) + ["generate"]
        }
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return {
            "draft_answer": f"Error generating answer: {e}",
            "node_history": state.get("node_history", []) + ["generate"]
        }

# Node 4: Judge
def judge_node(state: AgentState) -> Dict[str, Any]:
    """Evaluate answer quality."""
    print(f"\n   ⚖️ JUDGING answer...")
    
    try:
        context = "\n\n".join(state.get("retrieved_docs", []))
        if len(context) > 2000:
            context = context[:2000] + "..."
        
        response = (judge_prompt | llm).invoke({
            "context": context,
            "question": state["question"],
            "answer": state.get("draft_answer", "")
        })
        
        # Parse response
        text = response.content
        score = 0.0
        issue = "none"
        feedback = ""
        
        for line in text.split("\n"):
            if "SCORE:" in line:
                try:
                    score = float(line.split("SCORE:")[1].strip())
                except:
                    score = 5.0
            if "ISSUE:" in line:
                issue = line.split("ISSUE:")[1].strip().lower()
            if "REASON:" in line:
                feedback = line.split("REASON:")[1].strip()
        
        print(f"      Score: {score}/10")
        print(f"      Issue: {issue}")
        print(f"      Feedback: {feedback}")
        
        return {
            "judge_score": score,
            "judge_feedback": feedback,
            "judge_issue": issue,
            "node_history": state.get("node_history", []) + ["judge"]
        }
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return {
            "judge_score": 5.0,
            "judge_feedback": "Error in judging",
            "judge_issue": "none",
            "node_history": state.get("node_history", []) + ["judge"]
        }

# Node 5: Corrector
def correct_node(state: AgentState) -> Dict[str, Any]:
    """Refine retrieval based on judge feedback."""
    attempts = state.get("correction_attempts", 0) + 1
    print(f"\n   🔧 CORRECTING (attempt {attempts})...")
    print(f"      Feedback: {state.get('judge_feedback', 'No feedback')}")
    
    # Refine query with feedback
    refined_query = f"{state['question']} (focus on: {state.get('judge_feedback', '')})"
    print(f"      Refined query: {refined_query[:60]}...")
    
    try:
        docs = retriever.invoke(refined_query)
        new_docs = state.get("retrieved_docs", []) + [doc.page_content for doc in docs]
        
        # Update history
        history = state.get("correction_history", [])
        history.append({
            "attempt": attempts,
            "feedback": state.get("judge_feedback", ""),
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"      Added {len(docs)} new documents")
        print(f"      Total documents now: {len(new_docs)}")
        
        return {
            "retrieved_docs": new_docs,
            "correction_attempts": attempts,
            "correction_history": history,
            "node_history": state.get("node_history", []) + ["correct"]
        }
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return {
            "correction_attempts": attempts,
            "node_history": state.get("node_history", []) + ["correct"]
        }

# Node 6: Finalizer
def finalize_node(state: AgentState) -> Dict[str, Any]:
    """Produce final answer."""
    print(f"\n   ✅ FINALIZING answer...")
    
    draft = state.get("draft_answer", "No answer generated.")
    score = state.get("judge_score", 0)
    
    print(f"      Final score: {score}/10")
    print(f"      Correction attempts: {state.get('correction_attempts', 0)}")
    
    return {
        "final_answer": draft,
        "node_history": state.get("node_history", []) + ["finalize"]
    }

print("✅ All nodes defined!")

# ============================================
# Part 5: Conditional Logic
# ============================================
print("\n🔀 PART 5: CONDITIONAL LOGIC")
print("-" * 40)

def should_correct(state: AgentState) -> str:
    """Decide whether to correct or finalize."""
    score = state.get("judge_score", 0)
    attempts = state.get("correction_attempts", 0)
    
    print(f"\n   🔀 ROUTING: Score={score}/10, Attempts={attempts}/2")
    
    if score >= 7.0:
        print("      ✅ Score >= 7 → FINALIZE")
        return "finalize"
    elif attempts >= 2:
        print("      ⚠️ Max attempts reached → FINALIZE")
        return "finalize"
    else:
        print(f"      🔄 Score < 7 → CORRECT (attempt {attempts + 1})")
        return "correct"

print("✅ should_correct() defined!")

# ============================================
# Part 6: Build the Graph
# ============================================
print("\n🏗️ PART 6: BUILDING THE LANGGRAPH")
print("-" * 40)

# Create the graph
graph = StateGraph(AgentState)

# Add all nodes
graph.add_node("plan", plan_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_node("judge", judge_node)
graph.add_node("correct", correct_node)
graph.add_node("finalize", finalize_node)

# Set entry point
graph.set_entry_point("plan")

# Add edges
graph.add_edge("plan", "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "judge")

# Add conditional edge from judge
graph.add_conditional_edges(
    "judge",
    should_correct,
    {
        "correct": "correct",
        "finalize": "finalize"
    }
)

# Add edges back to generate for correction loop
graph.add_edge("correct", "generate")
graph.add_edge("finalize", END)

# Compile
app = graph.compile()

print("✅ Graph built and compiled!")
print("\n📊 Graph Structure:")
print("   plan → retrieve → generate → judge → (conditional)")
print("                                    ↓")
print("                              ┌─────┴─────┐")
print("                              ↓           ↓")
print("                          correct    finalize")
print("                              ↓           ↓")
print("                          generate       END")
print("                              ↓")
print("                           (loops back)")

# ============================================
# Part 7: Test the Complete Agent
# ============================================
print("\n🧪 PART 7: TESTING THE COMPLETE AGENT")
print("-" * 40)

test_questions = [
    "Which magazine did The Doors appear on the cover of in 1967?",
    "What is the population of the capital of France?",
]

print("\n📌 Running agent on test questions...")
print("-" * 40)

for i, question in enumerate(test_questions, 1):
    print(f"\n📌 Test {i}: {question}")
    print("=" * 50)
    
    # Create initial state
    state = create_initial_state(question)
    print(f"\n⏱️ Start time: {state['start_time']}")
    
    try:
        # Run the agent
        result = app.invoke(state)
        
        print("\n" + "=" * 50)
        print("📊 FINAL RESULTS:")
        print("-" * 40)
        print(f"Question: {result['question']}")
        print(f"\nFinal Answer:\n{result['final_answer']}")
        print(f"\nJudge Score: {result['judge_score']}/10")
        print(f"Correction Attempts: {result['correction_attempts']}")
        print(f"Sources: {', '.join(result['sources'][:3])}")
        print(f"\nNode History: {' → '.join(result['node_history'])}")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error running agent: {e}")

# ============================================
# Part 8: Summary
# ============================================
print("\n📊 DAY 13 SUMMARY")
print("-" * 40)
print("""
✅ Wired all nodes into a complete graph
✅ Added conditional edges for self-correction
✅ Built the full agent loop
✅ Tested the complete agent

🎯 The Complete Flow:
   1. PLAN: Break question into sub-questions
   2. RETRIEVE: Get documents for each sub-question
   3. GENERATE: Write answer with Chain-of-Thought
   4. JUDGE: Evaluate answer quality
   5. If score >= 7: FINALIZE
   6. If score < 7: CORRECT → GENERATE → JUDGE (loop)
   7. FINALIZE: Return final answer

📋 Key Concepts:
   1. State flows through all nodes
   2. Conditional edges enable self-correction
   3. The loop improves answers iteratively
   4. Max 2 correction attempts

🚀 Next: Test and evaluate the full agent!
""")

print("\n🚀 Ready for Day 14! (Test + Commit)")
print("=" * 60)