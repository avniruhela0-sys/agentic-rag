"""
Day 14: Test the Complete Agent + Commit Week 2
Testing the full agent on 10 questions and comparing with baseline.
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
import time

print("=" * 60)
print("🧪 DAY 14: TESTING THE COMPLETE AGENT")
print("=" * 60)

# ============================================
# Part 1: Define Agent State
# ============================================
print("\n📦 PART 1: DEFINING AGENT STATE")
print("-" * 40)

class AgentState(TypedDict):
    question: str
    sub_questions: List[str]
    retrieved_docs: List[str]
    draft_answer: str
    judge_score: float
    judge_feedback: str
    judge_issue: str
    correction_attempts: int
    correction_history: List[Dict[str, Any]]
    final_answer: str
    sources: List[str]
    start_time: str
    node_history: List[str]

def create_initial_state(question: str) -> AgentState:
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
# Part 2: Load Components
# ============================================
print("\n🔧 PART 2: LOADING COMPONENTS")
print("-" * 40)

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Load vector store
vector_store_paths = ['chroma_db_hf', 'chroma_db', 'chroma_db_50']
vectorstore = None

for path in vector_store_paths:
    if os.path.exists(path):
        try:
            vectorstore = Chroma(
                persist_directory=path,
                embedding_function=embeddings
            )
            print(f"✅ Loaded vector store from: {path}")
            break
        except Exception as e:
            print(f"   ❌ Could not load from {path}: {e}")

if vectorstore is None:
    print("❌ No vector store found!")
    exit(1)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("✅ Retriever created!")

# Initialize LLM
try:
    llm = ChatOllama(model="mistral", temperature=0)
    print("✅ LLM initialized!")
except Exception as e:
    print(f"❌ Error initializing LLM: {e}")
    exit(1)

# ============================================
# Part 3: Define Prompts and Nodes
# ============================================
print("\n📝 PART 3: DEFINING NODES")
print("-" * 40)

# Prompts
PLANNER_PROMPT = """
Break this question into 1-3 focused sub-questions. Return one per line.

Question: {question}

Sub-questions:
"""
planner_prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)

GENERATOR_PROMPT = """
Using the context, answer the question with step-by-step reasoning.

Context:
{context}

Question: {question}

Step-by-step reasoning:
Final Answer:
"""
generator_prompt = ChatPromptTemplate.from_template(GENERATOR_PROMPT)

JUDGE_PROMPT = """
Score the answer from 0-10. Also identify issues.

Context:
{context}

Question: {question}

Answer: {answer}

Respond:
SCORE: <number>
ISSUE: <hallucination/contradiction/irrelevant/none>
REASON: <one sentence>
"""
judge_prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)

# Nodes
def plan_node(state: AgentState) -> Dict[str, Any]:
    print(f"   📝 Planning: {state['question'][:40]}...")
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
        print(f"      ✅ {len(sub_questions)} sub-questions")
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

def retrieve_node(state: AgentState) -> Dict[str, Any]:
    sub_questions = state.get("sub_questions", [state["question"]])
    print(f"   🔍 Retrieving for {len(sub_questions)} queries...")
    all_docs, all_sources = [], []
    for query in sub_questions:
        try:
            docs = retriever.invoke(query)
            for doc in docs:
                all_docs.append(doc.page_content)
                all_sources.append(doc.metadata.get('title', 'Unknown'))
        except Exception as e:
            print(f"      ❌ Error: {e}")
    seen = set()
    unique_docs, unique_sources = [], []
    for doc, source in zip(all_docs, all_sources):
        if doc not in seen:
            seen.add(doc)
            unique_docs.append(doc)
            unique_sources.append(source)
    print(f"      ✅ {len(unique_docs)} unique documents")
    return {
        "retrieved_docs": unique_docs,
        "sources": unique_sources,
        "node_history": state.get("node_history", []) + ["retrieve"]
    }

def generate_node(state: AgentState) -> Dict[str, Any]:
    print(f"   ✍️ Generating answer...")
    if not state.get("retrieved_docs"):
        return {
            "draft_answer": "I don't have enough information.",
            "node_history": state.get("node_history", []) + ["generate"]
        }
    try:
        context = "\n\n".join(state["retrieved_docs"])[:4000]
        response = (generator_prompt | llm).invoke({
            "context": context,
            "question": state["question"]
        })
        print(f"      ✅ {len(response.content)} characters")
        return {
            "draft_answer": response.content,
            "node_history": state.get("node_history", []) + ["generate"]
        }
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return {
            "draft_answer": f"Error: {e}",
            "node_history": state.get("node_history", []) + ["generate"]
        }

def judge_node(state: AgentState) -> Dict[str, Any]:
    print(f"   ⚖️ Judging answer...")
    try:
        context = "\n\n".join(state.get("retrieved_docs", []))[:2000]
        response = (judge_prompt | llm).invoke({
            "context": context,
            "question": state["question"],
            "answer": state.get("draft_answer", "")
        })
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

def correct_node(state: AgentState) -> Dict[str, Any]:
    attempts = state.get("correction_attempts", 0) + 1
    print(f"   🔧 Correcting (attempt {attempts})...")
    refined_query = f"{state['question']} (focus on: {state.get('judge_feedback', '')})"
    try:
        docs = retriever.invoke(refined_query)
        new_docs = state.get("retrieved_docs", []) + [doc.page_content for doc in docs]
        history = state.get("correction_history", [])
        history.append({
            "attempt": attempts,
            "feedback": state.get("judge_feedback", ""),
            "timestamp": datetime.now().isoformat()
        })
        print(f"      Added {len(docs)} new documents")
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

def finalize_node(state: AgentState) -> Dict[str, Any]:
    print(f"   ✅ Finalizing answer...")
    return {
        "final_answer": state.get("draft_answer", "No answer generated."),
        "node_history": state.get("node_history", []) + ["finalize"]
    }

def should_correct(state: AgentState) -> str:
    score = state.get("judge_score", 0)
    attempts = state.get("correction_attempts", 0)
    if score >= 7.0 or attempts >= 2:
        return "finalize"
    return "correct"

print("✅ All nodes defined!")

# ============================================
# Part 4: Build the Graph
# ============================================
print("\n🏗️ PART 4: BUILDING THE GRAPH")
print("-" * 40)

graph = StateGraph(AgentState)
graph.add_node("plan", plan_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_node("judge", judge_node)
graph.add_node("correct", correct_node)
graph.add_node("finalize", finalize_node)

graph.set_entry_point("plan")
graph.add_edge("plan", "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "judge")

graph.add_conditional_edges(
    "judge",
    should_correct,
    {"correct": "correct", "finalize": "finalize"}
)

graph.add_edge("correct", "generate")
graph.add_edge("finalize", END)

app = graph.compile()
print("✅ Graph compiled!")

# ============================================
# Part 5: Define Baseline RAG for Comparison
# ============================================
print("\n📊 PART 5: BASELINE RAG FUNCTION")
print("-" * 40)

# Simple RAG for comparison
BASELINE_PROMPT = """
Answer the question using ONLY the context below.

Context:
{context}

Question: {question}

Answer:
"""
baseline_prompt = ChatPromptTemplate.from_template(BASELINE_PROMPT)

def simple_rag(question: str) -> str:
    """Simple RAG (no agent, no correction)."""
    try:
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        response = (baseline_prompt | llm).invoke({
            "context": context,
            "question": question
        })
        return response.content
    except Exception as e:
        return f"Error: {e}"

print("✅ Baseline RAG defined!")

# ============================================
# Part 6: Helper Function for Scoring
# ============================================
print("\n🔧 PART 6: SCORING FUNCTION")
print("-" * 40)

def is_correct(expected: str, generated: str) -> bool:
    """Check if generated answer matches expected."""
    expected_lower = expected.lower().strip()
    generated_lower = generated.lower().strip()
    
    # Check if expected is in generated or vice versa
    if expected_lower in generated_lower or generated_lower in expected_lower:
        return True
    
    # Check for simple variations
    words = expected_lower.split()
    if len(words) <= 3:
        # For short answers, check if most words are present
        matches = sum(1 for w in words if w in generated_lower)
        if matches >= len(words) * 0.5:
            return True
    
    return False

print("✅ Scoring function defined!")

# ============================================
# Part 7: Test on 10 Questions
# ============================================
print("\n🧪 PART 7: TESTING AGENT ON 10 QUESTIONS")
print("-" * 40)

# Load questions
try:
    with open('hotpot_sample.json', 'r') as f:
        all_questions = json.load(f)
    test_questions = all_questions[:10]  # Use first 10
    print(f"✅ Loaded {len(test_questions)} questions")
except Exception as e:
    print(f"❌ Error loading questions: {e}")
    exit(1)

# Results storage
agent_results = []
baseline_results = []
total_time = 0

print("\n📌 Running agent on 10 questions...")
print("-" * 60)

for i, q in enumerate(test_questions, 1):
    question = q['question']
    expected = q['answer']
    
    print(f"\n📌 Question {i}/{len(test_questions)}:")
    print(f"   Q: {question[:60]}...")
    print(f"   Expected: {expected}")
    
    # Run Agentic RAG
    print(f"\n   🤖 Running Agentic RAG...")
    start = time.time()
    try:
        state = create_initial_state(question)
        result = app.invoke(state)
        agent_answer = result['final_answer']
        agent_score = result['judge_score']
        agent_attempts = result['correction_attempts']
        agent_correct = is_correct(expected, agent_answer)
    except Exception as e:
        agent_answer = f"Error: {e}"
        agent_score = 0
        agent_attempts = 0
        agent_correct = False
    end = time.time()
    agent_time = end - start
    total_time += agent_time
    
    # Run Baseline RAG
    print(f"\n   📚 Running Baseline RAG...")
    baseline_answer = simple_rag(question)
    baseline_correct = is_correct(expected, baseline_answer)
    
    # Store results
    agent_results.append({
        "question": question,
        "expected": expected,
        "answer": agent_answer,
        "score": agent_score,
        "attempts": agent_attempts,
        "correct": agent_correct,
        "time": agent_time
    })
    
    baseline_results.append({
        "question": question,
        "expected": expected,
        "answer": baseline_answer,
        "correct": baseline_correct
    })
    
    print(f"\n   📊 Results:")
    print(f"      Agentic: {'✅' if agent_correct else '❌'} | Score: {agent_score}/10 | Attempts: {agent_attempts} | Time: {agent_time:.2f}s")
    print(f"      Baseline: {'✅' if baseline_correct else '❌'}")
    print(f"      Agent Answer: {agent_answer[:100]}...")

# ============================================
# Part 8: Calculate Results
# ============================================
print("\n📊 PART 8: RESULTS SUMMARY")
print("-" * 40)

agent_correct = sum(1 for r in agent_results if r['correct'])
baseline_correct = sum(1 for r in baseline_results if r['correct'])

agent_acc = (agent_correct / len(agent_results)) * 100
baseline_acc = (baseline_correct / len(baseline_results)) * 100
improvement = agent_acc - baseline_acc

print(f"""
📊 COMPARISON TABLE:
┌─────────────────┬──────────┬──────────┐
│ Metric          │ Baseline │ Agentic  │
├─────────────────┼──────────┼──────────┤
│ Correct Answers │ {baseline_correct}/10    │ {agent_correct}/10    │
│ Accuracy        │ {baseline_acc:.1f}%     │ {agent_acc:.1f}%     │
│ Improvement     │ -        │ +{improvement:.1f}%   │
│ Avg Time        │ ~2s      │ {total_time/len(agent_results):.2f}s │
└─────────────────┴──────────┴──────────┘
""")

# ============================================
# Part 9: Save Results
# ============================================
print("\n💾 PART 9: SAVING RESULTS")
print("-" * 40)

week2_results = {
    "timestamp": datetime.now().isoformat(),
    "total_questions": len(test_questions),
    "baseline": {
        "correct": baseline_correct,
        "accuracy": baseline_acc
    },
    "agentic": {
        "correct": agent_correct,
        "accuracy": agent_acc,
        "avg_score": sum(r['score'] for r in agent_results) / len(agent_results),
        "avg_attempts": sum(r['attempts'] for r in agent_results) / len(agent_results),
        "avg_time": total_time / len(agent_results)
    },
    "improvement": improvement,
    "details": {
        "agent_results": agent_results,
        "baseline_results": baseline_results
    }
}

with open('week2_results.json', 'w') as f:
    json.dump(week2_results, f, indent=2)

print(f"✅ Saved results to week2_results.json")

# ============================================
# Part 10: Summary
# ============================================
print("\n📊 DAY 14 SUMMARY")
print("-" * 40)
print(f"""
✅ Tested Agentic RAG on {len(test_questions)} questions
✅ Compared with Baseline RAG
✅ Baseline Accuracy: {baseline_acc:.1f}%
✅ Agentic Accuracy: {agent_acc:.1f}%
✅ Improvement: +{improvement:.1f}%

🎯 Week 2 Complete! You've built:
   1. LangGraph agent with plan-retrieve-generate-judge-correct loop
   2. Chain-of-Thought reasoning
   3. Self-correction mechanism
   4. Complete evaluation framework

📊 WEEK 2 FILES CREATED:
   - langgraph_basics.py
   - agent_state.py
   - planner_node.py
   - retriever_node.py
   - generator_node.py
   - agent_loop.py
   - test_agent.py
   - week2_results.json

🚀 Next: Week 3 - Add LLM-as-Judge + Self-Correction!
""")

print("\n🚀 Ready to commit Week 2 to GitHub!")
print("=" * 60)