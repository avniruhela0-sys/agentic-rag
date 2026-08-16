"""
Day 19: Test the Full Agent with CoT & Self-Correction
Comprehensive testing of the complete agent against baseline.
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
import re

print("=" * 60)
print("🧪 DAY 19: TESTING THE FULL AGENT (CoT + Self-Correction)")
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
    reasoning_steps: List[str]
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
        reasoning_steps=[],
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
# Part 2: Initialize Components
# ============================================
print("\n🔧 PART 2: INITIALIZING COMPONENTS")
print("-" * 40)

try:
    llm = ChatOllama(model="mistral", temperature=0)
    print("✅ LLM initialized! (Ollama - mistral)")
except Exception as e:
    print(f"❌ Error initializing LLM: {e}")
    exit(1)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vector_store_paths = ['chroma_db_hf', 'chroma_db', 'chroma_db_50']
vectorstore = None

for path in vector_store_paths:
    if os.path.exists(path):
        try:
            vectorstore = Chroma(persist_directory=path, embedding_function=embeddings)
            print(f"✅ Loaded vector store from: {path}")
            break
        except Exception as e:
            print(f"   ❌ Could not load from {path}: {e}")

if vectorstore is None:
    print("❌ No vector store found!")
    exit(1)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("✅ Retriever created!")

# ============================================
# Part 3: Create Prompts
# ============================================
print("\n📝 PART 3: CREATING PROMPTS")
print("-" * 40)

PLANNER_PROMPT = """
Break this question into 1-3 focused sub-questions. Return one per line.

Question: {question}

Sub-questions:
"""
planner_prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)

COT_GENERATOR_PROMPT = """
Using the context, answer the question with step-by-step reasoning.

Context:
{context}

Question: {question}

Step-by-step reasoning:
Step 1:
Step 2:
Step 3:

Final Answer:
"""
cot_generator_prompt = ChatPromptTemplate.from_template(COT_GENERATOR_PROMPT)

JUDGE_PROMPT = """
You are a strict evaluator. Score the answer from 0-10.

Context:
{context}

Question:
{question}

Answer:
{answer}

Response Format:
SCORE: <number>
ISSUE: <hallucination/contradiction/irrelevant/incomplete/none>
REASON: <one sentence>
"""
judge_prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)

print("✅ Prompts created!")

# ============================================
# Part 4: Define All Nodes
# ============================================
print("\n🔧 PART 4: DEFINING NODES")
print("-" * 40)

def plan_node(state: AgentState) -> Dict[str, Any]:
    print(f"   📝 Planning...")
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
        return {
            "sub_questions": sub_questions,
            "node_history": state.get("node_history", []) + ["plan"]
        }
    except Exception as e:
        return {
            "sub_questions": [state["question"]],
            "node_history": state.get("node_history", []) + ["plan"]
        }

def retrieve_node(state: AgentState) -> Dict[str, Any]:
    sub_questions = state.get("sub_questions", [state["question"]])
    all_docs, all_sources = [], []
    for query in sub_questions:
        try:
            docs = retriever.invoke(query)
            for doc in docs:
                all_docs.append(doc.page_content)
                all_sources.append(doc.metadata.get('title', 'Unknown'))
        except Exception as e:
            pass
    seen = set()
    unique_docs, unique_sources = [], []
    for doc, source in zip(all_docs, all_sources):
        if doc not in seen:
            seen.add(doc)
            unique_docs.append(doc)
            unique_sources.append(source)
    return {
        "retrieved_docs": unique_docs,
        "sources": unique_sources,
        "node_history": state.get("node_history", []) + ["retrieve"]
    }

def generate_node(state: AgentState) -> Dict[str, Any]:
    print(f"   🧠 Generating with CoT...")
    context = "\n\n".join(state.get("retrieved_docs", []))[:4000]
    if not context:
        return {
            "draft_answer": "I don't have enough information.",
            "reasoning_steps": ["No context available"],
            "node_history": state.get("node_history", []) + ["generate"]
        }
    try:
        response = (cot_generator_prompt | llm).invoke({
            "context": context,
            "question": state["question"]
        })
        full_answer = response.content
        reasoning_steps = []
        for line in full_answer.split("\n"):
            line = line.strip()
            if line.startswith("Step"):
                reasoning_steps.append(line)
        if not reasoning_steps:
            reasoning_steps = [full_answer[:100]]
        return {
            "draft_answer": full_answer,
            "reasoning_steps": reasoning_steps,
            "node_history": state.get("node_history", []) + ["generate"]
        }
    except Exception as e:
        return {
            "draft_answer": f"Error: {e}",
            "reasoning_steps": [f"Error: {e}"],
            "node_history": state.get("node_history", []) + ["generate"]
        }

def parse_judge_response(response: str) -> Dict[str, Any]:
    result = {"score": 5.0, "issue": "none", "reason": "Could not parse"}
    for line in response.strip().split("\n"):
        line = line.strip()
        if "SCORE:" in line.upper():
            try:
                score_part = line.split("SCORE:")[1].strip()
                if "/" in score_part:
                    score_part = score_part.split("/")[0]
                result["score"] = float(score_part)
            except:
                result["score"] = 5.0
        elif "ISSUE:" in line.upper():
            issue = line.split("ISSUE:")[1].strip().lower()
            valid_issues = ["hallucination", "contradiction", "irrelevant", "incomplete", "unsupported", "none"]
            result["issue"] = issue if issue in valid_issues else "none"
        elif "REASON:" in line.upper():
            result["reason"] = line.split("REASON:")[1].strip()
    result["score"] = max(0, min(10, result["score"]))
    return result

def judge_node(state: AgentState) -> Dict[str, Any]:
    print(f"   ⚖️ Judging...")
    answer = state.get("draft_answer", "")
    question = state.get("question", "")
    context = "\n\n".join(state.get("retrieved_docs", []))[:3000]
    if not answer or answer.startswith("Error"):
        return {
            "judge_score": 0.0,
            "judge_feedback": "Invalid answer",
            "judge_issue": "invalid",
            "node_history": state.get("node_history", []) + ["judge"]
        }
    try:
        response = (judge_prompt | llm).invoke({
            "context": context if context else "No context provided.",
            "question": question,
            "answer": answer
        })
        parsed = parse_judge_response(response.content)
        return {
            "judge_score": parsed["score"],
            "judge_feedback": parsed["reason"],
            "judge_issue": parsed["issue"],
            "node_history": state.get("node_history", []) + ["judge"]
        }
    except Exception as e:
        return {
            "judge_score": 5.0,
            "judge_feedback": f"Error: {e}",
            "judge_issue": "none",
            "node_history": state.get("node_history", []) + ["judge"]
        }

def correct_node(state: AgentState) -> Dict[str, Any]:
    attempts = state.get("correction_attempts", 0) + 1
    issue = state.get("judge_issue", "none")
    feedback = state.get("judge_feedback", "")
    print(f"   🔧 Correcting (attempt {attempts})...")
    question = state.get("question", "")
    refined_query = f"{question} (focus on: {feedback})" if feedback else question
    try:
        docs = retriever.invoke(refined_query)
        new_docs = state.get("retrieved_docs", []) + [doc.page_content for doc in docs]
        history = state.get("correction_history", [])
        history.append({
            "attempt": attempts,
            "issue": issue,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        })
        return {
            "retrieved_docs": new_docs,
            "correction_attempts": attempts,
            "correction_history": history,
            "node_history": state.get("node_history", []) + ["correct"]
        }
    except Exception as e:
        return {
            "correction_attempts": attempts,
            "node_history": state.get("node_history", []) + ["correct"]
        }

def finalize_node(state: AgentState) -> Dict[str, Any]:
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
# Part 5: Build the Graph
# ============================================
print("\n🏗️ PART 5: BUILDING THE GRAPH")
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
graph.add_conditional_edges("judge", should_correct, {"correct": "correct", "finalize": "finalize"})
graph.add_edge("correct", "generate")
graph.add_edge("finalize", END)

app = graph.compile()
print("✅ Graph compiled!")

# ============================================
# Part 6: Define Baseline RAG
# ============================================
print("\n📊 PART 6: BASELINE RAG")
print("-" * 40)

BASELINE_PROMPT = """
Answer the question using ONLY the context below.

Context:
{context}

Question: {question}

Answer:
"""
baseline_prompt = ChatPromptTemplate.from_template(BASELINE_PROMPT)

def simple_rag(question: str) -> str:
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
# Part 7: Scoring Function
# ============================================
print("\n🎯 PART 7: SCORING FUNCTION")
print("-" * 40)

def is_correct(expected: str, generated: str) -> bool:
    expected_lower = expected.lower().strip()
    generated_lower = generated.lower().strip()
    if expected_lower in generated_lower or generated_lower in expected_lower:
        return True
    words = expected_lower.split()
    if len(words) <= 3:
        matches = sum(1 for w in words if w in generated_lower)
        if matches >= len(words) * 0.5:
            return True
    return False

print("✅ Scoring function defined!")

# ============================================
# Part 8: Test on 10 Questions
# ============================================
print("\n🧪 PART 8: TESTING ON 10 QUESTIONS")
print("-" * 40)

# Load questions
try:
    with open('hotpot_sample.json', 'r') as f:
        all_questions = json.load(f)
    test_questions = all_questions[:10]
    print(f"✅ Loaded {len(test_questions)} questions")
except Exception as e:
    print(f"❌ Error loading questions: {e}")
    exit(1)

agent_results = []
baseline_results = []
total_time = 0

print("\n📌 Running tests...")
print("-" * 60)

for i, q in enumerate(test_questions, 1):
    question = q['question']
    expected = q['answer']
    
    print(f"\n📌 Question {i}/{len(test_questions)}:")
    print(f"   Q: {question[:60]}...")
    print(f"   Expected: {expected}")
    
    # Run Agentic RAG
    print(f"\n   🤖 Running Agentic RAG (CoT + Self-Correction)...")
    start = time.time()
    try:
        state = create_initial_state(question)
        result = app.invoke(state)
        agent_answer = result['final_answer']
        agent_score = result.get('judge_score', 0)
        agent_attempts = result.get('correction_attempts', 0)
        agent_reasoning = len(result.get('reasoning_steps', []))
        agent_correct = is_correct(expected, agent_answer)
    except Exception as e:
        agent_answer = f"Error: {e}"
        agent_score = 0
        agent_attempts = 0
        agent_reasoning = 0
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
        "reasoning_steps": agent_reasoning,
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
    print(f"      Agentic: {'✅' if agent_correct else '❌'} | Score: {agent_score:.1f}/10 | Attempts: {agent_attempts} | Steps: {agent_reasoning} | Time: {agent_time:.2f}s")
    print(f"      Baseline: {'✅' if baseline_correct else '❌'}")

# ============================================
# Part 9: Calculate Results
# ============================================
print("\n📊 PART 9: RESULTS SUMMARY")
print("-" * 40)

agent_correct = sum(1 for r in agent_results if r['correct'])
baseline_correct = sum(1 for r in baseline_results if r['correct'])

agent_acc = (agent_correct / len(agent_results)) * 100
baseline_acc = (baseline_correct / len(baseline_results)) * 100
improvement = agent_acc - baseline_acc

avg_agent_score = sum(r['score'] for r in agent_results) / len(agent_results)
avg_agent_time = total_time / len(agent_results)
avg_attempts = sum(r['attempts'] for r in agent_results) / len(agent_results)
avg_steps = sum(r['reasoning_steps'] for r in agent_results) / len(agent_results)

print(f"""
📊 COMPARISON TABLE:
┌─────────────────────────┬────────────┬────────────┬────────────┐
│ Metric                  │ Baseline   │ Agentic    │ Change     │
├─────────────────────────┼────────────┼────────────┼────────────┤
│ Correct Answers         │ {baseline_correct}/10     │ {agent_correct}/10     │ +{agent_correct - baseline_correct}    │
│ Accuracy                │ {baseline_acc:.1f}%       │ {agent_acc:.1f}%       │ +{improvement:.1f}%    │
│ Avg Judge Score         │ N/A        │ {avg_agent_score:.1f}/10   │ N/A        │
│ Avg Correction Attempts │ N/A        │ {avg_attempts:.1f}         │ N/A        │
│ Avg Reasoning Steps     │ N/A        │ {avg_steps:.1f}            │ N/A        │
│ Avg Time                │ ~2s        │ {avg_agent_time:.2f}s      │ +{avg_agent_time - 2:.2f}s │
└─────────────────────────┴────────────┴────────────┴────────────┘
""")

# ============================================
# Part 10: Save Results
# ============================================
print("\n💾 PART 10: SAVING RESULTS")
print("-" * 40)

week3_results = {
    "timestamp": datetime.now().isoformat(),
    "total_questions": len(test_questions),
    "method": "Chain-of-Thought + Self-Correction",
    "baseline": {
        "correct": baseline_correct,
        "accuracy": baseline_acc
    },
    "agentic": {
        "correct": agent_correct,
        "accuracy": agent_acc,
        "avg_score": avg_agent_score,
        "avg_attempts": avg_attempts,
        "avg_steps": avg_steps,
        "avg_time": avg_agent_time
    },
    "improvement": improvement,
    "details": {
        "agent_results": agent_results,
        "baseline_results": baseline_results
    }
}

with open('week3_results.json', 'w') as f:
    json.dump(week3_results, f, indent=2)

print(f"✅ Saved results to week3_results.json")

# ============================================
# Part 11: Summary
# ============================================
print("\n📊 DAY 19 SUMMARY")
print("-" * 40)
print(f"""
✅ Tested Agentic RAG with CoT on {len(test_questions)} questions
✅ Compared with Baseline RAG
✅ Baseline Accuracy: {baseline_acc:.1f}%
✅ Agentic Accuracy: {agent_acc:.1f}%
✅ Improvement: +{improvement:.1f}%

🎯 Week 3 Progress:
   - Added LLM-as-Judge (Day 15) ✅
   - Added Self-Correction (Day 16) ✅
   - Wired Full Loop (Day 17) ✅
   - Added Chain-of-Thought (Day 18) ✅
   - Completed Testing (Day 19) ✅

📊 Key Metrics:
   - Average Judge Score: {avg_agent_score:.1f}/10
   - Average Correction Attempts: {avg_attempts:.1f}
   - Average Reasoning Steps: {avg_steps:.1f}

🚀 Next: Day 20 - Analyze and Visualize Results!
""")

print("\n🚀 Ready for Day 20! (Analyze Results)")
print("=" * 60)