"""
Day 18: Chain-of-Thought Reasoning
Enhancing the agent with transparent step-by-step reasoning.
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
print("🧠 DAY 18: CHAIN-OF-THOUGHT REASONING")
print("=" * 60)

# ============================================
# Part 1: Understanding Chain-of-Thought
# ============================================
print("\n📚 WHAT IS CHAIN-OF-THOUGHT?")
print("-" * 40)
print("""
Chain-of-Thought (CoT) is a technique where the model
shows its reasoning step by step.

Without CoT:
    Answer: "The capital of France is Paris."

With CoT:
    Step 1: France is a country in Western Europe.
    Step 2: The capital of France is Paris.
    Step 3: Paris has a population of about 2.1 million.
    Final Answer: The capital of France is Paris.

Why CoT Matters:
1. Transparency - Shows how the answer was reached
2. Accuracy - Forces step-by-step reasoning
3. Trust - Users can verify the logic
4. Debugging - Easier to find errors
5. Education - Helps users learn
""")

# ============================================
# Part 2: Define Agent State with CoT
# ============================================
print("\n📦 PART 2: DEFINING AGENT STATE")
print("-" * 40)

class AgentState(TypedDict):
    question: str
    sub_questions: List[str]
    retrieved_docs: List[str]
    draft_answer: str
    reasoning_steps: List[str]  # NEW: Store reasoning steps
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
        reasoning_steps=[],  # NEW
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

print("✅ AgentState defined with reasoning_steps!")

# ============================================
# Part 3: Initialize Components
# ============================================
print("\n🔧 PART 3: INITIALIZING COMPONENTS")
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
# Part 4: Create CoT Prompts
# ============================================
print("\n📝 PART 4: CREATING COT PROMPTS")
print("-" * 40)

# 1. Planner Prompt
PLANNER_PROMPT = """
Break this question into 1-3 focused sub-questions. Return one per line.

Question: {question}

Sub-questions:
"""
planner_prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)

# 2. CoT Generator Prompt
COT_GENERATOR_PROMPT = """
You are a research assistant. Use the context to answer the question.

IMPORTANT: Show your reasoning step by step.

Context:
{context}

Question: {question}

Step-by-step reasoning:
Step 1: [What does the context tell us about the topic?]
Step 2: [What specific information answers the question?]
Step 3: [Is there any additional relevant information?]
Step 4: [What is the final answer?]

Final Answer:
"""
cot_generator_prompt = ChatPromptTemplate.from_template(COT_GENERATOR_PROMPT)

# 3. Judge Prompt
JUDGE_PROMPT = """
You are a strict evaluator. Score the answer from 0-10.

Scoring:
- 9-10: Excellent, completely correct with good reasoning
- 7-8: Good, mostly correct
- 5-6: Mediocre, partially correct
- 0-4: Poor, wrong or hallucinated

Issue Types:
- hallucination: Claims NOT in context
- contradiction: Claims that CONTRADICT context
- irrelevant: Doesn't answer the question
- incomplete: Missing important information
- none: No issues

Context:
{context}

Question:
{question}

Answer with reasoning:
{answer}

Response Format:
SCORE: <number>
ISSUE: <issue_type>
REASON: <one sentence>
"""
judge_prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)

print("✅ CoT prompts created!")

# ============================================
# Part 5: Define All Nodes with CoT
# ============================================
print("\n🔧 PART 5: DEFINING NODES WITH COT")
print("-" * 40)

def plan_node(state: AgentState) -> Dict[str, Any]:
    print(f"\n   📝 PLANNING: {state['question'][:40]}...")
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
    print(f"\n   🔍 RETRIEVING for {len(sub_questions)} query(ies)...")
    all_docs, all_sources = [], []
    for i, query in enumerate(sub_questions, 1):
        print(f"      Query {i}: {query[:40]}...")
        try:
            docs = retriever.invoke(query)
            for doc in docs:
                all_docs.append(doc.page_content)
                all_sources.append(doc.metadata.get('title', 'Unknown'))
            print(f"         Got {len(docs)} chunks")
        except Exception as e:
            print(f"         ❌ Error: {e}")
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
    """Generate answer with Chain-of-Thought reasoning."""
    print(f"\n   🧠 GENERATING with Chain-of-Thought...")
    
    context = "\n\n".join(state.get("retrieved_docs", []))[:4000]
    
    if not context:
        print("      ⚠️ No context available")
        return {
            "draft_answer": "I don't have enough information to answer this question.",
            "reasoning_steps": ["No context available"],
            "node_history": state.get("node_history", []) + ["generate"]
        }
    
    try:
        response = (cot_generator_prompt | llm).invoke({
            "context": context,
            "question": state["question"]
        })
        
        # Extract reasoning steps from the response
        full_answer = response.content
        
        # Parse reasoning steps (if they exist)
        reasoning_steps = []
        lines = full_answer.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("Step"):
                reasoning_steps.append(line)
        
        # If no explicit steps, extract as one step
        if not reasoning_steps:
            reasoning_steps = [full_answer[:200]]
        
        print(f"      ✅ Generated with {len(reasoning_steps)} reasoning steps")
        for step in reasoning_steps[:3]:
            print(f"         {step[:60]}...")
        
        return {
            "draft_answer": full_answer,
            "reasoning_steps": reasoning_steps,
            "node_history": state.get("node_history", []) + ["generate"]
        }
    except Exception as e:
        print(f"      ❌ Error: {e}")
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
    print(f"\n   ⚖️ JUDGING answer...")
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
        print(f"      Score: {parsed['score']}/10")
        print(f"      Issue: {parsed['issue']}")
        print(f"      Reason: {parsed['reason']}")
        return {
            "judge_score": parsed["score"],
            "judge_feedback": parsed["reason"],
            "judge_issue": parsed["issue"],
            "node_history": state.get("node_history", []) + ["judge"]
        }
    except Exception as e:
        print(f"      ❌ Error: {e}")
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
    print(f"\n   🔧 CORRECTING (attempt {attempts}):")
    print(f"      Issue: {issue}")
    print(f"      Feedback: {feedback}")
    question = state.get("question", "")
    if issue == "hallucination":
        refined_query = f"{question} (accurate facts only)"
    elif issue == "contradiction":
        refined_query = f"{question} (consistent information only)"
    elif issue == "irrelevant":
        refined_query = f"{question} (directly answer the question)"
    elif issue == "incomplete":
        refined_query = f"{question} (complete answer with all details)"
    else:
        refined_query = f"{question} (focus on: {feedback})" if feedback else question
    try:
        docs = retriever.invoke(refined_query)
        new_docs = state.get("retrieved_docs", []) + [doc.page_content for doc in docs]
        history = state.get("correction_history", [])
        history.append({
            "attempt": attempts,
            "issue": issue,
            "feedback": feedback,
            "refined_query": refined_query,
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
    print(f"\n   ✅ FINALIZING answer...")
    print(f"      Final score: {state.get('judge_score', 0)}/10")
    print(f"      Correction attempts: {state.get('correction_attempts', 0)}")
    print(f"      Reasoning steps: {len(state.get('reasoning_steps', []))}")
    return {
        "final_answer": state.get("draft_answer", "No answer generated."),
        "node_history": state.get("node_history", []) + ["finalize"]
    }

def should_correct(state: AgentState) -> str:
    score = state.get("judge_score", 0)
    attempts = state.get("correction_attempts", 0)
    issue = state.get("judge_issue", "none")
    print(f"\n   🔀 ROUTING DECISION:")
    print(f"      Score: {score}/10")
    print(f"      Attempts: {attempts}/2")
    print(f"      Issue: {issue}")
    if score >= 7.0:
        print(f"      ✅ Score >= 7 → FINALIZE")
        return "finalize"
    elif attempts >= 2:
        print(f"      ⚠️ Max attempts reached → FINALIZE")
        return "finalize"
    else:
        print(f"      🔄 Score < 7 → CORRECT (attempt {attempts + 1})")
        return "correct"

print("✅ All nodes with CoT defined!")

# ============================================
# Part 6: Build the Graph
# ============================================
print("\n🏗️ PART 6: BUILDING THE LANGGRAPH")
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

print("✅ Graph built and compiled!")
print("\n📊 Graph Structure:")
print("   plan → retrieve → generate (CoT) → judge → (conditional)")
print("                                    ↓")
print("                              ┌─────┴─────┐")
print("                              ↓           ↓")
print("                          correct    finalize")
print("                              ↓           ↓")
print("                          generate       END")
print("                              ↓")
print("                           (loops back)")

# ============================================
# Part 7: Test with CoT
# ============================================
print("\n🧪 PART 7: TESTING WITH CHAIN-OF-THOUGHT")
print("-" * 40)

test_questions = [
    "What is the capital of France?",
    "Which magazine did The Doors appear on the cover of in 1967?",
    "What is the population of the capital of France?",
]

print("\n📌 Testing agent with CoT on 3 questions...")
print("-" * 60)

for i, question in enumerate(test_questions, 1):
    print(f"\n{'='*60}")
    print(f"📌 Test {i}: {question}")
    print('='*60)
    
    start_time = time.time()
    state = create_initial_state(question)
    print(f"\n⏱️ Start time: {state['start_time']}")
    
    try:
        result = app.invoke(state)
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("📊 FINAL RESULTS:")
        print("-" * 40)
        print(f"Question: {result['question']}")
        
        # Show reasoning steps
        reasoning_steps = result.get('reasoning_steps', [])
        if reasoning_steps:
            print(f"\n🧠 Reasoning Steps:")
            for step in reasoning_steps[:5]:
                print(f"   {step[:100]}...")
        
        print(f"\n📝 Final Answer:\n{result['final_answer'][:300]}...")
        print(f"\n⭐ Judge Score: {result.get('judge_score', 0)}/10")
        print(f"🔄 Correction Attempts: {result.get('correction_attempts', 0)}")
        print(f"📋 Issue: {result.get('judge_issue', 'none')}")
        print(f"⏱️ Time: {elapsed:.2f}s")
        print(f"\n🗺️ Node History: {' → '.join(result.get('node_history', []))}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error running agent: {e}")

# ============================================
# Part 8: CoT vs No-CoT Comparison
# ============================================
print("\n🔄 PART 8: CoT vs No-CoT COMPARISON")
print("-" * 40)

print("""
┌─────────────────────────────────────────────────────────────────────────┐
│                     WITHOUT CHAIN-OF-THOUGHT                           │
├─────────────────────────────────────────────────────────────────────────┤
│ Answer: "The capital of France is Paris."                              │
│                                                                         │
│ Problems:                                                               │
│ - No reasoning shown                                                   │
│ - Hard to trust                                                         │
│ - Can't verify the logic                                              │
│ - Errors are hidden                                                   │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                     WITH CHAIN-OF-THOUGHT                              │
├─────────────────────────────────────────────────────────────────────────┤
│ Step 1: France is a country in Western Europe.                        │
│ Step 2: The capital of France is Paris.                               │
│ Step 3: Paris has a population of about 2.1 million.                 │
│ Step 4: Paris is also known as the City of Light.                    │
│                                                                         │
│ Final Answer: The capital of France is Paris.                         │
│                                                                         │
│ Benefits:                                                              │
│ ✅ Shows reasoning                                                    │
│ ✅ Builds trust                                                       │
│ ✅ Easy to verify                                                     │
│ ✅ Easier to debug                                                    │
│ ✅ Educational value                                                  │
└─────────────────────────────────────────────────────────────────────────┘
""")

# ============================================
# Part 9: Summary
# ============================================
print("\n📊 DAY 18 SUMMARY")
print("-" * 40)
print("""
✅ Added Chain-of-Thought reasoning to the agent
✅ Created CoT generator prompt with step-by-step structure
✅ Added reasoning_steps to AgentState
✅ Tested CoT on 3 questions
✅ Compared CoT vs No-CoT

🎯 What CoT Adds:
   1. Transparent reasoning process
   2. Step-by-step logic
   3. Easier debugging
   4. Builds user trust
   5. Educational value

📋 Key Features:
   - Step 1: Understand the context
   - Step 2: Identify key information
   - Step 3: Connect to the question
   - Step 4: Formulate final answer

🚀 Next: Test the full agent with CoT and commit Week 3!
""")

print("\n🚀 Ready for Day 19-20! (Testing the Full Agent)")
print("=" * 60)