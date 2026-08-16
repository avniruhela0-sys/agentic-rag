"""
Day 17: Wire the Full Self-Correcting Loop
Complete LangGraph agent with plan-retrieve-generate-judge-correct-finalize
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
print("🔄 DAY 17: FULL SELF-CORRECTING AGENT")
print("=" * 60)

# ============================================
# Part 1: Define Agent State
# ============================================
print("\n📦 PART 1: DEFINING AGENT STATE")
print("-" * 40)

class AgentState(TypedDict):
    """The complete state of our agentic RAG system"""
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
# Part 2: Initialize Components
# ============================================
print("\n🔧 PART 2: INITIALIZING COMPONENTS")
print("-" * 40)

# Initialize LLM
try:
    llm = ChatOllama(
        model="mistral",
        temperature=0,
    )
    print("✅ LLM initialized! (Ollama - mistral)")
except Exception as e:
    print(f"❌ Error initializing LLM: {e}")
    print("💡 Make sure Ollama is running: ollama serve")
    exit(1)

# Load vector store
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
    print("💡 Please run Day 4 first (vector_store_hf.py)")
    exit(1)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("✅ Retriever created!")

# ============================================
# Part 3: Create All Prompts
# ============================================
print("\n📝 PART 3: CREATING PROMPTS")
print("-" * 40)

# 1. Planner Prompt
PLANNER_PROMPT = """
Break this question into 1-3 focused sub-questions. Return one per line.

Question: {question}

Sub-questions:
"""
planner_prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)

# 2. Generator Prompt
GENERATOR_PROMPT = """
Using the context, answer the question with step-by-step reasoning.

Context:
{context}

Question: {question}

Step-by-step reasoning:
Final Answer:
"""
generator_prompt = ChatPromptTemplate.from_template(GENERATOR_PROMPT)

# 3. Judge Prompt
JUDGE_PROMPT = """
You are a strict evaluator. Score the answer from 0-10.

Scoring:
- 9-10: Excellent, completely correct
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

Answer:
{answer}

Response Format:
SCORE: <number>
ISSUE: <issue_type>
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
    print(f"\n   🔍 RETRIEVING for {len(sub_questions)} query(ies)...")
    
    all_docs = []
    all_sources = []
    
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
    
    # Remove duplicates
    seen = set()
    unique_docs = []
    unique_sources = []
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

# Node 3: Generator
def generate_node(state: AgentState) -> Dict[str, Any]:
    """Generate answer with reasoning."""
    print(f"\n   ✍️ GENERATING answer...")
    
    context = "\n\n".join(state.get("retrieved_docs", []))[:4000]
    
    if not context:
        print("      ⚠️ No context available")
        return {
            "draft_answer": "I don't have enough information to answer this question.",
            "node_history": state.get("node_history", []) + ["generate"]
        }
    
    try:
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

# Node 4: Judge
def parse_judge_response(response: str) -> Dict[str, Any]:
    """Parse the judge's response."""
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
    """Evaluate answer quality."""
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

# Node 5: Corrector
def correct_node(state: AgentState) -> Dict[str, Any]:
    """Correct the answer based on judge feedback."""
    attempts = state.get("correction_attempts", 0) + 1
    issue = state.get("judge_issue", "none")
    feedback = state.get("judge_feedback", "")
    
    print(f"\n   🔧 CORRECTING (attempt {attempts}):")
    print(f"      Issue: {issue}")
    print(f"      Feedback: {feedback}")
    
    question = state.get("question", "")
    
    # Different strategies based on issue
    if issue == "hallucination":
        refined_query = f"{question} (accurate facts only)"
        print(f"      Strategy: Re-retrieve for accurate facts")
    elif issue == "contradiction":
        refined_query = f"{question} (consistent information only)"
        print(f"      Strategy: Re-retrieve for consistent info")
    elif issue == "irrelevant":
        refined_query = f"{question} (directly answer the question)"
        print(f"      Strategy: Re-retrieve for relevant info")
    elif issue == "incomplete":
        refined_query = f"{question} (complete answer with all details)"
        print(f"      Strategy: Re-retrieve for complete info")
    else:
        refined_query = f"{question} (focus on: {feedback})" if feedback else question
        print(f"      Strategy: General refinement")
    
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
        print(f"      Total: {len(new_docs)} documents")
        
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
    print(f"      Final score: {state.get('judge_score', 0)}/10")
    print(f"      Correction attempts: {state.get('correction_attempts', 0)}")
    
    return {
        "final_answer": state.get("draft_answer", "No answer generated."),
        "node_history": state.get("node_history", []) + ["finalize"]
    }

# Conditional Logic
def should_correct(state: AgentState) -> str:
    """Decide whether to correct or finalize."""
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

print("✅ All nodes defined!")

# ============================================
# Part 5: Build the Graph
# ============================================
print("\n🏗️ PART 5: BUILDING THE LANGGRAPH")
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
# Part 6: Test the Complete Agent
# ============================================
print("\n🧪 PART 6: TESTING THE COMPLETE AGENT")
print("-" * 40)

test_questions = [
    "What is the capital of France?",
    "Which magazine did The Doors appear on the cover of in 1967?",
    "What is the population of the capital of France?",
]

print("\n📌 Testing agent on 3 questions...")
print("-" * 60)

for i, question in enumerate(test_questions, 1):
    print(f"\n{'='*60}")
    print(f"📌 Test {i}: {question}")
    print('='*60)
    
    start_time = time.time()
    
    # Create initial state
    state = create_initial_state(question)
    print(f"\n⏱️ Start time: {state['start_time']}")
    
    try:
        # Run the agent
        result = app.invoke(state)
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("📊 FINAL RESULTS:")
        print("-" * 40)
        print(f"Question: {result['question']}")
        print(f"\nFinal Answer:\n{result['final_answer'][:300]}...")
        print(f"\nJudge Score: {result.get('judge_score', 0)}/10")
        print(f"Correction Attempts: {result.get('correction_attempts', 0)}")
        print(f"Issue: {result.get('judge_issue', 'none')}")
        print(f"Time: {elapsed:.2f}s")
        print(f"\nNode History: {' → '.join(result.get('node_history', []))}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error running agent: {e}")

# ============================================
# Part 7: Summary
# ============================================
print("\n📊 DAY 17 SUMMARY")
print("-" * 40)
print("""
✅ Wired all nodes into a complete graph
✅ Added conditional edges for self-correction
✅ Built the full agent loop
✅ Tested the complete agent on 3 questions

🎯 The Complete Flow:
   1. PLAN: Break question into sub-questions
   2. RETRIEVE: Get documents for each sub-question
   3. GENERATE: Write answer with Chain-of-Thought
   4. JUDGE: Evaluate answer quality (0-10)
   5. If score >= 7: FINALIZE
   6. If score < 7 and attempts < 2: CORRECT → GENERATE → JUDGE
   7. If attempts >= 2: FINALIZE (max attempts reached)

📋 Key Features:
   - Self-evaluation (LLM-as-Judge)
   - Self-correction (iterative improvement)
   - Multi-step planning (sub-questions)
   - Chain-of-Thought reasoning
   - Source tracking
   - Max 2 correction attempts

🚀 Next: Day 18 - Add Chain-of-Thought Reasoning!
""")

print("\n🚀 Ready for Day 18! (Chain-of-Thought)")
print("=" * 60)