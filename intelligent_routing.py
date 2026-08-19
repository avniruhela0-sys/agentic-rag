"""
Day 23: Route to Web Search Intelligently (FIXED)
Smart routing based on question type and retrieval quality.
"""

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tavily import TavilyClient
from typing import TypedDict, List, Dict, Any
from datetime import datetime
import json
import os
import time
import re

print("=" * 60)
print("🧠 DAY 23: INTELLIGENT ROUTING (FIXED)")
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
    used_web_search: bool
    routing_reason: str
    web_search_used_in_loop: bool  # NEW: Track if web search was used in this loop
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
        used_web_search=False,
        routing_reason="",
        web_search_used_in_loop=False,
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
    print("💡 Make sure Ollama is running: ollama serve")
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

try:
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        print("⚠️ TAVILY_API_KEY not found in .env!")
        exit(1)
    tavily = TavilyClient(api_key=tavily_api_key)
    print("✅ Tavily client initialized!")
except Exception as e:
    print(f"❌ Error initializing Tavily: {e}")
    exit(1)

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

print("✅ All prompts created!")

# ============================================
# Part 4: The Intelligent Router (FIXED)
# ============================================
print("\n🧠 PART 4: BUILDING THE INTELLIGENT ROUTER (FIXED)")
print("-" * 40)

def should_route_to_web_search(state: AgentState) -> str:
    """
    Intelligent routing decision - FIXED to prevent infinite loop.
    """
    print(f"\n   🧠 ROUTING DECISION:")
    
    # Get current state
    score = state.get("judge_score", 0)
    attempts = state.get("correction_attempts", 0)
    used_web_search = state.get("used_web_search", False)
    web_search_used_in_loop = state.get("web_search_used_in_loop", False)
    
    print(f"      Score: {score}/10")
    print(f"      Correction Attempts: {attempts}/2")
    print(f"      Web Search Already Used: {web_search_used_in_loop}")
    
    # RULE 1: If score is good, finalize
    if score >= 7.0:
        print(f"      ✅ Score >= 7 → FINALIZE")
        return "finalize"
    
    # RULE 2: If max attempts reached, finalize
    if attempts >= 2:
        print(f"      ⚠️ Max attempts reached → FINALIZE")
        return "finalize"
    
    # RULE 3: If we already used web search in this loop, try correction
    if web_search_used_in_loop:
        print(f"      🔄 Web search already used → CORRECT")
        return "correct"
    
    # RULE 4: Analyze question type
    question_lower = state["question"].lower()
    current_keywords = ['current', 'latest', 'recent', 'now', 'today', '2024', '2025', '2026', 'ceo', 'president']
    is_current = any(kw in question_lower for kw in current_keywords)
    
    # RULE 5: If current topic and attempts < 1, use web search
    if is_current and attempts < 1:
        print(f"      🌐 Current topic → WEB_SEARCH")
        return "web_search"
    
    # RULE 6: If retrieval quality is poor and attempts >= 1
    num_docs = len(state.get("retrieved_docs", []))
    if num_docs < 3 and attempts >= 1:
        print(f"      🌐 Poor retrieval ({num_docs} docs) → WEB_SEARCH")
        return "web_search"
    
    # RULE 7: Default - try correction
    print(f"      🔄 Default → CORRECT")
    return "correct"

print("✅ Intelligent router defined!")

# ============================================
# Part 5: Define All Nodes
# ============================================
print("\n🔧 PART 5: DEFINING NODES")
print("-" * 40)

def plan_node(state: AgentState) -> Dict[str, Any]:
    print(f"\n   📝 PLANNING...")
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
    print(f"\n   🔍 RETRIEVING...")
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
    print(f"      ✅ {len(unique_docs)} documents")
    return {
        "retrieved_docs": unique_docs,
        "sources": unique_sources,
        "node_history": state.get("node_history", []) + ["retrieve"]
    }

def web_search_node(state: AgentState) -> Dict[str, Any]:
    print(f"\n   🌐 WEB SEARCH...")
    try:
        response = tavily.search(
            query=state["question"],
            max_results=3,
            search_depth="basic"
        )
        web_docs, web_sources = [], []
        for result in response.get("results", []):
            content = result.get("content", "")
            title = result.get("title", "Web Source")
            if content:
                web_docs.append(f"Source: {title}\nContent: {content}")
                web_sources.append(title)
        print(f"      ✅ {len(web_docs)} web results")
        existing_docs = state.get("retrieved_docs", [])
        all_docs = existing_docs + web_docs
        all_sources = state.get("sources", []) + web_sources
        return {
            "retrieved_docs": all_docs,
            "sources": all_sources,
            "used_web_search": True,
            "web_search_used_in_loop": True,  # Mark that we used web search
            "node_history": state.get("node_history", []) + ["web_search"]
        }
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return {
            "node_history": state.get("node_history", []) + ["web_search"]
        }

def generate_node(state: AgentState) -> Dict[str, Any]:
    print(f"\n   🧠 GENERATING...")
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
    print(f"\n   ⚖️ JUDGING...")
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
    print(f"\n   🔧 CORRECTING (attempt {attempts})...")
    question = state.get("question", "")
    refined_query = f"{question} (focus on: {state.get('judge_feedback', '')})"
    try:
        docs = retriever.invoke(refined_query)
        new_docs = state.get("retrieved_docs", []) + [doc.page_content for doc in docs]
        history = state.get("correction_history", [])
        history.append({
            "attempt": attempts,
            "feedback": state.get("judge_feedback", ""),
            "timestamp": datetime.now().isoformat()
        })
        print(f"      Added {len(docs)} documents")
        return {
            "retrieved_docs": new_docs,
            "correction_attempts": attempts,
            "correction_history": history,
            "web_search_used_in_loop": False,  # Reset web search flag after correction
            "node_history": state.get("node_history", []) + ["correct"]
        }
    except Exception as e:
        return {
            "correction_attempts": attempts,
            "web_search_used_in_loop": False,
            "node_history": state.get("node_history", []) + ["correct"]
        }

def finalize_node(state: AgentState) -> Dict[str, Any]:
    print(f"\n   ✅ FINALIZING...")
    print(f"      Score: {state.get('judge_score', 0)}/10")
    print(f"      Web Search Used: {state.get('used_web_search', False)}")
    return {
        "final_answer": state.get("draft_answer", "No answer generated."),
        "node_history": state.get("node_history", []) + ["finalize"]
    }

print("✅ All nodes defined!")

# ============================================
# Part 6: Build the Graph
# ============================================
print("\n🏗️ PART 6: BUILDING THE GRAPH")
print("-" * 40)

graph = StateGraph(AgentState)
graph.add_node("plan", plan_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("web_search", web_search_node)
graph.add_node("generate", generate_node)
graph.add_node("judge", judge_node)
graph.add_node("correct", correct_node)
graph.add_node("finalize", finalize_node)

graph.set_entry_point("plan")
graph.add_edge("plan", "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("web_search", "generate")
graph.add_edge("generate", "judge")

graph.add_conditional_edges(
    "judge",
    should_route_to_web_search,
    {
        "correct": "correct",
        "web_search": "web_search",
        "finalize": "finalize"
    }
)

graph.add_edge("correct", "generate")
graph.add_edge("finalize", END)

app = graph.compile()
print("✅ Graph compiled!")

# ============================================
# Part 7: Test
# ============================================
print("\n🧪 PART 7: TESTING")
print("-" * 40)

test_questions = [
    "What is the capital of France?",
]

print("\n📌 Testing...")

for i, question in enumerate(test_questions, 1):
    print(f"\n{'='*60}")
    print(f"📌 Test {i}: {question}")
    print('='*60)
    
    state = create_initial_state(question)
    
    try:
        result = app.invoke(state)
        
        print("\n" + "=" * 60)
        print("📊 FINAL RESULTS:")
        print("-" * 40)
        print(f"Question: {result['question']}")
        print(f"\n📝 Final Answer:\n{result['final_answer'][:300]}...")
        print(f"\n⭐ Judge Score: {result.get('judge_score', 0)}/10")
        print(f"🔄 Correction Attempts: {result.get('correction_attempts', 0)}")
        print(f"🌐 Web Search Used: {result.get('used_web_search', False)}")
        print(f"\n🗺️ Node History: {' → '.join(result.get('node_history', []))}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n📊 DAY 23 SUMMARY")
print("-" * 40)
print("""
✅ Fixed infinite loop bug
✅ Added web_search_used_in_loop tracking
✅ Added correction attempt increment
✅ Smart routing rules applied

🚀 Ready for Day 24-25!
""")