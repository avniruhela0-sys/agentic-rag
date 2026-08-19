"""
Day 22: Add Web Search (Tavily API)
Multi-tool agent with RAG + Web Search fallback.
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

print("=" * 60)
print("🌐 DAY 22: WEB SEARCH + MULTI-TOOL AGENT")
print("=" * 60)

# ============================================
# Part 1: Understanding Multi-Tool Agents
# ============================================
print("\n📚 WHAT IS A MULTI-TOOL AGENT?")
print("-" * 40)
print("""
A multi-tool agent can use different tools to answer questions:

1. RAG (Internal Knowledge):
   - Searches your document corpus
   - Fast, private, controlled
   - Good for known topics

2. Web Search (External Knowledge):
   - Searches the internet via Tavily API
   - Real-time, up-to-date
   - Good for new or unknown topics

The agent decides which tool to use based on:
- Internal retrieval quality
- Judge's feedback
- Correction attempts
""")

# ============================================
# Part 2: Define Agent State
# ============================================
print("\n📦 PART 2: DEFINING AGENT STATE")
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
    used_web_search: bool  # NEW: Track if web search was used
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
        used_web_search=False,  # NEW
        start_time=datetime.now().isoformat(),
        node_history=[]
    )

print("✅ AgentState defined with used_web_search!")

# ============================================
# Part 3: Initialize Components
# ============================================
print("\n🔧 PART 3: INITIALIZING COMPONENTS")
print("-" * 40)

# Initialize LLM
try:
    llm = ChatOllama(model="mistral", temperature=0)
    print("✅ LLM initialized! (Ollama - mistral)")
except Exception as e:
    print(f"❌ Error initializing LLM: {e}")
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

# Initialize Tavily (Web Search)
try:
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        print("⚠️ TAVILY_API_KEY not found in .env!")
        print("💡 Please add your Tavily API key to .env")
        print("   Get it from: https://tavily.com")
        exit(1)
    tavily = TavilyClient(api_key=tavily_api_key)
    print("✅ Tavily client initialized!")
except Exception as e:
    print(f"❌ Error initializing Tavily: {e}")
    exit(1)

# ============================================
# Part 4: Create All Prompts
# ============================================
print("\n📝 PART 4: CREATING PROMPTS")
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
# Part 5: Define All Nodes (Including Web Search)
# ============================================
print("\n🔧 PART 5: DEFINING NODES WITH WEB SEARCH")
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

# NEW: Web Search Node
def web_search_node(state: AgentState) -> Dict[str, Any]:
    """Search the web using Tavily API."""
    print(f"\n   🌐 SEARCHING WEB: {state['question'][:40]}...")
    
    try:
        # Search the web
        response = tavily.search(
            query=state["question"],
            max_results=3,
            search_depth="basic"  # "basic" or "advanced"
        )
        
        # Extract content from results
        web_docs = []
        web_sources = []
        
        for result in response.get("results", []):
            content = result.get("content", "")
            url = result.get("url", "")
            title = result.get("title", "Web Source")
            
            if content:
                web_docs.append(f"Source: {title}\nURL: {url}\nContent: {content}")
                web_sources.append(title)
        
        print(f"      ✅ Retrieved {len(web_docs)} web results")
        for i, source in enumerate(web_sources[:3], 1):
            print(f"         {i}. {source[:40]}...")
        
        # Add web results to retrieved docs
        existing_docs = state.get("retrieved_docs", [])
        all_docs = existing_docs + web_docs
        all_sources = state.get("sources", []) + web_sources
        
        return {
            "retrieved_docs": all_docs,
            "sources": all_sources,
            "used_web_search": True,  # Mark that web search was used
            "node_history": state.get("node_history", []) + ["web_search"]
        }
        
    except Exception as e:
        print(f"      ❌ Web search error: {e}")
        return {
            "used_web_search": False,
            "node_history": state.get("node_history", []) + ["web_search"]
        }

def generate_node(state: AgentState) -> Dict[str, Any]:
    print(f"\n   🧠 GENERATING with CoT...")
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
        print(f"      Added {len(docs)} new documents")
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
    print(f"\n   ✅ FINALIZING answer...")
    print(f"      Final score: {state.get('judge_score', 0)}/10")
    print(f"      Web search used: {state.get('used_web_search', False)}")
    return {
        "final_answer": state.get("draft_answer", "No answer generated."),
        "node_history": state.get("node_history", []) + ["finalize"]
    }

# Updated conditional logic with web search
def should_correct_or_web_search(state: AgentState) -> str:
    score = state.get("judge_score", 0)
    attempts = state.get("correction_attempts", 0)
    used_web = state.get("used_web_search", False)
    
    print(f"\n   🔀 ROUTING DECISION:")
    print(f"      Score: {score}/10")
    print(f"      Attempts: {attempts}/2")
    print(f"      Web Search Used: {used_web}")
    
    if score >= 7.0:
        print(f"      ✅ Score >= 7 → FINALIZE")
        return "finalize"
    elif attempts >= 2:
        print(f"      ⚠️ Max attempts reached → FINALIZE")
        return "finalize"
    elif attempts == 1 and not used_web:
        print(f"      🌐 After 1 correction, trying WEB SEARCH")
        return "web_search"
    else:
        print(f"      🔄 Score < 7 → CORRECT (attempt {attempts + 1})")
        return "correct"

print("✅ All nodes with web search defined!")

# ============================================
# Part 6: Build the Graph
# ============================================
print("\n🏗️ PART 6: BUILDING THE LANGGRAPH")
print("-" * 40)

graph = StateGraph(AgentState)
graph.add_node("plan", plan_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("web_search", web_search_node)  # NEW
graph.add_node("generate", generate_node)
graph.add_node("judge", judge_node)
graph.add_node("correct", correct_node)
graph.add_node("finalize", finalize_node)

graph.set_entry_point("plan")
graph.add_edge("plan", "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("web_search", "generate")  # NEW: Web search leads to generate
graph.add_edge("generate", "judge")

graph.add_conditional_edges(
    "judge",
    should_correct_or_web_search,
    {
        "correct": "correct",
        "web_search": "web_search",
        "finalize": "finalize"
    }
)

graph.add_edge("correct", "generate")
graph.add_edge("finalize", END)

app = graph.compile()

print("✅ Graph built and compiled!")
print("\n📊 Graph Structure:")
print("   plan → retrieve → generate → judge → (conditional)")
print("                                    ↓")
print("                              ┌─────┴─────┐")
print("                              ↓           ↓")
print("                          correct    web_search")
print("                              ↓           ↓")
print("                          generate       generate")
print("                              ↓           ↓")
print("                           (loops back)   ↓")
print("                                          ↓")
print("                                      finalize")
print("                                          ↓")
print("                                          END")

# ============================================
# Part 7: Test the Multi-Tool Agent
# ============================================
print("\n🧪 PART 7: TESTING MULTI-TOOL AGENT")
print("-" * 40)

test_questions = [
    "What is the capital of France?",
    "Who is the current CEO of Tesla?",
    "What is the population of Tokyo?",
]

print("\n📌 Testing multi-tool agent on 3 questions...")
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
            for step in reasoning_steps[:3]:
                print(f"   {step[:80]}...")
        
        print(f"\n📝 Final Answer:\n{result['final_answer'][:300]}...")
        print(f"\n⭐ Judge Score: {result.get('judge_score', 0)}/10")
        print(f"🔄 Correction Attempts: {result.get('correction_attempts', 0)}")
        print(f"🌐 Web Search Used: {result.get('used_web_search', False)}")
        print(f"⏱️ Time: {elapsed:.2f}s")
        print(f"\n🗺️ Node History: {' → '.join(result.get('node_history', []))}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error running agent: {e}")

# ============================================
# Part 8: Multi-Tool Benefits
# ============================================
print("\n🎯 PART 8: MULTI-TOOL BENEFITS")
print("-" * 40)

print("""
┌─────────────────────────────────────────────────────────────────────────┐
│                    RAG ONLY (Internal Knowledge)                       │
├─────────────────────────────────────────────────────────────────────────┤
│ ✅ Fast                                                               │
│ ✅ Private (data stays local)                                         │
│ ✅ Controlled                                                         │
│ ❌ Limited to what's in the corpus                                     │
│ ❌ Can't answer questions about new topics                            │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    RAG + WEB SEARCH (Multi-Tool)                       │
├─────────────────────────────────────────────────────────────────────────┤
│ ✅ RAG for known topics (fast)                                        │
│ ✅ Web Search for new topics (up-to-date)                             │
│ ✅ Best of both worlds                                                │
│ ✅ Fallback when internal knowledge fails                             │
│ ✅ Real-time information                                              │
└─────────────────────────────────────────────────────────────────────────┘
""")

# ============================================
# Part 9: Summary
# ============================================
print("\n📊 DAY 22 SUMMARY")
print("-" * 40)
print("""
✅ Added web search using Tavily API
✅ Created web_search_node
✅ Updated conditional routing (web search fallback)
✅ Built multi-tool agent (RAG + Web Search)
✅ Tested on 3 questions

🎯 What We Added:
   1. Tavily API integration for real-time web search
   2. Web search node that retrieves from the internet
   3. Smart routing: RAG → Correct → Web Search → Generate
   4. used_web_search tracking in state

📋 Multi-Tool Flow:
   1. Try RAG retrieval
   2. Generate answer
   3. Judge evaluates
   4. If score < 7 and attempts=1 → Web Search
   5. Generate with combined knowledge
   6. Re-judge and finalize

🚀 Next: Day 23 - Route to Web Search Intelligently!
""")

print("\n🚀 Ready for Day 23! (Intelligent Routing)")
print("=" * 60)