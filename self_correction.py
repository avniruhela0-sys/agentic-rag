"""
Day 16: Conditional Self-Correction Logic
Deciding when and how to correct answers.
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import TypedDict, List, Dict, Any, Literal
from datetime import datetime
import json
import os

print("=" * 60)
print("🔄 DAY 16: CONDITIONAL SELF-CORRECTION")
print("=" * 60)

# ============================================
# Part 1: Understanding Self-Correction
# ============================================
print("\n📚 WHAT IS SELF-CORRECTION?")
print("-" * 40)
print("""
Self-correction is the ability of an agent to:
1. Recognize when its answer is wrong or poor
2. Figure out what went wrong
3. Try again with better information
4. Iterate until the answer is good enough

The Correction Loop:
    Generate → Judge → Score
                        ↓
              ┌─────────┴─────────┐
              ↓                   ↓
        Score >= 7          Score < 7
              ↓                   ↓
          Finalize           Correct
                                  ↓
                              Re-generate
                                  ↓
                              Re-judge
                                  ↓
                         (repeat up to 2 times)

Why Self-Correction Matters:
1. Fixes hallucinations automatically
2. Improves answer quality iteratively
3. Reduces need for human intervention
4. Makes the agent more reliable
""")

# ============================================
# Part 2: Define Agent State
# ============================================
print("\n📦 PART 2: DEFINING AGENT STATE")
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
# Part 3: Initialize Components
# ============================================
print("\n🔧 PART 3: INITIALIZING COMPONENTS")
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
    exit(1)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("✅ Retriever created!")

# ============================================
# Part 4: Judge Node (from Day 15)
# ============================================
print("\n⚖️ PART 4: JUDGE NODE (from Day 15)")
print("-" * 40)

JUDGE_PROMPT = """
You are a strict, objective evaluator. Score the answer from 0-10.

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
    """Evaluate the answer quality."""
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

print("✅ Judge node loaded!")

# ============================================
# Part 5: Generator Node (simplified)
# ============================================
print("\n✍️ PART 5: GENERATOR NODE (simplified)")
print("-" * 40)

GENERATOR_PROMPT = """
Using the context, answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

generator_prompt = ChatPromptTemplate.from_template(GENERATOR_PROMPT)

def generate_node(state: AgentState) -> Dict[str, Any]:
    """Generate an answer."""
    print(f"\n   ✍️ GENERATING answer...")
    
    context = "\n\n".join(state.get("retrieved_docs", []))[:4000]
    
    if not context:
        return {
            "draft_answer": "I don't have enough information.",
            "node_history": state.get("node_history", []) + ["generate"]
        }
    
    try:
        response = (generator_prompt | llm).invoke({
            "context": context,
            "question": state["question"]
        })
        print(f"      ✅ Generated {len(response.content)} characters")
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

print("✅ Generator node loaded!")

# ============================================
# Part 6: The Correction Logic
# ============================================
print("\n🔧 PART 6: CORRECTION LOGIC")
print("-" * 40)

def should_correct(state: AgentState) -> str:
    """
    Decide whether to correct or finalize.
    This is the CONDITIONAL EDGE in LangGraph.
    """
    score = state.get("judge_score", 0)
    attempts = state.get("correction_attempts", 0)
    issue = state.get("judge_issue", "none")
    
    print(f"\n   🔀 ROUTING DECISION:")
    print(f"      Score: {score}/10")
    print(f"      Attempts: {attempts}/2")
    print(f"      Issue: {issue}")
    
    # Rule 1: If score is 7 or higher, finalize
    if score >= 7.0:
        print(f"      ✅ Score >= 7 → FINALIZE")
        return "finalize"
    
    # Rule 2: If we've tried 2 times, give up and finalize
    if attempts >= 2:
        print(f"      ⚠️ Max attempts reached → FINALIZE")
        return "finalize"
    
    # Rule 3: If answer is completely wrong, correct
    if score < 4.0:
        print(f"      ❌ Score < 4 → CORRECT (major issues)")
        return "correct"
    
    # Rule 4: For mediocre answers, correct if issue is serious
    if score < 7.0:
        serious_issues = ["hallucination", "contradiction", "irrelevant"]
        if issue in serious_issues:
            print(f"      ⚠️ Serious issue: {issue} → CORRECT")
            return "correct"
        else:
            print(f"      ⚠️ Score < 7 but minor issue → CORRECT (to improve)")
            return "correct"
    
    # Default: finalize
    print(f"      Default → FINALIZE")
    return "finalize"

print("✅ should_correct() defined!")

# ============================================
# Part 7: Correction Strategies
# ============================================
print("\n🔧 PART 7: CORRECTION STRATEGIES")
print("-" * 40)

def correct_node(state: AgentState) -> Dict[str, Any]:
    """
    Correct the answer based on the judge's feedback.
    Different strategies for different issues.
    """
    attempts = state.get("correction_attempts", 0) + 1
    issue = state.get("judge_issue", "none")
    feedback = state.get("judge_feedback", "")
    
    print(f"\n   🔧 CORRECTING (attempt {attempts}):")
    print(f"      Issue: {issue}")
    print(f"      Feedback: {feedback}")
    
    # Get the question safely
    question = state.get("question", "")
    
    # Strategy 1: Hallucination → Re-retrieve with focus on accuracy
    if issue == "hallucination":
        refined_query = f"{question} (accurate facts only)"
        print(f"      Strategy: Re-retrieve for accurate facts")
    
    # Strategy 2: Contradiction → Re-retrieve with focus on consistency
    elif issue == "contradiction":
        refined_query = f"{question} (consistent information only)"
        print(f"      Strategy: Re-retrieve for consistent info")
    
    # Strategy 3: Irrelevant → Re-retrieve with focus on relevance
    elif issue == "irrelevant":
        refined_query = f"{question} (directly answer the question)"
        print(f"      Strategy: Re-retrieve for relevant info")
    
    # Strategy 4: Incomplete → Re-retrieve with focus on missing info
    elif issue == "incomplete":
        refined_query = f"{question} (complete answer with all details)"
        print(f"      Strategy: Re-retrieve for complete info")
    
    # Strategy 5: Default → Re-retrieve with feedback
    else:
        refined_query = f"{question} (focus on: {feedback})" if feedback else question
        print(f"      Strategy: General refinement with feedback")
    
    # Execute retrieval
    try:
        docs = retriever.invoke(refined_query)
        new_docs = state.get("retrieved_docs", []) + [doc.page_content for doc in docs]
        
        # Record correction
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

print("✅ correct_node() defined!")

# ============================================
# Part 8: Simulate the Correction Loop
# ============================================
print("\n🔄 PART 8: SIMULATING THE CORRECTION LOOP")
print("-" * 40)

def simulate_agent_flow(question: str, initial_docs: List[str]) -> Dict[str, Any]:
    """
    Simulate the full agent flow with self-correction.
    """
    print(f"\n📌 Simulating: {question[:50]}...")
    print("-" * 50)
    
    # Create initial state
    state = create_initial_state(question)
    state["retrieved_docs"] = initial_docs[:4]
    state["sources"] = ["Doc1", "Doc2", "Doc3", "Doc4"]
    state["node_history"] = ["plan", "retrieve"]
    
    # Generate initial answer
    print("\n📝 Initial Generation:")
    state = generate_node(state)
    
    # Judge and correct loop
    max_attempts = 2
    attempt = 0
    
    while attempt <= max_attempts:
        # Judge
        result = judge_node(state)
        state.update(result)
        
        # Check if we should correct
        decision = should_correct(state)
        
        if decision == "finalize":
            print(f"\n   ✅ FINALIZING after {attempt + 1} attempt(s)")
            break
        else:
            attempt += 1
            if attempt > max_attempts:
                print(f"\n   ⚠️ Max attempts reached, finalizing...")
                break
            
            # Correct
            state.update(correct_node(state))
            
            # Re-generate
            print(f"\n📝 Re-generation (attempt {attempt}):")
            state.update(generate_node(state))
    
    # Ensure question is always in the final state
    state["question"] = question
    
    return state

# Test scenarios
test_scenarios = [
    {
        "name": "Good Answer (No Correction)",
        "question": "What is the capital of France?",
        "docs": [
            "France is a country in Western Europe. Its capital is Paris.",
            "Paris has a population of about 2.1 million.",
            "France is known for its culture and cuisine."
        ]
    },
    {
        "name": "Hallucinated Answer (Needs Correction)",
        "question": "Which magazine did The Doors appear on in 1967?",
        "docs": [
            "The Doors were an American rock band formed in 1965.",
            "They were known for their unique sound.",
            "Rolling Stone is a magazine that covers music."
        ]
    },
    {
        "name": "Irrelevant Answer (Needs Correction)",
        "question": "What is the population of the capital of France?",
        "docs": [
            "France is a country in Western Europe.",
            "French cuisine is famous worldwide.",
            "Paris is known as the City of Light."
        ]
    }
]

print("\n📌 Running simulations...")
print("-" * 60)

for i, scenario in enumerate(test_scenarios, 1):
    print(f"\n{'='*60}")
    print(f"Scenario {i}: {scenario['name']}")
    print('='*60)
    
    result = simulate_agent_flow(scenario['question'], scenario['docs'])
    
    print(f"\n📊 Final Result:")
    print(f"   Question: {result['question']}")
    print(f"   Final Answer: {result['draft_answer'][:100]}...")
    print(f"   Judge Score: {result.get('judge_score', 0)}/10")
    print(f"   Correction Attempts: {result.get('correction_attempts', 0)}")
    print(f"   Issue: {result.get('judge_issue', 'none')}")
    print(f"   Node History: {' → '.join(result.get('node_history', []))}")
# ============================================
# Part 9: Correction Decision Matrix
# ============================================
print("\n📊 PART 9: CORRECTION DECISION MATRIX")
print("-" * 40)

print("""
Decision Matrix for Self-Correction:

┌─────────────┬─────────────┬─────────────────┐
│ Score Range │ Issue       │ Decision        │
├─────────────┼─────────────┼─────────────────┤
│ 9-10        │ none        │ ✅ FINALIZE     │
│ 7-8         │ none        │ ✅ FINALIZE     │
│ 7-8         │ minor       │ 🔄 CORRECT      │
│ 5-6         │ incomplete  │ 🔄 CORRECT      │
│ 5-6         │ minor       │ 🔄 CORRECT      │
│ 0-4         │ hallucination│ 🔄 CORRECT     │
│ 0-4         │ contradiction│ 🔄 CORRECT     │
│ 0-4         │ irrelevant  │ 🔄 CORRECT     │
└─────────────┴─────────────┴─────────────────┘

Max Correction Attempts: 2
After 2 attempts: Force FINALIZE
""")

# ============================================
# Part 10: Summary
# ============================================
print("\n📊 DAY 16 SUMMARY")
print("-" * 40)
print("""
✅ Built conditional self-correction logic
✅ Implemented should_correct() decision function
✅ Added different correction strategies
✅ Simulated the correction loop
✅ Created decision matrix

🎯 What Self-Correction Does:
   1. Judges answer quality
   2. Decides if correction is needed
   3. Applies different strategies per issue
   4. Iterates up to 2 times
   5. Finalizes when good enough

📋 Correction Strategies:
   - Hallucination: Focus on accuracy
   - Contradiction: Focus on consistency
   - Irrelevant: Focus on relevance
   - Incomplete: Focus on completeness
   - General: Focus on feedback

🚀 Next: Wire the full loop with corrections!
""")

print("\n🚀 Ready for Day 17! (Wire Full Loop)")
print("=" * 60)