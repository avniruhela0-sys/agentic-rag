"""
Day 15: LLM-as-Judge Node
Self-evaluation mechanism for the agent.
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import TypedDict, List, Dict, Any
from datetime import datetime
import json
import os
import re

print("=" * 60)
print("⚖️ DAY 15: LLM-AS-JUDGE NODE")
print("=" * 60)

# ============================================
# Part 1: Understanding LLM-as-Judge
# ============================================
print("\n📚 WHAT IS LLM-AS-JUDGE?")
print("-" * 40)
print("""
LLM-as-Judge is a technique where we use an LLM to evaluate 
the quality of another LLM's output.

Why Self-Evaluation Matters:
1. No human needed to check answers
2. Identifies hallucinations (false information)
3. Catches contradictions with source material
4. Detects irrelevant or off-topic answers
5. Enables self-correction

How it works:
1. We give the judge: context + question + answer
2. The judge scores it (0-10)
3. The judge identifies issues
4. Based on the score, we decide to correct or finalize

Score Interpretation:
- 0-4: Poor (wrong, hallucinated, or contradictory)
- 5-6: Mediocre (partially correct, missing info)
- 7-8: Good (mostly correct, well-supported)
- 9-10: Excellent (completely correct with strong evidence)
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

# Load vector store for context (optional)
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
    print("⚠️ No vector store found! Judge will work without context.")

retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) if vectorstore else None
print("✅ Components ready!")

# ============================================
# Part 4: Create Judge Prompt
# ============================================
print("\n📝 PART 4: CREATING JUDGE PROMPT")
print("-" * 40)

JUDGE_PROMPT = """
You are a strict, objective evaluator. Your job is to score answers and identify issues.

**Scoring Criteria (0-10):**

Score 9-10: EXCELLENT
- Completely answers the question
- All information is supported by the context
- No hallucinations or contradictions
- Clear and concise

Score 7-8: GOOD
- Mostly answers the question
- Most information is supported by the context
- Minor missing details
- Generally accurate

Score 5-6: MEDIOCRE
- Partially answers the question
- Some information is supported, some isn't
- Missing key details
- May have minor hallucinations

Score 0-4: POOR
- Doesn't answer the question
- Major hallucinations or contradictions
- Information not in context
- Incorrect or misleading

**Issue Types:**
- hallucination: Claims made that are NOT in the context
- contradiction: Claims that CONTRADICT the context
- irrelevant: Doesn't answer the question asked
- incomplete: Missing important information
- unsupported: Claims not supported by sufficient evidence
- none: No issues found

**Context (source material):**
{context}

**Question:**
{question}

**Answer to evaluate:**
{answer}

**Your Response Format (EXACTLY):**
SCORE: <number 0-10>
ISSUE: <issue_type>
REASON: <one sentence explaining your score>
IMPROVEMENT: <one sentence on how to improve>

Now evaluate:
"""

# Create the prompt template
judge_prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)

print("✅ Judge prompt created!")
print("\n📋 Judge will look for:")
print("   - Hallucination: Claims not in context")
print("   - Contradiction: Claims that contradict context")
print("   - Irrelevant: Doesn't answer the question")
print("   - Incomplete: Missing important information")
print("   - Unsupported: Not enough evidence")

# ============================================
# Part 5: Build the Judge Node
# ============================================
print("\n🔧 PART 5: BUILDING THE JUDGE NODE")
print("-" * 40)

def parse_judge_response(response: str) -> Dict[str, Any]:
    """
    Parse the judge's response into structured data.
    """
    result = {
        "score": 0.0,
        "issue": "none",
        "reason": "Could not parse response",
        "improvement": ""
    }
    
    lines = response.strip().split("\n")
    
    for line in lines:
        line = line.strip()
        
        if line.upper().startswith("SCORE:"):
            try:
                # Extract number
                score_part = line.split("SCORE:")[1].strip()
                # Handle cases like "7/10" or "7.5"
                if "/" in score_part:
                    score_part = score_part.split("/")[0]
                result["score"] = float(score_part)
            except:
                result["score"] = 5.0
                
        elif line.upper().startswith("ISSUE:"):
            issue = line.split("ISSUE:")[1].strip().lower()
            # Validate issue type
            valid_issues = ["hallucination", "contradiction", "irrelevant", 
                          "incomplete", "unsupported", "none"]
            if issue in valid_issues:
                result["issue"] = issue
            else:
                result["issue"] = "none"
                
        elif line.upper().startswith("REASON:"):
            result["reason"] = line.split("REASON:")[1].strip()
            
        elif line.upper().startswith("IMPROVEMENT:"):
            result["improvement"] = line.split("IMPROVEMENT:")[1].strip()
    
    # Clamp score to 0-10
    result["score"] = max(0, min(10, result["score"]))
    
    return result

def judge_node(state: AgentState) -> Dict[str, Any]:
    """
    Judge Node: Evaluates the answer quality.
    """
    print(f"\n   ⚖️ JUDGING answer...")
    
    # Get the answer to judge
    answer = state.get("draft_answer", "")
    question = state.get("question", "")
    context = "\n\n".join(state.get("retrieved_docs", []))
    
    # Truncate context if too long
    if len(context) > 3000:
        context = context[:3000] + "..."
    
    print(f"      Question: {question[:50]}...")
    print(f"      Answer length: {len(answer)} characters")
    print(f"      Context length: {len(context)} characters")
    
    if not answer or answer.startswith("Error"):
        print("      ⚠️ Invalid answer, returning default score")
        return {
            "judge_score": 0.0,
            "judge_feedback": "Invalid or error answer",
            "judge_issue": "invalid",
            "node_history": state.get("node_history", []) + ["judge"]
        }
    
    try:
        # Call the judge LLM
        response = (judge_prompt | llm).invoke({
            "context": context if context else "No context provided.",
            "question": question,
            "answer": answer
        })
        
        # Parse the response
        parsed = parse_judge_response(response.content)
        
        score = parsed["score"]
        issue = parsed["issue"]
        reason = parsed["reason"]
        improvement = parsed["improvement"]
        
        print(f"      Score: {score}/10")
        print(f"      Issue: {issue}")
        print(f"      Reason: {reason}")
        if improvement:
            print(f"      Improvement: {improvement}")
        
        return {
            "judge_score": score,
            "judge_feedback": reason,
            "judge_issue": issue,
            "node_history": state.get("node_history", []) + ["judge"]
        }
        
    except Exception as e:
        print(f"      ❌ Error judging: {e}")
        return {
            "judge_score": 5.0,
            "judge_feedback": f"Error in judging: {e}",
            "judge_issue": "none",
            "node_history": state.get("node_history", []) + ["judge"]
        }

print("✅ judge_node() function defined!")

# ============================================
# Part 6: Test the Judge
# ============================================
print("\n🧪 PART 6: TESTING THE JUDGE")
print("-" * 40)

# Test cases with different scenarios
test_cases = [
    {
        "name": "Excellent Answer",
        "question": "What is the capital of France?",
        "context": "France is a country in Western Europe. Its capital is Paris. Paris has a population of about 2.1 million.",
        "answer": "The capital of France is Paris. It is located in Western Europe and has a population of approximately 2.1 million."
    },
    {
        "name": "Hallucinated Answer",
        "question": "What is the capital of France?",
        "context": "France is a country in Western Europe. Its capital is Paris.",
        "answer": "The capital of France is London. London is the largest city in the United Kingdom."
    },
    {
        "name": "Irrelevant Answer",
        "question": "What is the capital of France?",
        "context": "France is a country in Western Europe. Its capital is Paris.",
        "answer": "I like eating French food like croissants and baguettes."
    },
    {
        "name": "Incomplete Answer",
        "question": "What is the capital of France and its population?",
        "context": "France is a country in Western Europe. Its capital is Paris. Paris has a population of about 2.1 million.",
        "answer": "The capital of France is Paris."
    },
    {
        "name": "Perfect Answer",
        "question": "Which magazine did The Doors appear on the cover of in 1967?",
        "context": "The Doors were an American rock band formed in 1965. In 1967, they appeared on the cover of Rolling Stone magazine.",
        "answer": "The Doors appeared on the cover of Rolling Stone magazine in 1967."
    }
]

print("\n📌 Testing judge on various scenarios...")
print("-" * 60)

for i, test in enumerate(test_cases, 1):
    print(f"\n📌 Test {i}: {test['name']}")
    print("-" * 40)
    print(f"   Question: {test['question']}")
    print(f"   Answer: {test['answer'][:80]}...")
    
    # Create state
    state = create_initial_state(test['question'])
    state["draft_answer"] = test['answer']
    state["retrieved_docs"] = [test['context']]
    
    # Run judge
    result = judge_node(state)
    
    print(f"\n   📊 Judge Result:")
    print(f"      Score: {result['judge_score']}/10")
    print(f"      Issue: {result['judge_issue']}")
    print(f"      Feedback: {result['judge_feedback']}")
    print("-" * 40)

# ============================================
# Part 7: Judge Score Interpretation
# ============================================
print("\n📊 PART 7: JUDGE SCORE INTERPRETATION")
print("-" * 40)

def interpret_score(score: float) -> str:
    """Interpret the judge's score."""
    if score >= 9.0:
        return "🌟 EXCELLENT - Perfect answer, well-supported"
    elif score >= 7.0:
        return "✅ GOOD - Mostly correct, minor issues"
    elif score >= 5.0:
        return "⚠️ MEDIOCRE - Partially correct, needs improvement"
    else:
        return "❌ POOR - Wrong or hallucinated, needs correction"

print("""
Score Interpretation:

🌟 9-10: EXCELLENT
   - Completely answers the question
   - All claims are supported by context
   - No hallucinations or contradictions

✅ 7-8: GOOD
   - Mostly answers the question
   - Most claims are supported
   - May have minor missing details

⚠️ 5-6: MEDIOCRE
   - Partially answers the question
   - Some claims are supported, some aren't
   - Missing key information

❌ 0-4: POOR
   - Doesn't answer the question
   - Major hallucinations
   - Contradicts the context
""")

# ============================================
# Part 8: Why This Matters
# ============================================
print("\n🎯 PART 8: WHY LLM-AS-JUDGE MATTERS")
print("-" * 40)

print("""
Without LLM-as-Judge:
    Question → Generate → Answer (no quality check)
    → Wrong answers go undetected
    → No self-correction

With LLM-as-Judge:
    Question → Generate → Judge → Score
                              ↓
                       Score >= 7? 
                    YES ↓      ↓ NO
                 Finalize    Correct
                              ↓
                           Generate again
                              ↓
                           Judge again

Benefits:
1. Self-evaluation without humans
2. Identifies hallucinations automatically
3. Enables iterative improvement
4. Reduces false information
5. Builds trust in the system
""")

# ============================================
# Part 9: Summary
# ============================================
print("\n📊 DAY 15 SUMMARY")
print("-" * 40)
print("""
✅ Built the LLM-as-Judge Node
✅ Created detailed judge prompt with scoring criteria
✅ Implemented parse function for structured output
✅ Tested on multiple scenarios
✅ Understood why self-evaluation matters

🎯 What the Judge Does:
   1. Takes question, context, and answer
   2. Scores answer 0-10
   3. Identifies issues (hallucination, etc.)
   4. Provides feedback for improvement
   5. Enables self-correction

📋 Issue Types:
   - hallucination: Claims not in context
   - contradiction: Claims that contradict context
   - irrelevant: Doesn't answer the question
   - incomplete: Missing important information
   - unsupported: Insufficient evidence

🚀 Next: Add conditional self-correction logic!
""")

print("\n🚀 Ready for Day 16! (Conditional Self-Correction)")
print("=" * 60)