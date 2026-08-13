"""
Day 10: Planner Node
Breaking down complex questions into sub-questions for better retrieval.
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict, Any
from datetime import datetime
import json
import time

print("=" * 60)
print("🎯 DAY 10: PLANNER NODE")
print("=" * 60)

# ============================================
# Part 1: Understanding the Planner
# ============================================
print("\n📚 WHAT DOES THE PLANNER DO?")
print("-" * 40)
print("""
The Planner Node breaks down complex questions into simpler sub-questions.

Why Planning Matters:
1. Complex questions need multiple pieces of information
2. Each piece can be retrieved separately
3. Combining information gives better answers
4. Single retrieval often misses important context

Example:
    Complex Question: "Which magazine did The Doors appear on the cover of in 1967?"
    
    Sub-questions:
    1. "What is The Doors?"
    2. "Which magazines featured The Doors?"
    3. "What happened with The Doors in 1967?"
    
    Each sub-question retrieves different relevant documents!
""")

# ============================================
# Part 2: Define Agent State
# ============================================
print("\n📦 PART 2: DEFINING AGENT STATE")
print("-" * 40)

from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    """The state of our agentic RAG system"""
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
# Part 3: Initialize LLM
# ============================================
print("\n🤖 PART 3: INITIALIZING LLM")
print("-" * 40)

try:
    # Initialize Ollama (free, local)
    llm = ChatOllama(
        model="mistral",
        temperature=0,  # Deterministic output
    )
    print("✅ LLM initialized!")
    print("   Model: mistral (free, local)")
    print("   Temperature: 0 (deterministic)")
except Exception as e:
    print(f"❌ Error initializing LLM: {e}")
    print("\n💡 Make sure Ollama is running:")
    print("   1. Open a new terminal")
    print("   2. Run: ollama serve")
    print("   3. Run: ollama pull mistral")
    exit(1)

# ============================================
# Part 4: Create Planner Prompt
# ============================================
print("\n📝 PART 4: CREATING PLANNER PROMPT")
print("-" * 40)

# The prompt that tells the LLM how to plan
PLANNER_PROMPT = """
You are a research planner. Your job is to break down complex questions into 1-3 focused sub-questions.

Rules:
1. Sub-questions should be SPECIFIC and SEARCHABLE
2. Each sub-question should target ONE piece of information
3. Sub-questions should COVER all aspects of the original question
4. Return ONLY the sub-questions, one per line
5. Do NOT number them or add any extra text

Examples:

Question: "Which magazine did The Doors appear on the cover of in 1967?"
Sub-questions:
What is The Doors?
Which magazines featured The Doors on their cover in 1967?

Question: "Who was the first woman to win a Nobel Prize, and in which category?"
Sub-questions:
Who was the first woman to win a Nobel Prize?
Which category did the first female Nobel laureate win in?

Question: "What is the population of the capital of France?"
Sub-questions:
What is the capital of France?
What is the population of Paris?

Now, break down this question:
Question: {question}

Sub-questions:
"""

# Create the prompt template
planner_prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)

print("✅ Planner prompt created!")

# ============================================
# Part 5: Build the Planner Node
# ============================================
print("\n🔧 PART 5: BUILDING THE PLANNER NODE")
print("-" * 40)

def plan_node(state: AgentState) -> Dict[str, Any]:
    """
    Planner Node: Breaks down the question into sub-questions.
    
    Args:
        state: Current agent state with question
        
    Returns:
        Updates to the state with sub_questions
    """
    print(f"\n   📝 Planning for: {state['question']}")
    
    try:
        # Generate sub-questions using the LLM
        response = (planner_prompt | llm).invoke({
            "question": state["question"]
        })
        
        # Parse the response into a list
        raw_response = response.content
        print(f"   Raw response: {raw_response}")
        
        # Clean and split into sub-questions
        sub_questions = []
        for line in raw_response.strip().split("\n"):
            line = line.strip()
            # Remove numbering if present
            if line and not line.startswith("Sub-questions:"):
                # Remove numbers like "1.", "2.", etc.
                if line and line[0].isdigit() and len(line) > 2 and line[1] in ". )":
                    line = line[2:].strip()
                if line:
                    sub_questions.append(line)
        
        # If no sub-questions generated, use the original question
        if not sub_questions:
            print(f"   ⚠️ No sub-questions generated, using original question")
            sub_questions = [state["question"]]
        
        print(f"   ✅ Generated {len(sub_questions)} sub-questions:")
        for i, sq in enumerate(sub_questions, 1):
            print(f"      {i}. {sq}")
        
        # Return updates to state
        return {
            "sub_questions": sub_questions,
            "node_history": state.get("node_history", []) + ["plan"]
        }
        
    except Exception as e:
        print(f"   ❌ Error in planning: {e}")
        # Fallback: use the original question
        return {
            "sub_questions": [state["question"]],
            "node_history": state.get("node_history", []) + ["plan"]
        }

print("✅ plan_node() function defined!")

# ============================================
# Part 6: Test the Planner
# ============================================
print("\n🧪 PART 6: TESTING THE PLANNER")
print("-" * 40)

# Test questions
test_questions = [
    "Which magazine did The Doors appear on the cover of in 1967?",
    "Who was the first woman to win a Nobel Prize, and in which category?",
    "What is the population of the capital of France?",
    "What is the name of the fight song of the university whose main campus is in Lawrence, Kansas?",
]

print("\n📌 Testing planner on multiple questions...")
print("-" * 40)

for i, question in enumerate(test_questions, 1):
    print(f"\n📌 Test {i}:")
    print(f"   Question: {question}")
    
    # Create initial state
    state = create_initial_state(question)
    
    # Run the planner
    print(f"   Running planner...")
    result = plan_node(state)
    
    # Update state with results
    state["sub_questions"] = result["sub_questions"]
    state["node_history"] = result["node_history"]
    
    print(f"   Sub-questions: {len(state['sub_questions'])}")
    for j, sq in enumerate(state["sub_questions"], 1):
        print(f"      {j}. {sq}")
    print()

# ============================================
# Part 7: Analyze the Planner's Output
# ============================================
print("\n📊 PART 7: ANALYZING PLANNER OUTPUT")
print("-" * 40)

def analyze_plan(sub_questions: List[str], original_question: str) -> Dict[str, Any]:
    """
    Analyze the quality of the plan.
    """
    analysis = {
        "num_sub_questions": len(sub_questions),
        "avg_length": sum(len(q) for q in sub_questions) / len(sub_questions) if sub_questions else 0,
        "coverage": "Good" if len(sub_questions) >= 2 else "May be insufficient",
    }
    
    # Check if sub-questions are specific
    specific_count = 0
    for q in sub_questions:
        # Check for question words
        question_words = ["what", "who", "where", "when", "why", "how", "which"]
        if any(word in q.lower() for word in question_words):
            specific_count += 1
    
    analysis["specific_questions"] = specific_count
    analysis["specific_ratio"] = specific_count / len(sub_questions) if sub_questions else 0
    
    return analysis

# Analyze the plans from test questions
for i, question in enumerate(test_questions):
    print(f"\n📌 Test {i+1}:")
    print(f"   Original: {question[:50]}...")
    
    # Get the sub-questions from the test
    state = create_initial_state(question)
    result = plan_node(state)
    
    analysis = analyze_plan(result["sub_questions"], question)
    
    print(f"   Sub-questions: {analysis['num_sub_questions']}")
    print(f"   Avg length: {analysis['avg_length']:.0f} chars")
    print(f"   Specific questions: {analysis['specific_questions']}/{analysis['num_sub_questions']}")
    print(f"   Coverage: {analysis['coverage']}")

# ============================================
# Part 8: Why Planning Improves Retrieval
# ============================================
print("\n🎯 PART 8: WHY PLANNING IMPROVES RETRIEVAL")
print("-" * 40)

print("""
Without Planning (Simple RAG):
    Question → One retrieval → One generation → Answer
    
    Problem: A single retrieval often misses important context
    Example: "Which magazine did The Doors appear on in 1967?"
    → One retrieval might find "The Doors" but not "1967 magazine"


With Planning (Agentic RAG):
    Question → Multiple retrievals → Better generation → Answer
    
    Sub-question 1: "What is The Doors?"
        → Retrieves: Band history, formation, members
    
    Sub-question 2: "Which magazines featured The Doors in 1967?"
        → Retrieves: Magazine covers, Rolling Stone, 1967 events
    
    Result: Both pieces of information combine for a complete answer!
""")

# ============================================
# Part 9: Planning Examples
# ============================================
print("\n📋 PART 9: PLANNING EXAMPLES")
print("-" * 40)

examples = [
    {
        "question": "What is the name of the fight song of the university whose main campus is in Lawrence, Kansas?",
        "expected": ["What university has its main campus in Lawrence, Kansas?", "What is the fight song of that university?"]
    },
    {
        "question": "Which director won the Academy Award for Best Director in 1994?",
        "expected": ["Who won the Academy Award for Best Director in 1994?", "Which director won the Oscar for Best Director in 1994?"]
    },
    {
        "question": "What is the population of the largest city in Texas?",
        "expected": ["What is the largest city in Texas?", "What is the population of that city?"]
    }
]

print("📌 Good Planning Examples:")
for i, example in enumerate(examples, 1):
    print(f"\n   Example {i}:")
    print(f"   Question: {example['question']}")
    print(f"   Sub-questions:")
    for j, sq in enumerate(example['expected'], 1):
        print(f"      {j}. {sq}")

print("\n💡 Key Insight:")
print("   Good sub-questions are:")
print("   1. SPECIFIC - target one piece of information")
print("   2. SEARCHABLE - easy to find documents for")
print("   3. COVER ALL - collectively answer the original question")

# ============================================
# Part 10: Summary
# ============================================
print("\n📊 DAY 10 SUMMARY")
print("-" * 40)
print("""
✅ Built the Planner Node
✅ Created the planning prompt
✅ Tested on multiple questions
✅ Analyzed planning quality
✅ Understood why planning improves retrieval

🎯 What the Planner Does:
   1. Takes a complex question
   2. Breaks it into 1-3 sub-questions
   3. Each sub-question targets ONE piece of information
   4. Sub-questions are specific and searchable

📋 Key Concepts:
   1. Planning enables MULTI-HOP reasoning
   2. Each hop retrieves different information
   3. Combined information gives complete answers
   4. This is what makes our agent better than simple RAG

🚀 Next: The Retriever Node will use these sub-questions
   to get relevant documents!
""")

print("\n🚀 Ready for Day 11! (Retriever Node)")
print("=" * 60)