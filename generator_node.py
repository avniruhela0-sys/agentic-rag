"""
Day 12: Generator Node
Generating answers with Chain-of-Thought reasoning.
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List, Dict, Any, TypedDict, Optional
from datetime import datetime
import json
import os

print("=" * 60)
print("✍️ DAY 12: GENERATOR NODE")
print("=" * 60)

# ============================================
# Part 1: Understanding the Generator
# ============================================
print("\n📚 WHAT DOES THE GENERATOR DO?")
print("-" * 40)
print("""
The Generator Node takes retrieved documents and writes an answer.

How it works:
1. Takes retrieved documents from Retriever Node
2. Uses Chain-of-Thought reasoning (reason step by step)
3. Generates a comprehensive answer
4. Returns the draft answer for evaluation

Why Chain-of-Thought matters:
1. Shows the reasoning process
2. Makes answers more accurate
3. Helps identify errors
4. Builds trust with users
""")

# ============================================
# Part 2: Define Agent State
# ============================================
print("\n📦 PART 2: DEFINING AGENT STATE")
print("-" * 40)

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
# Part 4: Load Vector Store (for context)
# ============================================
print("\n🗄️ PART 4: LOADING VECTOR STORE")
print("-" * 40)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("✅ Embeddings initialized!")

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

# ============================================
# Part 5: Create Generator Prompt
# ============================================
print("\n📝 PART 5: CREATING GENERATOR PROMPT")
print("-" * 40)

# The prompt with Chain-of-Thought reasoning
GENERATOR_PROMPT = """
You are a research assistant. Using the provided context, answer the question accurately.

IMPORTANT: First, reason step by step in 2-3 sentences. Then give your final answer clearly labeled.

Context:
{context}

Question: {question}

Step-by-step reasoning:
1. What does the context tell us about this question?
2. Is there enough information to answer?
3. What is the most accurate answer?

Final Answer:
"""

# Create the prompt template
generator_prompt = ChatPromptTemplate.from_template(GENERATOR_PROMPT)

print("✅ Generator prompt created with Chain-of-Thought!")

# ============================================
# Part 6: Build the Generator Node
# ============================================
print("\n🔧 PART 6: BUILDING THE GENERATOR NODE")
print("-" * 40)

def generate_node(state: AgentState) -> Dict[str, Any]:
    """
    Generator Node: Writes an answer with Chain-of-Thought reasoning.
    
    Args:
        state: Current agent state with question and retrieved_docs
        
    Returns:
        Updates to the state with draft_answer
    """
    question = state["question"]
    retrieved_docs = state.get("retrieved_docs", [])
    sources = state.get("sources", [])
    
    print(f"\n   ✍️ Generating answer for: {question[:50]}...")
    print(f"   📄 Using {len(retrieved_docs)} documents as context")
    
    if not retrieved_docs:
        print("   ⚠️ No documents retrieved! Using fallback response.")
        return {
            "draft_answer": "I don't have enough information to answer this question. Please try rephrasing or provide more context.",
            "node_history": state.get("node_history", []) + ["generate"]
        }
    
    try:
        # Combine documents into context
        context = "\n\n".join(retrieved_docs)
        
        # Truncate context if too long (for token limits)
        if len(context) > 4000:
            context = context[:4000] + "..."
        
        # Generate answer with Chain-of-Thought
        response = (generator_prompt | llm).invoke({
            "context": context,
            "question": question
        })
        
        draft_answer = response.content
        
        print(f"   ✅ Generated answer ({len(draft_answer)} characters)")
        
        # Show first few lines of the answer
        preview = draft_answer[:150] + "..." if len(draft_answer) > 150 else draft_answer
        print(f"   Preview: {preview}")
        
        # Show sources used
        if sources:
            unique_sources = list(set(sources))[:3]
            print(f"   Sources: {', '.join(unique_sources)}")
        
        # Return updates to state
        return {
            "draft_answer": draft_answer,
            "node_history": state.get("node_history", []) + ["generate"]
        }
        
    except Exception as e:
        print(f"   ❌ Error generating answer: {e}")
        return {
            "draft_answer": f"Error generating answer: {str(e)}",
            "node_history": state.get("node_history", []) + ["generate"]
        }

print("✅ generate_node() function defined!")

# ============================================
# Part 7: Helper Function for Testing
# ============================================
print("\n🔧 PART 7: HELPER FUNCTION")
print("-" * 40)

def simulate_retrieval(question: str, sub_questions: List[str]) -> List[str]:
    """
    Simulate retrieval for testing the generator.
    """
    all_docs = []
    
    for query in sub_questions:
        try:
            docs = retriever.invoke(query)
            for doc in docs:
                all_docs.append(doc.page_content)
        except Exception as e:
            print(f"   ⚠️ Error retrieving for '{query}': {e}")
    
    # Remove duplicates
    seen = set()
    unique_docs = []
    for doc in all_docs:
        if doc not in seen:
            seen.add(doc)
            unique_docs.append(doc)
    
    return unique_docs

print("✅ simulate_retrieval() function defined!")

# ============================================
# Part 8: Test the Generator
# ============================================
print("\n🧪 PART 8: TESTING THE GENERATOR")
print("-" * 40)

# Test cases with questions and sub-questions
test_cases = [
    {
        "question": "Which magazine did The Doors appear on the cover of in 1967?",
        "sub_questions": [
            "What is The Doors?",
            "Which magazines featured The Doors on their cover in 1967?"
        ]
    },
    {
        "question": "Who was the first woman to win a Nobel Prize?",
        "sub_questions": [
            "Who was the first woman to win a Nobel Prize?"
        ]
    },
    {
        "question": "What is the population of the capital of France?",
        "sub_questions": [
            "What is the capital of France?",
            "What is the population of Paris?"
        ]
    }
]

print("\n📌 Testing generator on multiple questions...")
print("-" * 40)

for i, test in enumerate(test_cases, 1):
    print(f"\n📌 Test {i}:")
    print(f"   Question: {test['question']}")
    print(f"   Sub-questions: {len(test['sub_questions'])}")
    
    # Create initial state
    state = create_initial_state(test['question'])
    state["sub_questions"] = test['sub_questions']
    state["node_history"] = ["plan"]
    
    # Simulate retrieval
    print(f"\n   📄 Retrieving documents...")
    retrieved_docs = simulate_retrieval(test['question'], test['sub_questions'])
    state["retrieved_docs"] = retrieved_docs
    print(f"   Retrieved {len(retrieved_docs)} documents")
    
    # Run the generator
    print(f"\n   ✍️ Generating answer...")
    result = generate_node(state)
    
    # Update state
    state["draft_answer"] = result["draft_answer"]
    state["node_history"] = result["node_history"]
    
    print(f"\n   📊 Generated Answer:")
    print(f"   {state['draft_answer']}")
    print("-" * 40)

# ============================================
# Part 9: Compare with/without Chain-of-Thought
# ============================================
print("\n🔄 PART 9: WITH vs WITHOUT CHAIN-OF-THOUGHT")
print("-" * 40)

print("""
WITHOUT Chain-of-Thought:
    Answer: The Doors appeared on Rolling Stone in 1967.
    
    Problem: No reasoning shown. Hard to trust.


WITH Chain-of-Thought:
    Step 1: The context tells us The Doors are a rock band.
    Step 2: The context mentions Rolling Stone magazine.
    Step 3: The context confirms they appeared in 1967.
    
    Final Answer: The Doors appeared on the cover of Rolling Stone magazine in 1967.
    
    Benefits:
    ✅ Shows the reasoning process
    ✅ Builds trust
    ✅ Helps identify errors
    ✅ More accurate answers
""")

# ============================================
# Part 10: Chain-of-Thought Examples
# ============================================
print("\n📋 PART 10: CHAIN-OF-THOUGHT EXAMPLES")
print("-" * 40)

print("""
Good Chain-of-Thought Example:

Question: "What is the population of the capital of France?"

Step 1: The context tells us the capital of France is Paris.
Step 2: The context gives the population of Paris as 2.1 million.
Step 3: This is the most direct answer to the question.

Final Answer: The population of Paris (the capital of France) is approximately 2.1 million.


Why Chain-of-Thought helps:
1. Shows the reasoning path
2. Identifies key facts
3. Confirms understanding
4. Enables correction if reasoning is wrong
""")

# ============================================
# Part 11: Summary
# ============================================
print("\n📊 DAY 12 SUMMARY")
print("-" * 40)
print("""
✅ Built the Generator Node
✅ Created Chain-of-Thought prompt
✅ Tested on multiple questions
✅ Compared with/without Chain-of-Thought
✅ Understood why reasoning matters

🎯 What the Generator Does:
   1. Takes retrieved documents
   2. Uses Chain-of-Thought reasoning
   3. Writes a comprehensive answer
   4. Returns draft answer for evaluation

📋 Key Concepts:
   1. Chain-of-Thought = reasoning step by step
   2. Shows the logic behind answers
   3. Builds user trust
   4. Enables better evaluation

🚀 Next: The Judge Node will evaluate if this
   answer is correct or needs correction!
""")

print("\n🚀 Ready for Day 13! (Wire the Loop)")
print("=" * 60)