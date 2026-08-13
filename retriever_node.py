"""
Day 11: Retriever Node
Retrieving relevant documents for each sub-question.
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict, Any, TypedDict, Optional
from datetime import datetime
import json
import os

print("=" * 60)
print("🔍 DAY 11: RETRIEVER NODE")
print("=" * 60)

# ============================================
# Part 1: Understanding the Retriever
# ============================================
print("\n📚 WHAT DOES THE RETRIEVER DO?")
print("-" * 40)
print("""
The Retriever Node takes sub-questions and finds relevant documents.

How it works:
1. For each sub-question, it searches the vector store
2. Finds the top K most relevant chunks
3. Combines all retrieved documents
4. Returns them for the Generator

Why this matters:
1. Multiple sub-questions = more comprehensive retrieval
2. Each retrieval targets a specific piece of information
3. Combined documents cover all aspects of the question
4. Better retrieval = better answers
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
# Part 3: Load Vector Store
# ============================================
print("\n🗄️ PART 3: LOADING VECTOR STORE")
print("-" * 40)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("✅ Embeddings initialized!")

# Try to load vector store
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

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}  # Return top 4 chunks per query
)
print(f"✅ Retriever created! (k=4)")

# ============================================
# Part 4: Build the Retriever Node
# ============================================
print("\n🔧 PART 4: BUILDING THE RETRIEVER NODE")
print("-" * 40)

def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """
    Retriever Node: Gets documents for each sub-question.
    
    Args:
        state: Current agent state with sub_questions
        
    Returns:
        Updates to the state with retrieved_docs
    """
    sub_questions = state.get("sub_questions", [state["question"]])
    
    print(f"\n   🔍 Retrieving for {len(sub_questions)} sub-question(s)...")
    
    all_docs = []
    all_sources = []
    
    for i, query in enumerate(sub_questions, 1):
        print(f"      Query {i}: {query}")
        
        try:
            # Retrieve documents for this query
            docs = retriever.invoke(query)
            
            # Extract content and sources
            doc_contents = [doc.page_content for doc in docs]
            doc_sources = [doc.metadata.get('title', 'Unknown') for doc in docs]
            
            # Add to combined results
            all_docs.extend(doc_contents)
            all_sources.extend(doc_sources)
            
            print(f"         Retrieved {len(docs)} chunks")
            print(f"         First source: {doc_sources[0] if doc_sources else 'Unknown'}")
            
        except Exception as e:
            print(f"         ❌ Error retrieving: {e}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_docs = []
    unique_sources = []
    
    for doc, source in zip(all_docs, all_sources):
        if doc not in seen:
            seen.add(doc)
            unique_docs.append(doc)
            unique_sources.append(source)
    
    print(f"\n   ✅ Retrieved {len(unique_docs)} unique documents")
    print(f"   Sources: {', '.join(unique_sources[:3])}{'...' if len(unique_sources) > 3 else ''}")
    
    # Return updates to state
    return {
        "retrieved_docs": unique_docs,
        "sources": unique_sources,
        "node_history": state.get("node_history", []) + ["retrieve"]
    }

print("✅ retrieve_node() function defined!")

# ============================================
# Part 5: Test the Retriever
# ============================================
print("\n🧪 PART 5: TESTING THE RETRIEVER")
print("-" * 40)

# Create a state with sub-questions
test_questions = [
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

print("\n📌 Testing retriever on multiple questions...")
print("-" * 40)

for i, test in enumerate(test_questions, 1):
    print(f"\n📌 Test {i}:")
    print(f"   Question: {test['question']}")
    print(f"   Sub-questions: {len(test['sub_questions'])}")
    
    # Create initial state with sub-questions
    state = create_initial_state(test['question'])
    state["sub_questions"] = test['sub_questions']
    state["node_history"] = ["plan"]  # Simulate that planning happened
    
    # Run the retriever
    print(f"\n   Running retriever...")
    result = retrieve_node(state)
    
    # Update state with results
    state["retrieved_docs"] = result["retrieved_docs"]
    state["sources"] = result["sources"]
    state["node_history"] = result["node_history"]
    
    print(f"\n   📊 Results:")
    print(f"      Total retrieved: {len(state['retrieved_docs'])} documents")
    print(f"      Unique sources: {len(set(state['sources']))}")
    
    # Show sample of retrieved documents
    if state['retrieved_docs']:
        print(f"\n   📄 Sample document:")
        sample = state['retrieved_docs'][0]
        print(f"      Preview: {sample[:150]}...")
        print(f"      Source: {state['sources'][0] if state['sources'] else 'Unknown'}")

# ============================================
# Part 6: Analyze Retrieval Quality
# ============================================
print("\n📊 PART 6: ANALYZING RETRIEVAL QUALITY")
print("-" * 40)

def analyze_retrieval(retrieved_docs: List[str], sub_questions: List[str]) -> Dict[str, Any]:
    """
    Analyze the quality of retrieval.
    """
    analysis = {
        "total_docs": len(retrieved_docs),
        "avg_length": sum(len(doc) for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0,
        "has_content": len(retrieved_docs) > 0,
    }
    
    # Check if documents are relevant to sub-questions
    relevant_count = 0
    for doc in retrieved_docs:
        doc_lower = doc.lower()
        for q in sub_questions:
            # Check if any key terms from sub-question appear in document
            terms = q.lower().split()
            if any(term in doc_lower for term in terms[:3]):
                relevant_count += 1
                break
    
    analysis["relevant_ratio"] = relevant_count / len(retrieved_docs) if retrieved_docs else 0
    
    return analysis

# Analyze retrieval for test cases
print("\n📌 Retrieval Analysis:")
for i, test in enumerate(test_questions, 1):
    state = create_initial_state(test['question'])
    state["sub_questions"] = test['sub_questions']
    
    result = retrieve_node(state)
    state["retrieved_docs"] = result["retrieved_docs"]
    
    analysis = analyze_retrieval(state["retrieved_docs"], test['sub_questions'])
    
    print(f"\n   Test {i}:")
    print(f"      Total docs: {analysis['total_docs']}")
    print(f"      Avg length: {analysis['avg_length']:.0f} chars")
    print(f"      Relevant ratio: {analysis['relevant_ratio']:.1%}")
    print(f"      Has content: {'✅' if analysis['has_content'] else '❌'}")

# ============================================
# Part 7: How Retrieval Works
# ============================================
print("\n🔍 PART 7: HOW RETRIEVAL WORKS")
print("-" * 40)

print("""
Retrieval Process:

1. Query → Embedding (Vector)
   "What is The Doors?" → [0.12, 0.87, -0.34, ...]

2. Vector Search in ChromaDB
   - Compares query vector to all chunk vectors
   - Uses cosine similarity (1 = most similar)
   - Returns top K matches (K=4)

3. Return Documents
   - Each match has:
     - page_content: The actual text
     - metadata: title, source, etc.
   - Combined and returned to state

Cosine Similarity Formula:
    similarity = cos(θ) = (A · B) / (||A|| × ||B||)
    
    A = query vector
    B = document vector
    Higher value = more similar
""")

# ============================================
# Part 8: Compare with Simple RAG
# ============================================
print("\n🔄 PART 8: COMPARE WITH SIMPLE RAG")
print("-" * 40)

print("""
Simple RAG (One Query):
    Query: "Which magazine did The Doors appear on in 1967?"
    → One search
    → 4 documents
    → Limited context

Agentic RAG (Multiple Queries):
    Query 1: "What is The Doors?"
    → 4 documents
    
    Query 2: "Which magazines featured The Doors in 1967?"
    → 4 documents
    
    Total: 8 documents (more context!)
    
Why multiple queries are better:
1. Each query targets different aspects
2. Retrieves more relevant information
3. Covers more ground
4. Better final answers
""")

# ============================================
# Part 9: Retrieval Statistics
# ============================================
print("\n📊 PART 9: RETRIEVAL STATISTICS")
print("-" * 40)

# Calculate statistics from vector store
try:
    collection = vectorstore._collection
    total_vectors = collection.count()
    print(f"📊 Vector Store Statistics:")
    print(f"   Total vectors: {total_vectors}")
    print(f"   Vector dimension: 384")
    print(f"   Retriever k: 4")
    print(f"   Storage: {loaded_path}")
except Exception as e:
    print(f"   Could not get statistics: {e}")

# ============================================
# Part 10: Summary
# ============================================
print("\n📊 DAY 11 SUMMARY")
print("-" * 40)
print("""
✅ Loaded vector store and retriever
✅ Built the Retriever Node
✅ Tested on multiple questions
✅ Analyzed retrieval quality
✅ Understood how retrieval works

🎯 What the Retriever Does:
   1. Takes sub-questions from Planner
   2. For each, searches the vector store
   3. Returns top K most relevant chunks
   4. Combines all results

📋 Key Concepts:
   1. Multiple queries = more comprehensive retrieval
   2. Each query targets one piece of information
   3. Combined documents provide full context
   4. Better retrieval = better answers

🚀 Next: The Generator Node will use these
   documents to write the answer!
""")

print("\n🚀 Ready for Day 12! (Generator Node)")
print("=" * 60)