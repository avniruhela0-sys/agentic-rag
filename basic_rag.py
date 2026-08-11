"""
Basic RAG Pipeline - FREE VERSION (Ollama)
No OpenAI credits needed! Uses local LLM.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
import json
import os
from datetime import datetime

print("=" * 60)
print("📚 DAY 5: BASIC RAG PIPELINE (FREE - Ollama)")
print("=" * 60)

# ============================================
# Step 1: Load the vector store
# ============================================
print("\n📂 Loading vector store...")

try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Try all possible vector store locations
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
                continue

    if vectorstore is None:
        print("❌ No vector store found!")
        print("💡 Please run Day 4 first (vector_store_hf.py)")
        exit(1)
        
except Exception as e:
    print(f"❌ Error loading vector store: {e}")
    exit(1)

# ============================================
# Step 2: Create retriever
# ============================================
print("\n🔍 Creating retriever...")

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)
print("✅ Retriever created! Will return top 4 chunks")

# ============================================
# Step 3: Initialize FREE LLM (Ollama)
# ============================================
print("\n🤖 Initializing Ollama (FREE local LLM)...")

try:
    llm = ChatOllama(
        model="mistral",
        temperature=0,
    )
    print("✅ LLM initialized!")
    print("   Model: mistral (free, local)")
    print("   Temperature: 0 (deterministic)")
    print("   ✅ No OpenAI credits needed!")
    
except Exception as e:
    print(f"❌ Error initializing Ollama: {e}")
    print("\n💡 Troubleshooting:")
    print("   1. Make sure Ollama is running: 'ollama serve'")
    print("   2. Download a model: 'ollama pull mistral'")
    print("   3. Install: pip install langchain-ollama")
    exit(1)

# ============================================
# Step 4: Create the RAG prompt
# ============================================
print("\n📝 Creating RAG prompt...")

prompt = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below. 
If the context doesn't contain enough information to answer, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""
)

print("✅ Prompt created!")

# ============================================
# Step 5: Define the RAG function
# ============================================
print("\n⚙️ Defining RAG pipeline...")

def basic_rag(question):
    """Simple RAG pipeline: retrieve → generate"""
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    response = (prompt | llm).invoke({
        "context": context,
        "question": question
    })
    
    return {
        "answer": response.content,
        "retrieved_chunks": docs,
        "num_chunks": len(docs),
        "context": context
    }

print("✅ RAG pipeline ready!")

# ============================================
# Step 6: Test on sample questions
# ============================================
print("\n🧪 TESTING RAG PIPELINE")
print("-" * 40)

test_questions = [
    "What is the capital of France?",
    "Who directed the movie Ed Wood?",
    "What is the population of Woodson, Arkansas?"
]

for i, question in enumerate(test_questions, 1):
    print(f"\n📌 Test {i}: {question}")
    print("-" * 30)
    
    try:
        result = basic_rag(question)
        print(f"Answer: {result['answer']}")
        print(f"Retrieved chunks: {result['num_chunks']}")
        
        if result['retrieved_chunks']:
            first_doc = result['retrieved_chunks'][0]
            print(f"Source: {first_doc.metadata.get('title', 'Unknown')}")
    except Exception as e:
        print(f"❌ Error: {e}")

# ============================================
# Step 7: Test on HotpotQA questions
# ============================================
print("\n🔗 TESTING ON HOTPOTQA QUESTIONS")
print("-" * 40)

try:
    with open('hotpot_sample.json', 'r') as f:
        sample_questions = json.load(f)
    
    hotpot_results = []
    
    for i, q in enumerate(sample_questions[:5], 1):
        print(f"\n📌 HotpotQA Question {i}:")
        print(f"   Question: {q['question']}")
        print(f"   Expected Answer: {q['answer']}")
        
        try:
            result = basic_rag(q['question'])
            print(f"   RAG Answer: {result['answer'][:100]}...")
            print(f"   Retrieved chunks: {result['num_chunks']}")
            
            hotpot_results.append({
                "question": q['question'],
                "expected_answer": q['answer'],
                "rag_answer": result['answer'],
                "num_chunks": result['num_chunks']
            })
            
            expected = q['answer'].lower()
            generated = result['answer'].lower()
            
            if expected in generated or generated in expected:
                print("   ✅ Likely correct!")
            else:
                print("   ⚠️ May be incorrect")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            
except Exception as e:
    print(f"❌ Error testing on HotpotQA: {e}")

# ============================================
# Step 8: Save baseline results
# ============================================
print("\n💾 SAVING BASELINE RESULTS")
print("-" * 40)

baseline_data = {
    "timestamp": datetime.now().isoformat(),
    "model": "mistral (Ollama - free)",
    "temperature": 0,
    "retriever_k": 4,
    "vector_store": loaded_path,
    "test_results": hotpot_results,
    "notes": "Baseline RAG - FREE version using Ollama"
}

with open('baseline_results_free.json', 'w') as f:
    json.dump(baseline_data, f, indent=2)

print(f"✅ Saved baseline results to baseline_results_free.json")
print(f"   Tested on {len(hotpot_results)} questions")

# ============================================
# Step 9: Summary
# ============================================
print("\n📊 DAY 5 SUMMARY")
print("-" * 40)
print(f"✅ Created baseline RAG pipeline (FREE)")
print(f"✅ Using Ollama (local LLM) - no API costs!")
print(f"✅ Tested on {len(hotpot_results)} HotpotQA questions")
print(f"✅ Saved baseline results to baseline_results_free.json")

print("\n🎯 Key Takeaway:")
print("   ✅ This solution is COMPLETELY FREE!")
print("   ✅ No OpenAI credits needed!")
print("   ✅ Runs locally on your Mac!")

print("\n🚀 Ready for Week 2! (LangGraph Agent)")
print("=" * 60)