"""
Day 6: Test Baseline RAG on 20 HotpotQA Questions
This script tests our basic RAG pipeline and saves results.
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
import json
import os
from datetime import datetime
import time

print("=" * 60)
print("📊 DAY 6: TESTING BASELINE RAG ON 20 QUESTIONS")
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
print("✅ Retriever created!")

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
    
except Exception as e:
    print(f"❌ Error initializing Ollama: {e}")
    print("\n💡 Make sure Ollama is running: 'ollama serve'")
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
        "context": context,
        "sources": [doc.metadata.get('title', 'Unknown') for doc in docs]
    }

print("✅ RAG pipeline ready!")

# ============================================
# Step 6: Load HotpotQA questions
# ============================================
print("\n📂 Loading HotpotQA questions...")

try:
    # Load the sample questions from Day 1
    with open('hotpot_sample.json', 'r') as f:
        all_questions = json.load(f)
    
    # Take first 20 questions
    test_questions = all_questions[:20]
    
    print(f"✅ Loaded {len(test_questions)} questions for testing")
    
except Exception as e:
    print(f"❌ Error loading questions: {e}")
    print("💡 Please run load_dataset.py first")
    exit(1)

# ============================================
# Step 7: Test on all 20 questions
# ============================================
print("\n🧪 TESTING ON 20 HOTPOTQA QUESTIONS")
print("-" * 60)

results = []
correct = 0
total = len(test_questions)

print(f"Total questions: {total}")
print("\nProcessing questions...\n")

for i, q in enumerate(test_questions, 1):
    question = q['question']
    expected_answer = q['answer']
    
    print(f"📌 Question {i}/{total}:")
    print(f"   Q: {question[:80]}...")
    print(f"   Expected: {expected_answer}")
    
    try:
        # Get RAG answer
        result = basic_rag(question)
        rag_answer = result['answer']
        
        print(f"   RAG Answer: {rag_answer[:80]}...")
        print(f"   Sources: {', '.join(result['sources'][:2])}")
        
        # Check if answer is correct (simple check)
        expected_lower = expected_answer.lower()
        generated_lower = rag_answer.lower()
        
        # Check if expected answer appears in generated answer
        if expected_lower in generated_lower or generated_lower in expected_lower:
            is_correct = True
            correct += 1
            print("   ✅ CORRECT!")
        else:
            is_correct = False
            print("   ❌ INCORRECT")
        
        print()
        
        # Store result
        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "rag_answer": rag_answer,
            "sources": result['sources'],
            "num_chunks": result['num_chunks'],
            "is_correct": is_correct
        })
        
        # Small delay to avoid overloading
        time.sleep(0.5)
        
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "rag_answer": f"ERROR: {str(e)}",
            "sources": [],
            "num_chunks": 0,
            "is_correct": False
        })

# ============================================
# Step 8: Calculate metrics
# ============================================
print("\n📊 CALCULATING METRICS")
print("-" * 40)

accuracy = correct / total if total > 0 else 0

print(f"Total questions: {total}")
print(f"Correct: {correct}")
print(f"Incorrect: {total - correct}")
print(f"Accuracy: {accuracy:.1%}")

# Calculate average sources per answer
avg_sources = sum(r['num_chunks'] for r in results) / total if total > 0 else 0
print(f"Average sources retrieved: {avg_sources:.1f}")

# ============================================
# Step 9: Save results
# ============================================
print("\n💾 SAVING BASELINE RESULTS")
print("-" * 40)

# Prepare data for saving
baseline_data = {
    "timestamp": datetime.now().isoformat(),
    "model": "mistral (Ollama - free)",
    "temperature": 0,
    "retriever_k": 4,
    "vector_store": loaded_path,
    "total_questions": total,
    "correct": correct,
    "incorrect": total - correct,
    "accuracy": accuracy,
    "avg_sources": avg_sources,
    "test_results": results,
    "notes": "Baseline RAG - no agent, no self-correction"
}

# Save to file
with open('baseline_results_20.json', 'w') as f:
    json.dump(baseline_data, f, indent=2)

print(f"✅ Saved baseline results to baseline_results_20.json")

# ============================================
# Step 10: Show detailed results
# ============================================
print("\n📋 DETAILED RESULTS")
print("-" * 40)

print("\nCorrect answers:")
correct_count = 0
incorrect_count = 0

for i, r in enumerate(results, 1):
    if r['is_correct']:
        correct_count += 1
        print(f"   {i}. ✅ {r['expected_answer']}")
    else:
        incorrect_count += 1
        print(f"   {i}. ❌ Expected: {r['expected_answer']} | Got: {r['rag_answer'][:50]}...")

print(f"\n✅ Correct: {correct_count}")
print(f"❌ Incorrect: {incorrect_count}")

# ============================================
# Step 11: Summary
# ============================================
print("\n📊 DAY 6 SUMMARY")
print("-" * 40)
print(f"✅ Tested baseline RAG on {total} questions")
print(f"✅ Accuracy: {accuracy:.1%}")
print(f"✅ Correct: {correct} | Incorrect: {total - correct}")
print(f"✅ Results saved to baseline_results_20.json")

print("\n🎯 Key Takeaway:")
print("   This is our BASELINE measurement.")
print("   We'll compare this against our AGENTIC system.")
print("   The agent should perform BETTER than this!")

print("\n📊 WHAT'S NEXT:")
print("   Day 7: Commit and push to GitHub")
print("   Week 2: Build the agentic loop with LangGraph")
print("   Week 3: Add self-correction with LLM-as-Judge")
print("   Week 4: Compare agentic results vs this baseline")

print("\n🚀 Ready for Day 7! (Commit and Push)")
print("=" * 60)