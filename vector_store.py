"""
Embeddings + Chroma Vector Store - HUGGINGFACE (FREE)
This script uses free HuggingFace embeddings - no OpenAI credits needed!
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import json
import os
import shutil

print("=" * 60)
print("🧠 DAY 4: EMBEDDINGS + VECTOR STORE (FREE - HuggingFace)")
print("=" * 60)

# ============================================
# Step 1: Load chunks
# ============================================
print("\n📂 Loading chunks...")

try:
    if not os.path.exists('hotpot_chunks.json'):
        print("❌ hotpot_chunks.json not found!")
        print("💡 Please run chunking.py first")
        exit(1)
    
    with open('hotpot_chunks.json', 'r') as f:
        chunks_data = json.load(f)
    
    # Use only first 100 chunks to save time
    chunks_data = chunks_data[:100]
    
    chunks = []
    for chunk_data in chunks_data:
        doc = Document(
            page_content=chunk_data['page_content'],
            metadata=chunk_data['metadata']
        )
        chunks.append(doc)
    
    print(f"✅ Loaded {len(chunks)} chunks (using 100 for speed)")
    
except FileNotFoundError:
    print("❌ hotpot_chunks.json not found!")
    print("💡 Please run chunking.py first")
    exit(1)

# ============================================
# Step 2: Initialize FREE HuggingFace embeddings
# ============================================
print("\n⚙️ Initializing HuggingFace Embeddings (FREE)...")
print("   Downloading model (first time only, may take 1-2 minutes)...")

try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ Embeddings initialized!")
    print("   Model: all-MiniLM-L6-v2 (384 dimensions)")
    print("   ✅ This is FREE! No OpenAI credits needed!")
    
except Exception as e:
    print(f"❌ Error initializing embeddings: {e}")
    exit(1)

# ============================================
# Step 3: Create ChromaDB vector store
# ============================================
print("\n🗄️ Creating ChromaDB vector store...")
print("   (This may take 1-2 minutes)")

persist_dir = "chroma_db_hf"

if os.path.exists(persist_dir):
    print("   Removing existing vector store...")
    shutil.rmtree(persist_dir)

try:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectorstore.persist()
    print("✅ Vector store created successfully!")
    
except Exception as e:
    print(f"❌ Error creating vector store: {e}")
    exit(1)

# ============================================
# Step 4: Create retriever
# ============================================
print("\n🔍 Creating retriever...")

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("✅ Retriever created! Will return top 4 chunks")

# ============================================
# Step 5: Test retrieval
# ============================================
print("\n🧪 TESTING RETRIEVAL")
print("-" * 40)

test_queries = [
    "What is the capital of France?",
    "Who is the director of Ed Wood?",
    "What is the population of Woodson, Arkansas?"
]

for query in test_queries:
    print(f"\n📌 Query: {query}")
    try:
        results = retriever.invoke(query)
        print(f"   Retrieved {len(results)} chunks")
        if results:
            print(f"   Top chunk preview: {results[0].page_content[:100]}...")
            print(f"   Source: {results[0].metadata.get('title', 'Unknown')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

# ============================================
# Step 6: Test with HotpotQA question
# ============================================
print("\n🔗 TESTING WITH HOTPOTQA QUESTION")
print("-" * 40)

try:
    with open('hotpot_sample.json', 'r') as f:
        sample_questions = json.load(f)
    
    if sample_questions:
        test_question = sample_questions[0]['question']
        test_answer = sample_questions[0]['answer']
        
        print(f"Question: {test_question}")
        print(f"Expected Answer: {test_answer}")
        print()
        
        results = retriever.invoke(test_question)
        
        print("📄 Retrieved documents:")
        for i, doc in enumerate(results, 1):
            print(f"\n   {i}. Title: {doc.metadata.get('title', 'Unknown')}")
            print(f"      Preview: {doc.page_content[:100]}...")
            
except Exception as e:
    print(f"❌ Error testing with HotpotQA: {e}")

# ============================================
# Step 7: Save config
# ============================================
print("\n💾 SAVING CONFIGURATION")

config = {
    "vector_store_path": persist_dir,
    "embedding_model": "all-MiniLM-L6-v2",
    "chunk_count": len(chunks),
    "retriever_k": 4,
    "free": True,
    "model_type": "HuggingFace"
}

with open('vector_store_config_hf.json', 'w') as f:
    json.dump(config, f, indent=2)

print("✅ Saved config to vector_store_config_hf.json")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("📊 DAY 4 SUMMARY")
print("=" * 60)
print(f"✅ Created vector store with {len(chunks)} chunks")
print(f"✅ Embedding model: all-MiniLM-L6-v2 (384 dimensions)")
print(f"✅ Vector store saved to {persist_dir}/")
print(f"✅ Retriever created (k=4)")

print("\n🎯 Key Takeaway:")
print("   ✅ This solution is COMPLETELY FREE!")
print("   ✅ No OpenAI credits needed!")
print("   ✅ Quality is slightly lower but perfect for learning")

print("\n🚀 Ready for Day 5! (Basic RAG)")
print("=" * 60)