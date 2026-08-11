"""
Day 3: Chunking Documents
This script splits documents into smaller chunks for vector storage.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import json
import os

print("=" * 60)
print("🔪 DAY 3: CHUNKING DOCUMENTS")
print("=" * 60)

# ============================================
# Step 1: Load documents from Day 2
# ============================================
print("\n📂 Loading documents from Day 2...")

try:
    # Check if documents exist
    if not os.path.exists('hotpot_docs.json'):
        print("❌ hotpot_docs.json not found!")
        print("💡 Please run day2_build_corpus.py first")
        exit(1)
    
    # Load the documents
    with open('hotpot_docs.json', 'r') as f:
        docs_data = json.load(f)
    
    # Convert to LangChain Documents
    docs = []
    for doc_data in docs_data:
        doc = Document(
            page_content=doc_data['page_content'],
            metadata=doc_data['metadata']
        )
        docs.append(doc)
    
    print(f"✅ Loaded {len(docs)} documents")
    
except FileNotFoundError:
    print("❌ hotpot_docs.json not found!")
    print("💡 Please run day2_build_corpus.py first")
    exit(1)

# ============================================
# Step 2: Understand why chunking matters
# ============================================
print("\n📚 WHY CHUNKING MATTERS")
print("-" * 40)
print("1. LLMs have token limits (e.g., 4,096 tokens)")
print("2. Retrieval works better with smaller chunks")
print("3. Small chunks = more precise retrieval")
print("4. Overlap prevents information loss at boundaries")

print("\n🔧 CHUNK PARAMETERS:")
print("   - chunk_size: 500 characters")
print("   - chunk_overlap: 50 characters")
print("   - separator: Recursive splitting (paragraphs, sentences, words)")

# ============================================
# Step 3: Create the text splitter
# ============================================
print("\n⚙️ Creating text splitter...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Each chunk is about 500 characters
    chunk_overlap=50,    # 50 characters overlap between chunks
    length_function=len, # Count characters
    separators=[         # Split in this order
        "\n\n",          # Paragraphs first
        "\n",            # Then newlines
        ". ",            # Then sentences
        ", ",            # Then clauses
        " ",             # Then words
        ""               # Then characters (last resort)
    ]
)

print("✅ Text splitter created!")

# ============================================
# Step 4: Split documents into chunks
# ============================================
print("\n🔪 Splitting documents into chunks...")

try:
    chunks = text_splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks from {len(docs)} documents")
except Exception as e:
    print(f"❌ Error during splitting: {e}")
    exit(1)

# ============================================
# Step 5: Analyze the chunks
# ============================================
print("\n📊 CHUNK ANALYSIS")
print("-" * 40)

# Calculate average chunk size
total_chars = sum(len(chunk.page_content) for chunk in chunks)
avg_chars = total_chars / len(chunks)

print(f"Total chunks: {len(chunks)}")
print(f"Average chunk size: {avg_chars:.0f} characters")
print(f"Smallest chunk: {min(len(c.page_content) for c in chunks)} characters")
print(f"Largest chunk: {max(len(c.page_content) for c in chunks)} characters")

# Show chunk size distribution
print("\n📊 Chunk size distribution:")
size_ranges = [
    (0, 100, "Very small"),
    (100, 200, "Small"),
    (200, 400, "Medium"),
    (400, 600, "Large"),
    (600, 800, "Very large"),
    (800, float('inf'), "Huge")
]

for min_size, max_size, label in size_ranges:
    count = sum(1 for c in chunks if min_size <= len(c.page_content) < max_size)
    if count > 0:
        print(f"   {label}: {count} chunks")

# ============================================
# Step 6: Show example chunks
# ============================================
print("\n📝 EXAMPLE CHUNKS")
print("-" * 40)

# Show 3 example chunks
for i in range(min(3, len(chunks))):
    chunk = chunks[i]
    print(f"\n📌 Chunk {i+1}:")
    print(f"   Length: {len(chunk.page_content)} characters")
    print(f"   Source: {chunk.metadata.get('title', 'Unknown')}")
    print(f"   Preview: {chunk.page_content[:150]}...")
    
    # Show overlap info
    if i > 0:
        prev_chunk = chunks[i-1]
        print("   Note: This chunk has overlap with previous chunk")

# ============================================
# Step 7: Save chunks for Day 4
# ============================================
print("\n💾 SAVING CHUNKS FOR DAY 4")
print("-" * 40)

# Convert chunks to serializable format
chunks_data = []
for chunk in chunks:
    chunks_data.append({
        "page_content": chunk.page_content,
        "metadata": chunk.metadata
    })

# Save to file
with open('hotpot_chunks.json', 'w') as f:
    json.dump(chunks_data, f, indent=2)

print(f"✅ Saved {len(chunks)} chunks to hotpot_chunks.json")

# ============================================
# Step 8: Save a small sample for testing
# ============================================
print("\n💾 Saving sample chunks (first 10) for quick testing...")

sample_chunks = chunks_data[:10]
with open('sample_chunks.json', 'w') as f:
    json.dump(sample_chunks, f, indent=2)

print("✅ Saved 10 sample chunks to sample_chunks.json")

# ============================================
# Step 9: Summary
# ============================================
print("\n📊 DAY 3 SUMMARY")
print("-" * 40)
print(f"✅ Created {len(chunks)} chunks from {len(docs)} documents")
print(f"✅ Average chunk size: {avg_chars:.0f} characters")
print(f"✅ Chunks saved to hotpot_chunks.json")
print(f"✅ Sample chunks saved to sample_chunks.json")

print("\n🎯 Key Takeaway:")
print("   Chunking makes documents searchable and retrievable.")
print("   Good chunking = better retrieval = better answers!")

print("\n📊 CHUNK STATISTICS:")
print(f"   Total chunks: {len(chunks)}")
print(f"   Average length: {avg_chars:.0f} chars")
print(f"   Chunk size range: 100-500 chars (mostly)")

print("\n🚀 Ready for Day 4! (Embeddings + Vector Store)")
print("=" * 60)