"""
Day 2: Build Document Corpus from HotpotQA
This script converts HotpotQA data into LangChain Documents.
"""

from langchain.schema import Document
from datasets import load_dataset
import json
import os

print("=" * 60)
print("📚 DAY 2: BUILDING DOCUMENT CORPUS")
print("=" * 60)

# ============================================
# Step 1: Load the dataset
# ============================================
print("\n📥 Loading HotpotQA dataset...")

try:
    # First check if dataset is already loaded from Day 1
    # If not, load it fresh
    try:
        hotpot = load_dataset("hotpot_qa", "distractor", split="validation[:200]")
        print("✅ Dataset loaded successfully!")
        print(f"   Total questions: {len(hotpot)}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("💡 Trying to load from cached version...")
        # Try loading from the sample file from Day 1
        if os.path.exists('hotpot_sample.json'):
            with open('hotpot_sample.json', 'r') as f:
                sample_data = json.load(f)
            print("✅ Loaded sample data from hotpot_sample.json")
            # Convert sample data back to format we need
            # We'll create a custom dataset object
            class CustomDataset:
                def __init__(self, data):
                    self.data = data
                def __getitem__(self, idx):
                    return self.data[idx]
                def __len__(self):
                    return len(self.data)
            hotpot = CustomDataset(sample_data)
            print(f"   Total questions: {len(hotpot)}")
        else:
            print("❌ No dataset found. Please run Day 1 first!")
            exit(1)

except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit(1)

# ============================================
# Step 2: Understand the data structure
# ============================================
print("\n🔍 EXAMINING DATA STRUCTURE")
print("-" * 40)

# Look at one example
sample = hotpot[0]
print(f"Keys in each entry: {sample.keys()}")

# Check if it's the full dataset or our custom one
if hasattr(sample, 'get'):
    # This is a dictionary-like object
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer']}")
    print(f"Number of context entries: {len(sample['context']['title'])}")
else:
    # This might be the custom dataset
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer']}")
    print(f"Number of context entries: {len(sample['context']['title'])}")

# ============================================
# Step 3: Build documents from the dataset
# ============================================
print("\n📄 BUILDING DOCUMENTS FROM DATASET")
print("-" * 40)

docs = []
total_docs = 0

# Process each question
for idx, row in enumerate(hotpot):
    # Get the context (supporting documents)
    try:
        titles = row["context"]["title"]
        sentences_list = row["context"]["sentences"]
        
        # Process each document in the context
        for title, sentences in zip(titles, sentences_list):
            # Join sentences into a single text
            text = " ".join(sentences)
            
            # Create a LangChain Document
            doc = Document(
                page_content=text,
                metadata={
                    "title": title,
                    "source_question_index": idx,
                    "source_question": row["question"][:100],  # Store question (truncated)
                    "answer": row["answer"] if "answer" in row else "Unknown",
                    "document_index": total_docs
                }
            )
            docs.append(doc)
            total_docs += 1
            
    except KeyError as e:
        print(f"⚠️ Missing key in row {idx}: {e}")
        continue
    except Exception as e:
        print(f"⚠️ Error processing row {idx}: {e}")
        continue

print(f"✅ Created {len(docs)} documents")

# ============================================
# Step 4: Analyze the corpus
# ============================================
print("\n📊 CORPUS ANALYSIS")
print("-" * 40)

# Count unique titles
unique_titles = set()
for doc in docs:
    unique_titles.add(doc.metadata["title"])

print(f"Total documents: {len(docs)}")
print(f"Unique document titles: {len(unique_titles)}")

# Calculate average document length
total_chars = sum(len(doc.page_content) for doc in docs)
avg_chars = total_chars / len(docs) if docs else 0

print(f"Average document length: {avg_chars:.0f} characters")
print(f"Total characters in corpus: {total_chars:,}")

# Show document length distribution
print("\n📊 Document length distribution:")
length_ranges = [
    (0, 100, "Very short"),
    (100, 500, "Short"),
    (500, 1000, "Medium"),
    (1000, 2000, "Long"),
    (2000, float('inf'), "Very long")
]

for min_len, max_len, label in length_ranges:
    count = sum(1 for doc in docs if min_len <= len(doc.page_content) < max_len)
    if count > 0:
        print(f"   {label}: {count} documents")

# ============================================
# Step 5: Show example documents
# ============================================
print("\n📝 EXAMPLE DOCUMENTS")
print("-" * 40)

# Show 3 example documents
for i in range(min(3, len(docs))):
    doc = docs[i]
    print(f"\n📌 Document {i+1}:")
    print(f"   Title: {doc.metadata['title']}")
    print(f"   Length: {len(doc.page_content)} characters")
    print(f"   Source Question: {doc.metadata['source_question']}...")
    print(f"   Preview: {doc.page_content[:150]}...")

# ============================================
# Step 6: Show a document with its question
# ============================================
print("\n🔗 EXAMPLE: DOCUMENT WITH ITS QUESTION")
print("-" * 40)

# Find a document and show its question
for doc in docs:
    if "source_question" in doc.metadata:
        print(f"Question: {doc.metadata['source_question']}...")
        print(f"Answer: {doc.metadata.get('answer', 'Unknown')}")
        print(f"Document Title: {doc.metadata['title']}")
        print(f"Document Content Preview: {doc.page_content[:200]}...")
        break

# ============================================
# Step 7: Save the documents for Day 3
# ============================================
print("\n💾 SAVING DOCUMENTS FOR DAY 3")
print("-" * 40)

# Convert documents to serializable format
docs_data = []
for doc in docs:
    docs_data.append({
        "page_content": doc.page_content,
        "metadata": doc.metadata
    })

# Save to file
with open('hotpot_docs.json', 'w') as f:
    json.dump(docs_data, f, indent=2)

print(f"✅ Saved {len(docs)} documents to hotpot_docs.json")

# ============================================
# Step 8: Save a small sample for testing
# ============================================
print("\n💾 Saving sample documents (first 10) for quick testing...")

sample_docs = docs_data[:10]
with open('sample_docs.json', 'w') as f:
    json.dump(sample_docs, f, indent=2)

print("✅ Saved 10 sample documents to sample_docs.json")

# ============================================
# Step 9: Summary
# ============================================
print("\n📊 DAY 2 SUMMARY")
print("-" * 40)
print(f"✅ Created {len(docs)} documents from {len(hotpot)} questions")
print(f"✅ Unique document titles: {len(unique_titles)}")
print(f"✅ Average document length: {avg_chars:.0f} characters")
print(f"✅ Documents saved to hotpot_docs.json")
print(f"✅ Sample documents saved to sample_docs.json")

print("\n🎯 Key Takeaway:")
print("   We now have a proper document corpus ready for chunking.")
print("   Each document has metadata (title, source question, answer).")
print("   This metadata will help trace answers back to sources.")

print("\n📋 DOCUMENT STRUCTURE:")
print("   page_content: The text of the document")
print("   metadata: {'title', 'source_question_index', 'source_question', 'answer'}")

print("\n🚀 Ready for Day 3! (Chunking)")
print("=" * 60)