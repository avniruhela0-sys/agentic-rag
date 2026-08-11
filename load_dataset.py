"""
Day 1: Load HotpotQA Dataset
This script loads the HotpotQA benchmark dataset and explores its structure.
"""

from datasets import load_dataset
import json
from datetime import datetime

print("=" * 60)
print("🚀 DAY 1: LOADING HOTPOTQA DATASET")
print("=" * 60)

# ============================================
# Step 1: Load the dataset
# ============================================
print("\n📥 Loading HotpotQA dataset...")
print("   (This may take 30-60 seconds to download)")

try:
    # Load the distractor version (harder - has irrelevant docs)
    # Using validation split (200 samples for testing)
    hotpot = load_dataset(
        "hotpot_qa", 
        "distractor", 
        split="validation[:200]"
    )
    print("✅ Dataset loaded successfully!")
    
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("\n💡 Troubleshooting tips:")
    print("   1. Check your internet connection")
    print("   2. Make sure datasets library is installed")
    print("   3. Try: pip install datasets --upgrade")
    exit(1)

# ============================================
# Step 2: Explore dataset structure
# ============================================
print("\n📊 DATASET INFO")
print("-" * 40)
print(f"Number of questions: {len(hotpot)}")
print(f"Dataset features: {hotpot.features.keys()}")

# ============================================
# Step 3: Examine first question
# ============================================
print("\n📝 FIRST QUESTION")
print("-" * 40)

first = hotpot[0]
print(f"Question: {first['question']}")
print(f"Answer: {first['answer']}")
print(f"Number of supporting documents: {len(first['context']['sentences'])}")

# Show document titles
print("\n📄 Supporting Documents:")
for i, title in enumerate(first['context']['title'][:3], 1):
    print(f"   {i}. {title}")

# ============================================
# Step 4: Understand the structure
# ============================================
print("\n🔍 UNDERSTANDING THE DATA STRUCTURE")
print("-" * 40)

sample = hotpot[0]

print("\n📋 What's in each example:")
print(f"   - 'question': The question to answer")
print(f"   - 'answer': The correct answer (ground truth)")
print(f"   - 'context': Dictionary containing:")
print(f"       - 'title': List of document titles")
print(f"       - 'sentences': List of sentence lists (one per document)")

print("\n📋 Example of a supporting document:")
doc_title = sample['context']['title'][0]
doc_sentences = sample['context']['sentences'][0][:2]  # First 2 sentences
print(f"   Title: {doc_title}")
print(f"   First sentences: {' '.join(doc_sentences)}")

# ============================================
# Step 5: Show a sample of 3 questions
# ============================================
print("\n📋 SAMPLE OF 3 QUESTIONS")
print("-" * 40)

for i in range(min(3, len(hotpot))):
    print(f"\n📌 Question {i+1}:")
    print(f"   {hotpot[i]['question']}")
    print(f"   Answer: {hotpot[i]['answer']}")

# ============================================
# Step 6: Save sample for later use
# ============================================
print("\n💾 SAVING SAMPLE DATA")
print("-" * 40)

# Save first 20 questions for baseline testing (Day 6)
sample_data = []
for i in range(20):
    sample_data.append({
        "question": hotpot[i]['question'],
        "answer": hotpot[i]['answer'],
        "context": hotpot[i]['context']
    })

# Save to file
with open('hotpot_sample.json', 'w') as f:
    json.dump(sample_data, f, indent=2)

print("✅ Saved 20 questions to hotpot_sample.json")

# ============================================
# Step 7: Summary
# ============================================
print("\n📊 DAY 1 SUMMARY")
print("-" * 40)
print(f"✅ Loaded {len(hotpot)} questions from HotpotQA")
print(f"✅ Saved 20 sample questions for testing")
print(f"✅ Dataset is ready for Day 2")

print("\n🎯 Key Takeaway:")
print("   This is a REAL benchmark dataset, not synthetic data.")
print("   This means our evaluation results will be credible!")
print("   (Recruiters love this because they can verify our results)")

print("\n🚀 Ready for Day 2!")
print("=" * 60)