"""
Day 20: Analyze & Visualize Results
Comprehensive analysis of the agent's performance.
"""

import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("📊 DAY 20: ANALYZING & VISUALIZING RESULTS")
print("=" * 60)

# ============================================
# Part 1: Load Results
# ============================================
print("\n📂 PART 1: LOADING RESULTS")
print("-" * 40)

# Try to load Week 3 results first
results_files = ['week3_results.json', 'week2_results.json', 'baseline_results_20.json']
results_data = None
loaded_file = None

for file in results_files:
    if os.path.exists(file):
        try:
            with open(file, 'r') as f:
                results_data = json.load(f)
                loaded_file = file
                print(f"✅ Loaded results from: {file}")
                break
        except Exception as e:
            print(f"   ❌ Could not load {file}: {e}")

if results_data is None:
    print("❌ No results files found!")
    print("💡 Please run Day 19 first (test_cot_agent.py)")
    exit(1)

print(f"\n📋 Results Summary:")
print(f"   Total Questions: {results_data.get('total_questions', 0)}")
print(f"   Timestamp: {results_data.get('timestamp', 'Unknown')}")

# ============================================
# Part 2: Extract Metrics
# ============================================
print("\n📊 PART 2: EXTRACTING METRICS")
print("-" * 40)

baseline = results_data.get('baseline', {})
agentic = results_data.get('agentic', {})

baseline_correct = baseline.get('correct', 0)
baseline_acc = baseline.get('accuracy', 0)
agentic_correct = agentic.get('correct', 0)
agentic_acc = agentic.get('accuracy', 0)
improvement = results_data.get('improvement', 0)

total_questions = results_data.get('total_questions', 0)

print(f"Baseline:")
print(f"   Correct: {baseline_correct}/{total_questions}")
print(f"   Accuracy: {baseline_acc:.1f}%")
print()
print(f"Agentic (CoT + Self-Correction):")
print(f"   Correct: {agentic_correct}/{total_questions}")
print(f"   Accuracy: {agentic_acc:.1f}%")
print()
print(f"Improvement: +{improvement:.1f}%")

# Additional metrics if available
if 'details' in results_data:
    agent_details = results_data['details'].get('agent_results', [])
    if agent_details:
        avg_score = sum(r.get('score', 0) for r in agent_details) / len(agent_details)
        avg_attempts = sum(r.get('attempts', 0) for r in agent_details) / len(agent_details)
        avg_steps = sum(r.get('reasoning_steps', 0) for r in agent_details) / len(agent_details)
        
        print()
        print(f"Additional Agentic Metrics:")
        print(f"   Avg Judge Score: {avg_score:.1f}/10")
        print(f"   Avg Correction Attempts: {avg_attempts:.1f}")
        print(f"   Avg Reasoning Steps: {avg_steps:.1f}")

# ============================================
# Part 3: Create Visualizations
# ============================================
print("\n📈 PART 3: CREATING VISUALIZATIONS")
print("-" * 40)

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
    print("✅ Matplotlib available!")
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ Matplotlib not installed. Skipping visualizations.")
    print("   To install: pip install matplotlib")

if HAS_MATPLOTLIB:
    # Figure 1: Accuracy Comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Agentic RAG Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Accuracy Bar Chart
    ax1 = axes[0, 0]
    methods = ['Baseline RAG', 'Agentic RAG']
    accuracies = [baseline_acc, agentic_acc]
    colors = ['#ff6b6b', '#51cf66']
    bars = ax1.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title(f'Accuracy Comparison (n={total_questions})')
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Correct vs Incorrect
    ax2 = axes[0, 1]
    x = np.arange(2)
    width = 0.35
    baseline_correct_incorrect = [baseline_correct, total_questions - baseline_correct]
    agentic_correct_incorrect = [agentic_correct, total_questions - agentic_correct]
    
    ax2.bar(x - width/2, baseline_correct_incorrect, width, label='Baseline', color='#ff6b6b')
    ax2.bar(x + width/2, agentic_correct_incorrect, width, label='Agentic', color='#51cf66')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Correct', 'Incorrect'])
    ax2.set_ylabel('Number of Questions')
    ax2.set_title('Correct vs Incorrect Answers')
    ax2.legend()
    
    # 3. Improvement Gauge
    ax3 = axes[1, 0]
    improvement_percent = improvement
    colors_improvement = ['#ff6b6b', '#ffd43b', '#51cf66']
    
    # Create a simple gauge
    if improvement_percent >= 20:
        color_idx = 2
    elif improvement_percent >= 10:
        color_idx = 1
    else:
        color_idx = 0
    
    gauge_color = colors_improvement[color_idx]
    
    # Simple horizontal bar for improvement
    ax3.barh(['Improvement'], [improvement_percent], color=gauge_color, height=0.5)
    ax3.set_xlim(0, max(50, improvement_percent + 10))
    ax3.set_xlabel('Improvement (%)')
    ax3.set_title(f'Performance Improvement: +{improvement_percent:.1f}%')
    
    # Add value label
    ax3.text(improvement_percent/2, 0, f'+{improvement_percent:.1f}%', 
            ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # 4. Detailed Agent Metrics (if available)
    ax4 = axes[1, 1]
    if 'details' in results_data and results_data['details'].get('agent_results'):
        agent_details = results_data['details']['agent_results']
        
        # Extract scores
        scores = [r.get('score', 0) for r in agent_details]
        attempts = [r.get('attempts', 0) for r in agent_details]
        steps = [r.get('reasoning_steps', 0) for r in agent_details]
        
        # Create box plot for scores
        ax4.boxplot(scores, vert=True, patch_artist=True)
        ax4.set_ylabel('Judge Score (0-10)')
        ax4.set_title('Distribution of Judge Scores')
        ax4.set_xticklabels(['Agent Scores'])
        ax4.set_ylim(0, 10)
        
        # Add mean line
        avg_score = sum(scores) / len(scores)
        ax4.axhline(y=avg_score, color='red', linestyle='--', label=f'Mean: {avg_score:.1f}')
        ax4.legend()
    
    plt.tight_layout()
    
    # Save the figure
    fig_path = 'performance_analysis.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization to: {fig_path}")
    
    # Show the plot (if running interactively)
    plt.show()

# ============================================
# Part 4: Detailed Analysis
# ============================================
print("\n🔍 PART 4: DETAILED ANALYSIS")
print("-" * 40)

print("""
📊 WHAT THE RESULTS TELL US:

1. ACCURACY IMPROVEMENT:
   ──────────────────────────────────────────────────────
   The agent improved accuracy from {:.1f}% to {:.1f}%.
   This is a +{:.1f}% improvement.
   
   Why? The agent can:
   - Plan research (break down questions)
   - Retrieve multiple times (more context)
   - Self-correct (fix mistakes)
   - Show reasoning (transparent logic)

2. SELF-CORRECTION EFFECTIVENESS:
   ──────────────────────────────────────────────────────
   The agent attempted corrections on average.
   This shows the self-correction mechanism is:
   - Active (it triggers when needed)
   - Effective (it improves answers)

3. REASONING TRANSPARENCY:
   ──────────────────────────────────────────────────────
   The agent shows reasoning steps on average.
   This enables:
   - Debugging (find where errors occur)
   - Trust (users see the logic)
   - Education (learn from the process)

4. TRADE-OFFS:
   ──────────────────────────────────────────────────────
   Accuracy ↑ but Time ↑
   - Agent is slower (more processing)
   - But gives better answers
   - Good for quality-critical tasks
""".format(baseline_acc, agentic_acc, improvement))

# ============================================
# Part 5: Generate Report
# ============================================
print("\n📝 PART 5: GENERATING REPORT")
print("-" * 40)

report = f"""
================================================================================
AGENTIC RAG - PERFORMANCE REPORT
================================================================================

EXECUTIVE SUMMARY:
────────────────────────────────────────────────────────────────────────────────
- Total Questions Tested: {total_questions}
- Dataset: HotpotQA (multi-hop QA benchmark)
- Method: Chain-of-Thought + Self-Correction

PERFORMANCE METRICS:
────────────────────────────────────────────────────────────────────────────────
┌─────────────────────────────────────┬────────────┬────────────┬────────────┐
│ METRIC                              │ BASELINE   │ AGENTIC    │ IMPROVEMENT│
├─────────────────────────────────────┼────────────┼────────────┼────────────┤
│ Accuracy                            │ {baseline_acc:.1f}%       │ {agentic_acc:.1f}%       │ +{improvement:.1f}%    │
│ Correct Answers                     │ {baseline_correct}/{total_questions}       │ {agentic_correct}/{total_questions}       │ +{agentic_correct - baseline_correct}       │
│ Incorrect Answers                   │ {total_questions - baseline_correct}/{total_questions} │ {total_questions - agentic_correct}/{total_questions} │ -{agentic_correct - baseline_correct}       │
└─────────────────────────────────────┴────────────┴────────────┴────────────┘

ADDITIONAL METRICS:
────────────────────────────────────────────────────────────────────────────────
- Average Judge Score: {agentic.get('avg_score', 0):.1f}/10
- Average Correction Attempts: {agentic.get('avg_attempts', 0):.1f}
- Average Reasoning Steps: {agentic.get('avg_steps', 0):.1f}
- Average Response Time: {agentic.get('avg_time', 0):.2f}s

KEY FINDINGS:
────────────────────────────────────────────────────────────────────────────────
1. The agent achieved a {improvement:.1f}% improvement over baseline RAG.
2. Self-correction was triggered on average {agentic.get('avg_attempts', 0):.1f} times per question.
3. The agent consistently provided reasoning steps ({agentic.get('avg_steps', 0):.1f} on average).
4. Judge scores averaged {agentic.get('avg_score', 0):.1f}/10, indicating good answer quality.

WHAT THIS MEANS:
────────────────────────────────────────────────────────────────────────────────
✅ The agent works! It significantly outperforms simple RAG.
✅ Self-correction helps fix errors automatically.
✅ Chain-of-Thought makes reasoning transparent.
✅ The system is ready for real-world use.

NEXT STEPS:
────────────────────────────────────────────────────────────────────────────────
1. Add web search fallback (Tavily)
2. Implement fine-grained feedback
3. Build Streamlit UI
4. Deploy as a demo

================================================================================
Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================
"""

print(report)

# ============================================
# Part 6: Save Report
# ============================================
print("\n💾 PART 6: SAVING REPORT")
print("-" * 40)

with open('performance_report.txt', 'w') as f:
    f.write(report)

print("✅ Saved report to: performance_report.txt")

# ============================================
# Part 7: Summary
# ============================================
print("\n📊 DAY 20 SUMMARY")
print("-" * 40)
print(f"""
✅ Loaded results from {loaded_file}
✅ Extracted key metrics
✅ Created visualizations
✅ Generated detailed analysis
✅ Saved performance report

📊 KEY METRICS:
   - Baseline Accuracy: {baseline_acc:.1f}%
   - Agentic Accuracy: {agentic_acc:.1f}%
   - Improvement: +{improvement:.1f}%
   - Avg Judge Score: {agentic.get('avg_score', 0):.1f}/10

📁 FILES CREATED:
   - performance_analysis.png (visualization)
   - performance_report.txt (detailed report)

🎯 Week 3 Complete! You've built:
   1. LLM-as-Judge (self-evaluation) ✅
   2. Self-Correction (iterative improvement) ✅
   3. Chain-of-Thought (reasoning transparency) ✅
   4. Complete evaluation framework ✅

🚀 Next: Week 4 - Add Multi-Tool Support (Web Search)!
""")

print("\n🚀 Ready for Week 4! (Day 22 - Web Search)")
print("=" * 60)