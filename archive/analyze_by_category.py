import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Define question categories
QUESTION_CATEGORIES = {
    "Genetic Variant Interpretation": [
        "What does this genetic variant mean for me?",
        "Does this mean I will definitely have cancer?",
        "Does this genetic variant affect my cancer treatment?",
        "How might my genetic test results change over time?",
        "What is Lynch Syndrome?"
    ],
    
    "Inheritance Patterns": [
        "Is this variant something I inherited?",
        "Can only women can carry a BRCA inherited mutation?",
        "Can I give this to my kids?",
        "Can this variant skip a generation?",
        "What if I want to have children and have a hereditary cancer gene? What are my reproductive options?",
        "I have a BRCA pathogenic variant and I want to have children, what are my options?",
        "Why do some families with Lynch syndrome have more cases of cancer than others?"
    ],
    
    "Family Risk Assessment": [
        "Why should I share with family my genetic results?",
        "Who are my first-degree relatives?",
        "Should my family members get tested?",
        "Which of my relatives are at risk?",
        "Should I contact my male and female relatives?",
        "What if a family member doesn't want to get tested?",
        "How can I get my kids tested?",
        "At what age should my children get tested?",
        "Why would my relatives want to know if they have this? What can they do about it?",
        "I don't talk to my family/parents/sister/brother. How can I share this with them?",
        "Who do my family members call to have genetic testing?"
    ],
    
    "Gene-Specific Recommendations": [
        "What are the recommendations for my family members if I have a mutation in (specify gene: MSH2, MSH1, MSH6, PMS2, EPCAM/MSH2, BRCA1, BRCA2)?",
        "What types of cancers am I at risk for?",
        "What screening tests do you recommend?",
        "What steps can I take to manage my cancer risk if I have Lynch syndrome? (not specific to variant)",
        "What are the Risks and Benefits of Risk-Reducing Surgeries for Lynch Syndrome?",
        "What is my cancer risk if I have MSH2 or EPCAM- associated Lynch syndrome?",
        "What is my cancer risk if I have PMS2 Lynch syndrome?",
        "What is my cancer risk if I have MSH1 Lynch syndrome?",
        "What is my cancer risk if I have MSH6 Lynch syndrome?",
        "What is my cancer risk if I have BRCA2 Hereditary Breast and Ovarian Cancer syndrome?",
        "What is my cancer risk if I have BRCA1 Hereditary Breast and Ovarian Cancer syndrome?",
        "What are the surveillance and preventions I can take to reduce my risk of cancer or detecting cancer early if I have a EPCAM/MSH2 mutation?",
        "What are the surveillance and preventions I can take to reduce my risk of cancer or detecting cancer early if I have an MSH2 mutation?",
        "What are the surveillance and preventions I can take to reduce my risk of cancer or detecting cancer early if I have a BRCA mutation?"
    ],
    
    "Support and Resources": [
        "Is genetic testing for my family members covered by insurance?",
        "Will this affect my health insurance?",
        "People who test positive for a genetic mutation are they at risk of losing their health insurance?",
        "Does GINA cover life or disability insurance?",
        "Will my insurance cover testing for my parents/brother/sister?",
        "My [relative] doesn't have insurance. What should they do?",
        "How can I cope with this diagnosis?",
        "What if I feel overwhelmed?",
        "Is new research being done on my condition?",
        "How can I help others with my condition?",
        "Where can I find a genetic counselor?",
        "What other resources are available to help me?"
    ]
}

def categorize_questions(questions):
    """Categorize questions into predefined categories."""
    question_to_category = {}
    for category, question_list in QUESTION_CATEGORIES.items():
        for question in question_list:
            question_to_category[question] = category
    return question_to_category

def load_and_merge_data():
    """Load both RAG and raw results and merge them."""
    base_dir = Path(__file__).parent
    
    # Load data
    raw_df = pd.read_csv(base_dir / "qa_outputs" / "questions_answers_raw.csv")
    rag_df = pd.read_csv(base_dir / "qa_outputs" / "questions_answers_rag.csv")
    
    # Merge on Question
    merged_df = pd.merge(raw_df, rag_df, on='Question', suffixes=('_raw', '_rag'))
    
    # Add categories
    question_to_category = categorize_questions(merged_df['Question'].tolist())
    merged_df['Category'] = merged_df['Question'].map(question_to_category)
    
    return merged_df

def calculate_category_statistics(df):
    """Calculate statistics for each category."""
    metrics = ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 
               'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity']
    
    results = {}
    
    for category in QUESTION_CATEGORIES.keys():
        category_df = df[df['Category'] == category]
        if len(category_df) == 0:
            continue
            
        category_results = {
            'n_questions': len(category_df),
            'metrics': {}
        }
        
        for metric in metrics:
            raw_values = category_df[f'{metric}_raw'].values
            rag_values = category_df[f'{metric}_rag'].values
            
            # Calculate basic statistics
            raw_mean = np.mean(raw_values)
            rag_mean = np.mean(rag_values)
            improvement = rag_mean - raw_mean
            improvement_pct = (improvement / raw_mean) * 100 if raw_mean != 0 else 0
            
            # Statistical tests
            t_stat, p_value = stats.ttest_rel(rag_values, raw_values)
            
            # Effect size (Cohen's d)
            diff = rag_values - raw_values
            cohens_d = np.mean(diff) / np.std(diff, ddof=1)
            
            # Count improvements
            improvements = np.sum(rag_values > raw_values) if metric not in ['flesch_reading_ease'] else np.sum(rag_values < raw_values)
            
            category_results['metrics'][metric] = {
                'raw_mean': raw_mean,
                'rag_mean': rag_mean,
                'improvement': improvement,
                'improvement_pct': improvement_pct,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05,
                'improvements': improvements,
                'total': len(category_df)
            }
        
        results[category] = category_results
    
    return results

def create_category_comparison_plots(df):
    """Create visualizations comparing categories."""
    # Create output directory
    plots_dir = Path(__file__).parent / "analysis_results" / "category_plots"
    plots_dir.mkdir(exist_ok=True)
    
    metrics = ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 'flesch_kincaid_grade']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Prepare data for plotting
        categories = []
        raw_values = []
        rag_values = []
        
        for category in QUESTION_CATEGORIES.keys():
            category_df = df[df['Category'] == category]
            if len(category_df) > 0:
                categories.append(category.replace(' ', '\n'))
                raw_values.append(category_df[f'{metric}_raw'].mean())
                rag_values.append(category_df[f'{metric}_rag'].mean())
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, raw_values, width, label='Raw', alpha=0.8, color='lightcoral')
        bars2 = ax.bar(x + width/2, rag_values, width, label='RAG', alpha=0.8, color='skyblue')
        
        ax.set_xlabel('Question Categories')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} by Category')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "category_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_improvement_heatmap(category_stats):
    """Create a heatmap showing improvements by category and metric."""
    plots_dir = Path(__file__).parent / "analysis_results" / "category_plots"
    plots_dir.mkdir(exist_ok=True)
    
    metrics = ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 
               'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity']
    
    # Create improvement matrix
    categories = list(category_stats.keys())
    improvement_matrix = []
    
    for category in categories:
        row = []
        for metric in metrics:
            if metric in category_stats[category]['metrics']:
                improvement_pct = category_stats[category]['metrics'][metric]['improvement_pct']
                row.append(improvement_pct)
            else:
                row.append(0)
        improvement_matrix.append(row)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(improvement_matrix, 
                xticklabels=[m.replace('_', ' ').title() for m in metrics],
                yticklabels=[c.replace(' ', '\n') for c in categories],
                annot=True, fmt='.1f', cmap='RdYlBu_r', center=0,
                cbar_kws={'label': 'Improvement %'})
    
    plt.title('RAG vs Raw Performance Improvement by Category (%)')
    plt.xlabel('Metrics')
    plt.ylabel('Question Categories')
    plt.tight_layout()
    plt.savefig(plots_dir / "improvement_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_category_report(category_stats):
    """Generate a detailed report by category."""
    report_path = Path(__file__).parent / "analysis_results" / "category_analysis_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("CGT CHATBOT ANALYSIS BY QUESTION CATEGORY\n")
        f.write("=" * 50 + "\n\n")
        
        for category, stats in category_stats.items():
            f.write(f"CATEGORY: {category.upper()}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Number of Questions: {stats['n_questions']}\n\n")
            
            # Sort metrics by improvement
            metrics_sorted = sorted(stats['metrics'].items(), 
                                  key=lambda x: x[1]['improvement_pct'], reverse=True)
            
            f.write("PERFORMANCE METRICS:\n")
            for metric, data in metrics_sorted:
                f.write(f"\n{metric.replace('_', ' ').title()}:\n")
                f.write(f"  Raw Mean: {data['raw_mean']:.3f}\n")
                f.write(f"  RAG Mean: {data['rag_mean']:.3f}\n")
                f.write(f"  Improvement: {data['improvement']:+.3f} ({data['improvement_pct']:+.1f}%)\n")
                f.write(f"  P-value: {data['p_value']:.6f}\n")
                f.write(f"  Cohen's d: {data['cohens_d']:.3f}\n")
                f.write(f"  Significant: {'Yes' if data['significant'] else 'No'}\n")
                f.write(f"  Questions Improved: {data['improvements']}/{data['total']}\n")
            
            f.write("\n" + "=" * 50 + "\n\n")
    
    print(f"Category analysis report saved to: {report_path}")

def main():
    """Main analysis function."""
    print("Loading data...")
    df = load_and_merge_data()
    
    print("Calculating category statistics...")
    category_stats = calculate_category_statistics(df)
    
    print("Creating visualizations...")
    create_category_comparison_plots(df)
    create_improvement_heatmap(category_stats)
    
    print("Generating report...")
    generate_category_report(category_stats)
    
    # Print summary to console
    print("\nCATEGORY ANALYSIS SUMMARY:")
    print("=" * 40)
    
    for category, stats in category_stats.items():
        print(f"\n{category} ({stats['n_questions']} questions):")
        
        # Find best and worst performing metrics
        best_metric = max(stats['metrics'].items(), key=lambda x: x[1]['improvement_pct'])
        worst_metric = min(stats['metrics'].items(), key=lambda x: x[1]['improvement_pct'])
        
        print(f"  Best improvement: {best_metric[0]} ({best_metric[1]['improvement_pct']:+.1f}%)")
        print(f"  Worst change: {worst_metric[0]} ({worst_metric[1]['improvement_pct']:+.1f}%)")
        
        # Count significant improvements
        significant_improvements = sum(1 for m in stats['metrics'].values() 
                                     if m['significant'] and m['improvement_pct'] > 0)
        print(f"  Significant improvements: {significant_improvements}/6 metrics")

if __name__ == "__main__":
    main() 