import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from bert_score import score as bert_score
import warnings
warnings.filterwarnings('ignore')

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

def calculate_bertscore(raw_answers, rag_answers):
    """Calculate BERTScore between RAG and Raw answers."""
    print("Calculating BERTScore... This may take a few minutes.")
    
    # Clean answers - remove empty strings
    valid_pairs = [(r, g) for r, g in zip(raw_answers, rag_answers) 
                   if r.strip() and g.strip()]
    
    if not valid_pairs:
        return [], [], []
    
    references, candidates = zip(*valid_pairs)
    
    # Calculate BERTScore using distilbert-base-uncased for efficiency
    P, R, F1 = bert_score(candidates, references, model_type="distilbert-base-uncased", 
                         verbose=False, device='cpu')
    
    return P.numpy(), R.numpy(), F1.numpy()

def load_and_process_data():
    """Load both RAG and raw results, calculate BERTScore, and merge."""
    base_dir = Path(__file__).parent
    
    # Load data
    print("Loading data files...")
    raw_df = pd.read_csv(base_dir / "qa_outputs" / "questions_answers_raw.csv")
    rag_df = pd.read_csv(base_dir / "qa_outputs" / "questions_answers_rag.csv")
    
    # Merge on Question
    merged_df = pd.merge(raw_df, rag_df, on='Question', suffixes=('_raw', '_rag'))
    
    # Calculate BERTScore (RAG vs Raw, using Raw as reference)
    raw_answers = merged_df['Answer_raw'].fillna('').astype(str).tolist()
    rag_answers = merged_df['Answer_rag'].fillna('').astype(str).tolist()
    
    # Calculate BERTScore
    bert_P, bert_R, bert_F1 = calculate_bertscore(raw_answers, rag_answers)
    
    # Add BERTScore metrics to dataframe
    if len(bert_F1) == len(merged_df):
        merged_df['bertscore_precision'] = bert_P
        merged_df['bertscore_recall'] = bert_R
        merged_df['bertscore_f1'] = bert_F1
    else:
        # Handle case where some answers were filtered out
        merged_df['bertscore_precision'] = np.nan
        merged_df['bertscore_recall'] = np.nan
        merged_df['bertscore_f1'] = np.nan
        
        valid_indices = [i for i, (r, g) in enumerate(zip(raw_answers, rag_answers)) 
                        if r.strip() and g.strip()]
        
        for idx, bert_idx in enumerate(valid_indices):
            merged_df.loc[bert_idx, 'bertscore_precision'] = bert_P[idx]
            merged_df.loc[bert_idx, 'bertscore_recall'] = bert_R[idx]
            merged_df.loc[bert_idx, 'bertscore_f1'] = bert_F1[idx]
    
    # Add categories
    question_to_category = categorize_questions(merged_df['Question'].tolist())
    merged_df['Category'] = merged_df['Question'].map(question_to_category)
    
    return merged_df

def calculate_overall_statistics(df):
    """Calculate overall statistics including BERTScore."""
    metrics = ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 
               'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity']
    
    # Add BERTScore metrics
    bertscore_metrics = ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
    
    results = {}
    
    # Standard metrics comparison (RAG vs Raw)
    for metric in metrics:
        raw_values = df[f'{metric}_raw'].values
        rag_values = df[f'{metric}_rag'].values
        
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
        
        results[metric] = {
            'raw_mean': raw_mean,
            'rag_mean': rag_mean,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05,
            'improvements': improvements,
            'total': len(df)
        }
    
    # BERTScore metrics (these compare RAG to Raw directly)
    for metric in bertscore_metrics:
        values = df[metric].dropna().values
        if len(values) > 0:
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            results[metric] = {
                'mean': mean_val,
                'std': std_val,
                'min': np.min(values),
                'max': np.max(values),
                'total_valid': len(values),
                'total': len(df)
            }
        else:
            results[metric] = {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'total_valid': 0,
                'total': len(df)
            }
    
    return results

def calculate_category_statistics(df):
    """Calculate statistics for each category including BERTScore."""
    metrics = ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 
               'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity']
    bertscore_metrics = ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
    
    results = {}
    
    for category in QUESTION_CATEGORIES.keys():
        category_df = df[df['Category'] == category]
        if len(category_df) == 0:
            continue
            
        category_results = {
            'n_questions': len(category_df),
            'metrics': {}
        }
        
        # Standard metrics comparison
        for metric in metrics:
            raw_values = category_df[f'{metric}_raw'].values
            rag_values = category_df[f'{metric}_rag'].values
            
            # Calculate basic statistics
            raw_mean = np.mean(raw_values)
            rag_mean = np.mean(rag_values)
            improvement = rag_mean - raw_mean
            improvement_pct = (improvement / raw_mean) * 100 if raw_mean != 0 else 0
            
            # Statistical tests (only if we have enough data)
            if len(raw_values) > 1:
                t_stat, p_value = stats.ttest_rel(rag_values, raw_values)
                
                # Effect size (Cohen's d)
                diff = rag_values - raw_values
                cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
            else:
                p_value = 1.0
                cohens_d = 0.0
            
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
        
        # BERTScore metrics
        for metric in bertscore_metrics:
            values = category_df[metric].dropna().values
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                category_results['metrics'][metric] = {
                    'mean': mean_val,
                    'std': std_val,
                    'min': np.min(values),
                    'max': np.max(values),
                    'total_valid': len(values),
                    'total': len(category_df)
                }
            else:
                category_results['metrics'][metric] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'total_valid': 0,
                    'total': len(category_df)
                }
        
        results[category] = category_results
    
    return results

def create_comprehensive_plots(df, overall_stats, category_stats):
    """Create comprehensive visualizations including BERTScore."""
    plots_dir = Path(__file__).parent / "analysis_results" / "enhanced_plots"
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Overall comparison including BERTScore
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Traditional metrics
    traditional_metrics = ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 'flesch_kincaid_grade']
    
    for i, metric in enumerate(traditional_metrics):
        ax = axes[i//2, i%2]
        
        raw_values = df[f'{metric}_raw'].values
        rag_values = df[f'{metric}_rag'].values
        
        ax.boxplot([raw_values, rag_values], labels=['Raw', 'RAG'])
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        
        # Add improvement annotation
        improvement = overall_stats[metric]['improvement_pct']
        significance = "***" if overall_stats[metric]['p_value'] < 0.001 else \
                      "**" if overall_stats[metric]['p_value'] < 0.01 else \
                      "*" if overall_stats[metric]['p_value'] < 0.05 else "ns"
        
        ax.text(0.5, 0.95, f'Improvement: {improvement:+.1f}% {significance}',
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # BERTScore metrics
    bertscore_metrics = ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
    for i, metric in enumerate(bertscore_metrics):
        ax = axes[1, 2] if i == 2 else axes[i//2, 2]
        if i < 2:
            continue  # Skip first two, use the last subplot
            
        values = df[metric].dropna().values
        if len(values) > 0:
            ax.hist(values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(np.mean(values), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(values):.3f}')
            ax.set_title(f'BERTScore F1 Distribution\n(RAG vs Raw)')
            ax.set_xlabel('BERTScore F1')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplot
    axes[1, 1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "overall_enhanced_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Category comparison heatmap with BERTScore
    create_enhanced_heatmap(category_stats, plots_dir)
    
    # 3. BERTScore by category
    create_bertscore_category_plot(df, plots_dir)

def create_enhanced_heatmap(category_stats, plots_dir):
    """Create an enhanced heatmap including BERTScore."""
    all_metrics = ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 
                   'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity',
                   'bertscore_f1']
    
    categories = list(category_stats.keys())
    improvement_matrix = []
    
    for category in categories:
        row = []
        for metric in all_metrics:
            if metric in category_stats[category]['metrics']:
                if metric == 'bertscore_f1':
                    # For BERTScore, use mean value as "improvement" measure
                    improvement = category_stats[category]['metrics'][metric]['mean'] * 100
                else:
                    improvement = category_stats[category]['metrics'][metric]['improvement_pct']
                row.append(improvement)
            else:
                row.append(0)
        improvement_matrix.append(row)
    
    plt.figure(figsize=(14, 8))
    
    # Custom colormap for better visualization
    cmap = plt.cm.RdYlBu_r.copy()
    
    sns.heatmap(improvement_matrix, 
                xticklabels=[m.replace('_', ' ').title().replace('Bertscore', 'BERTScore') for m in all_metrics],
                yticklabels=[c.replace(' ', '\n') for c in categories],
                annot=True, fmt='.1f', cmap=cmap, center=0,
                cbar_kws={'label': 'Improvement % / BERTScore F1 × 100'})
    
    plt.title('Enhanced Performance Analysis by Category\n(RAG vs Raw + BERTScore)')
    plt.xlabel('Metrics')
    plt.ylabel('Question Categories')
    plt.tight_layout()
    plt.savefig(plots_dir / "enhanced_improvement_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_bertscore_category_plot(df, plots_dir):
    """Create BERTScore comparison by category."""
    plt.figure(figsize=(12, 8))
    
    categories = []
    bertscore_means = []
    bertscore_stds = []
    
    for category in QUESTION_CATEGORIES.keys():
        category_df = df[df['Category'] == category]
        if len(category_df) > 0:
            bert_values = category_df['bertscore_f1'].dropna()
            if len(bert_values) > 0:
                categories.append(category.replace(' ', '\n'))
                bertscore_means.append(bert_values.mean())
                bertscore_stds.append(bert_values.std())
    
    if categories:
        x = np.arange(len(categories))
        bars = plt.bar(x, bertscore_means, yerr=bertscore_stds, capsize=5, 
                      alpha=0.8, color='lightgreen', edgecolor='darkgreen')
        
        plt.xlabel('Question Categories')
        plt.ylabel('BERTScore F1')
        plt.title('BERTScore F1 by Question Category\n(RAG vs Raw Comparison)')
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, bertscore_means):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                    f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(plots_dir / "bertscore_by_category.png", dpi=300, bbox_inches='tight')
    
    plt.close()

def generate_comprehensive_report(overall_stats, category_stats):
    """Generate a comprehensive report including BERTScore."""
    report_path = Path(__file__).parent / "analysis_results" / "enhanced_analysis_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("COMPREHENSIVE CGT CHATBOT ANALYSIS WITH BERTSCORE\n")
        f.write("=" * 60 + "\n\n")
        
        # Overall analysis
        f.write("OVERALL PERFORMANCE ANALYSIS\n")
        f.write("-" * 30 + "\n\n")
        
        f.write("Traditional Metrics (RAG vs Raw):\n")
        traditional_metrics = ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 
                               'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity']
        
        for metric in traditional_metrics:
            data = overall_stats[metric]
            f.write(f"\n{metric.replace('_', ' ').title()}:\n")
            f.write(f"  Raw Mean: {data['raw_mean']:.3f}\n")
            f.write(f"  RAG Mean: {data['rag_mean']:.3f}\n")
            f.write(f"  Improvement: {data['improvement']:+.3f} ({data['improvement_pct']:+.1f}%)\n")
            f.write(f"  P-value: {data['p_value']:.6f}\n")
            f.write(f"  Cohen's d: {data['cohens_d']:.3f}\n")
            f.write(f"  Significant: {'Yes' if data['significant'] else 'No'}\n")
        
        f.write(f"\nBERTScore Analysis (RAG vs Raw):\n")
        bertscore_metrics = ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
        for metric in bertscore_metrics:
            if metric in overall_stats:
                data = overall_stats[metric]
                f.write(f"\n{metric.replace('_', ' ').title()}:\n")
                f.write(f"  Mean: {data['mean']:.3f}\n")
                f.write(f"  Std: {data['std']:.3f}\n")
                f.write(f"  Range: {data['min']:.3f} - {data['max']:.3f}\n")
                f.write(f"  Valid samples: {data['total_valid']}/{data['total']}\n")
        
        # Category analysis
        f.write("\n" + "=" * 60 + "\n\n")
        f.write("CATEGORY-SPECIFIC ANALYSIS\n")
        f.write("-" * 30 + "\n\n")
        
        for category, stats in category_stats.items():
            f.write(f"CATEGORY: {category.upper()}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Number of Questions: {stats['n_questions']}\n\n")
            
            # Traditional metrics
            f.write("Traditional Metrics:\n")
            for metric in traditional_metrics:
                if metric in stats['metrics']:
                    data = stats['metrics'][metric]
                    f.write(f"\n{metric.replace('_', ' ').title()}:\n")
                    f.write(f"  Improvement: {data['improvement_pct']:+.1f}%\n")
                    f.write(f"  Significant: {'Yes' if data['significant'] else 'No'}\n")
            
            # BERTScore metrics
            f.write(f"\nBERTScore Metrics:\n")
            for metric in bertscore_metrics:
                if metric in stats['metrics']:
                    data = stats['metrics'][metric]
                    f.write(f"\n{metric.replace('_', ' ').title()}:\n")
                    f.write(f"  Mean: {data['mean']:.3f}\n")
                    f.write(f"  Valid: {data['total_valid']}/{data['total']}\n")
            
            f.write("\n" + "=" * 60 + "\n\n")
    
    print(f"Enhanced analysis report saved to: {report_path}")

def main():
    """Main enhanced analysis function."""
    print("Enhanced CGT Chatbot Analysis with BERTScore")
    print("=" * 50)
    
    # Load and process data
    df = load_and_process_data()
    
    print("Calculating overall statistics...")
    overall_stats = calculate_overall_statistics(df)
    
    print("Calculating category statistics...")
    category_stats = calculate_category_statistics(df)
    
    print("Creating enhanced visualizations...")
    create_comprehensive_plots(df, overall_stats, category_stats)
    
    print("Generating comprehensive report...")
    generate_comprehensive_report(overall_stats, category_stats)
    
    # Print enhanced summary
    print("\nENHANCED ANALYSIS SUMMARY:")
    print("=" * 40)
    
    # Overall BERTScore summary
    if 'bertscore_f1' in overall_stats:
        bert_data = overall_stats['bertscore_f1']
        print(f"\nOverall BERTScore F1: {bert_data['mean']:.3f} ± {bert_data['std']:.3f}")
        print(f"Valid BERTScore calculations: {bert_data['total_valid']}/{bert_data['total']}")
    
    # Traditional metrics summary
    print(f"\nTraditional Metrics Summary:")
    significant_improvements = sum(1 for m in ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 
                                              'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity']
                                  if overall_stats[m]['significant'] and overall_stats[m]['improvement_pct'] > 0)
    print(f"Significant improvements: {significant_improvements}/6 metrics")
    
    # Category summary
    print(f"\nCategory Analysis Summary:")
    for category, stats in category_stats.items():
        bert_f1 = stats['metrics'].get('bertscore_f1', {}).get('mean', 0)
        print(f"{category}: BERTScore F1 = {bert_f1:.3f} ({stats['n_questions']} questions)")
    
    print(f"\nAnalysis complete! Check the 'analysis_results/enhanced_plots/' directory for visualizations.")

if __name__ == "__main__":
    main() 