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

def load_gold_standard_answers():
    """Load gold standard answers from questions_answers.txt."""
    base_dir = Path(__file__).parent
    gold_standard_file = base_dir / "questions_answers.txt"
    
    questions = []
    answers = []
    
    with open(gold_standard_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by Q: and A: patterns
    sections = content.split('Q: ')[1:]  # Skip empty first element
    
    for section in sections:
        lines = section.strip().split('\n')
        if len(lines) < 2:
            continue
            
        # Extract question (first line)
        question = lines[0].strip()
        
        # Extract answer (everything after A:)
        answer_lines = []
        answer_started = False
        for line in lines[1:]:
            if line.startswith('A: '):
                answer_started = True
                answer_lines.append(line[3:])  # Remove 'A: ' prefix
            elif answer_started and line.strip() and not line.startswith('Q: '):
                answer_lines.append(line)
            elif line.startswith('Q: '):
                break
        
        if answer_lines:
            answer = ' '.join(answer_lines).strip()
            questions.append(question)
            answers.append(answer)
    
    return questions, answers

def calculate_bertscore_vs_gold(answers, gold_answers):
    """Calculate BERTScore against gold standard answers."""
    print("Calculating BERTScore against gold standard...")
    
    # Clean answers - remove empty strings and match with gold standard
    valid_pairs = []
    for ans, gold in zip(answers, gold_answers):
        if ans.strip() and gold.strip():
            valid_pairs.append((ans.strip(), gold.strip()))
    
    if not valid_pairs:
        return [], [], []
    
    candidates, references = zip(*valid_pairs)
    
    # Calculate BERTScore using distilbert-base-uncased for efficiency
    P, R, F1 = bert_score(candidates, references, model_type="distilbert-base-uncased", 
                         verbose=False, device='cpu')
    
    return P.numpy(), R.numpy(), F1.numpy()

def load_and_process_data():
    """Load both RAG and raw results, calculate BERTScore vs gold standard, and merge."""
    base_dir = Path(__file__).parent
    
    # Load data
    print("Loading data files...")
    raw_df = pd.read_csv(base_dir / "qa_outputs" / "questions_answers_raw.csv")
    rag_df = pd.read_csv(base_dir / "qa_outputs" / "questions_answers_rag.csv")
    
    # Load gold standard
    print("Loading gold standard answers...")
    gold_questions, gold_answers = load_gold_standard_answers()
    gold_df = pd.DataFrame({'Question': gold_questions, 'Gold_Answer': gold_answers})
    
    # Merge on Question
    merged_df = pd.merge(raw_df, rag_df, on='Question', suffixes=('_raw', '_rag'))
    merged_df = pd.merge(merged_df, gold_df, on='Question', how='left')
    
    # Fill missing gold answers with empty strings
    merged_df['Gold_Answer'] = merged_df['Gold_Answer'].fillna('')
    
    # Calculate BERTScore for both approaches vs gold standard
    raw_answers = merged_df['Answer_raw'].fillna('').astype(str).tolist()
    rag_answers = merged_df['Answer_rag'].fillna('').astype(str).tolist()
    gold_answers_matched = merged_df['Gold_Answer'].fillna('').astype(str).tolist()
    
    # Calculate BERTScore for Raw vs Gold
    print("Calculating BERTScore for Raw answers vs Gold standard...")
    raw_bert_P, raw_bert_R, raw_bert_F1 = calculate_bertscore_vs_gold(raw_answers, gold_answers_matched)
    
    # Calculate BERTScore for RAG vs Gold
    print("Calculating BERTScore for RAG answers vs Gold standard...")
    rag_bert_P, rag_bert_R, rag_bert_F1 = calculate_bertscore_vs_gold(rag_answers, gold_answers_matched)
    
    # Add BERTScore metrics to dataframe
    merged_df['bertscore_precision_raw'] = np.nan
    merged_df['bertscore_recall_raw'] = np.nan
    merged_df['bertscore_f1_raw'] = np.nan
    merged_df['bertscore_precision_rag'] = np.nan
    merged_df['bertscore_recall_rag'] = np.nan
    merged_df['bertscore_f1_rag'] = np.nan
    
    # Map valid BERTScore results back to dataframe
    valid_indices = [i for i, (r, g, gold) in enumerate(zip(raw_answers, rag_answers, gold_answers_matched)) 
                    if r.strip() and g.strip() and gold.strip()]
    
    print(f"BERTScore calculated for {len(valid_indices)} out of {len(merged_df)} questions")
    
    # Map Raw BERTScore results
    if len(raw_bert_F1) == len(valid_indices):
        for idx, bert_idx in enumerate(valid_indices):
            merged_df.loc[bert_idx, 'bertscore_precision_raw'] = raw_bert_P[idx]
            merged_df.loc[bert_idx, 'bertscore_recall_raw'] = raw_bert_R[idx]
            merged_df.loc[bert_idx, 'bertscore_f1_raw'] = raw_bert_F1[idx]
    
    # Map RAG BERTScore results
    if len(rag_bert_F1) == len(valid_indices):
        for idx, bert_idx in enumerate(valid_indices):
            merged_df.loc[bert_idx, 'bertscore_precision_rag'] = rag_bert_P[idx]
            merged_df.loc[bert_idx, 'bertscore_recall_rag'] = rag_bert_R[idx]
            merged_df.loc[bert_idx, 'bertscore_f1_rag'] = rag_bert_F1[idx]
    
    # Add categories
    question_to_category = categorize_questions(merged_df['Question'].tolist())
    merged_df['Category'] = merged_df['Question'].map(question_to_category)
    
    return merged_df

def calculate_overall_statistics(df):
    """Calculate overall statistics including BERTScore vs gold standard."""
    metrics = ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 
               'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity']
    
    # BERTScore metrics for both approaches vs gold
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
    
    # BERTScore metrics comparing both approaches against gold standard
    for metric in bertscore_metrics:
        raw_values = df[f'{metric}_raw'].dropna().values
        rag_values = df[f'{metric}_rag'].dropna().values
        
        if len(raw_values) > 0 and len(rag_values) > 0:
            raw_mean = np.mean(raw_values)
            rag_mean = np.mean(rag_values)
            improvement = rag_mean - raw_mean
            improvement_pct = (improvement / raw_mean) * 100 if raw_mean != 0 else 0
            
            # Statistical test (paired if same length, independent otherwise)
            if len(raw_values) == len(rag_values):
                t_stat, p_value = stats.ttest_rel(rag_values, raw_values)
                diff = rag_values - raw_values
                cohens_d = np.mean(diff) / np.std(diff, ddof=1)
            else:
                t_stat, p_value = stats.ttest_ind(rag_values, raw_values)
                pooled_std = np.sqrt(((len(raw_values) - 1) * np.var(raw_values, ddof=1) + 
                                    (len(rag_values) - 1) * np.var(rag_values, ddof=1)) / 
                                   (len(raw_values) + len(rag_values) - 2))
                cohens_d = (rag_mean - raw_mean) / pooled_std
            
            improvements = np.sum(rag_values > raw_values) if len(raw_values) == len(rag_values) else 0
            
            results[metric] = {
                'raw_mean': raw_mean,
                'rag_mean': rag_mean,
                'improvement': improvement,
                'improvement_pct': improvement_pct,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05,
                'improvements': improvements,
                'total_raw': len(raw_values),
                'total_rag': len(rag_values),
                'raw_std': np.std(raw_values),
                'rag_std': np.std(rag_values)
            }
    
    return results

def calculate_category_statistics(df):
    """Calculate statistics for each category including BERTScore vs gold standard."""
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
        
        # BERTScore metrics - now comparing RAG vs Raw against gold standard
        for metric in bertscore_metrics:
            raw_values = category_df[f'{metric}_raw'].dropna().values
            rag_values = category_df[f'{metric}_rag'].dropna().values
            
            if len(raw_values) > 0 and len(rag_values) > 0:
                raw_mean = np.mean(raw_values)
                rag_mean = np.mean(rag_values)
                improvement = rag_mean - raw_mean
                improvement_pct = (improvement / raw_mean) * 100 if raw_mean != 0 else 0
                
                # Statistical test if we have paired data
                if len(raw_values) == len(rag_values) and len(raw_values) > 1:
                    t_stat, p_value = stats.ttest_rel(rag_values, raw_values)
                    diff = rag_values - raw_values
                    cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
                else:
                    p_value = 1.0
                    cohens_d = 0.0
                
                category_results['metrics'][metric] = {
                    'raw_mean': raw_mean,
                    'rag_mean': rag_mean,
                    'improvement': improvement,
                    'improvement_pct': improvement_pct,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significant': p_value < 0.05,
                    'total_raw': len(raw_values),
                    'total_rag': len(rag_values)
                }
            else:
                category_results['metrics'][metric] = {
                    'raw_mean': 0.0,
                    'rag_mean': 0.0,
                    'improvement': 0.0,
                    'improvement_pct': 0.0,
                    'p_value': 1.0,
                    'cohens_d': 0.0,
                    'significant': False,
                    'total_raw': len(raw_values) if len(raw_values) > 0 else 0,
                    'total_rag': len(rag_values) if len(rag_values) > 0 else 0
                }
        
        results[category] = category_results
    
    return results

def create_comprehensive_plots(df, overall_stats, category_stats):
    """Create comprehensive visualizations including BERTScore vs gold standard."""
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
    
    # BERTScore F1 comparison (Raw vs RAG against gold standard)
    ax = axes[1, 2]
    raw_bert_values = df['bertscore_f1_raw'].dropna().values
    rag_bert_values = df['bertscore_f1_rag'].dropna().values
    
    if len(raw_bert_values) > 0 and len(rag_bert_values) > 0:
        ax.boxplot([raw_bert_values, rag_bert_values], labels=['Raw vs Gold', 'RAG vs Gold'])
        ax.set_title('BERTScore F1 vs Gold Standard')
        ax.grid(True, alpha=0.3)
        
        # Add improvement annotation
        if 'bertscore_f1' in overall_stats:
            improvement = overall_stats['bertscore_f1']['improvement_pct']
            significance = "***" if overall_stats['bertscore_f1']['p_value'] < 0.001 else \
                          "**" if overall_stats['bertscore_f1']['p_value'] < 0.01 else \
                          "*" if overall_stats['bertscore_f1']['p_value'] < 0.05 else "ns"
            
            ax.text(0.5, 0.95, f'Improvement: {improvement:+.1f}% {significance}',
                    transform=ax.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
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
    """Create an enhanced heatmap including BERTScore improvements."""
    all_metrics = ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 
                   'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity',
                   'bertscore_f1']
    
    categories = list(category_stats.keys())
    improvement_matrix = []
    
    for category in categories:
        row = []
        for metric in all_metrics:
            if metric in category_stats[category]['metrics']:
                # All metrics now use improvement_pct
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
                cbar_kws={'label': 'Improvement % (RAG vs Raw)'})
    
    plt.title('Enhanced Performance Analysis by Category\n(RAG vs Raw including BERTScore vs Gold Standard)')
    plt.xlabel('Metrics')
    plt.ylabel('Question Categories')
    plt.tight_layout()
    plt.savefig(plots_dir / "enhanced_improvement_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_bertscore_category_plot(df, plots_dir):
    """Create BERTScore comparison by category (RAG vs Raw against gold standard)."""
    plt.figure(figsize=(12, 8))
    
    categories = []
    raw_means = []
    rag_means = []
    raw_stds = []
    rag_stds = []
    
    for category in QUESTION_CATEGORIES.keys():
        category_df = df[df['Category'] == category]
        if len(category_df) > 0:
            raw_bert_values = category_df['bertscore_f1_raw'].dropna()
            rag_bert_values = category_df['bertscore_f1_rag'].dropna()
            
            if len(raw_bert_values) > 0 and len(rag_bert_values) > 0:
                categories.append(category.replace(' ', '\n'))
                raw_means.append(raw_bert_values.mean())
                rag_means.append(rag_bert_values.mean())
                raw_stds.append(raw_bert_values.std())
                rag_stds.append(rag_bert_values.std())
    
    if categories:
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, raw_means, width, yerr=raw_stds, capsize=5, 
                       alpha=0.8, color='lightcoral', edgecolor='darkred', label='Raw vs Gold')
        bars2 = plt.bar(x + width/2, rag_means, width, yerr=rag_stds, capsize=5, 
                       alpha=0.8, color='lightgreen', edgecolor='darkgreen', label='RAG vs Gold')
        
        plt.xlabel('Question Categories')
        plt.ylabel('BERTScore F1')
        plt.title('BERTScore F1 by Question Category\n(Raw vs RAG Performance against Gold Standard)')
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars, means in [(bars1, raw_means), (bars2, rag_means)]:
            for bar, mean_val in zip(bars, means):
                plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                        f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "bertscore_by_category.png", dpi=300, bbox_inches='tight')
    
    plt.close()

def generate_comprehensive_report(overall_stats, category_stats):
    """Generate a comprehensive report including BERTScore vs gold standard."""
    report_path = Path(__file__).parent / "analysis_results" / "enhanced_analysis_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("COMPREHENSIVE CGT CHATBOT ANALYSIS WITH BERTSCORE VS GOLD STANDARD\n")
        f.write("=" * 70 + "\n\n")
        
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
        
        f.write(f"\nBERTScore Analysis vs Gold Standard:\n")
        bertscore_metrics = ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
        for metric in bertscore_metrics:
            if metric in overall_stats:
                data = overall_stats[metric]
                f.write(f"\n{metric.replace('_', ' ').title()}:\n")
                f.write(f"  Raw Mean: {data['raw_mean']:.3f}\n")
                f.write(f"  RAG Mean: {data['rag_mean']:.3f}\n")
                f.write(f"  Improvement: {data['improvement']:+.3f} ({data['improvement_pct']:+.1f}%)\n")
                f.write(f"  P-value: {data['p_value']:.6f}\n")
                f.write(f"  Significant: {'Yes' if data['significant'] else 'No'}\n")
                f.write(f"  Valid samples: {data['total_raw']}/{data['total_rag']}\n")
        
        # Category analysis
        f.write("\n" + "=" * 70 + "\n\n")
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
            f.write(f"\nBERTScore Metrics vs Gold Standard:\n")
            for metric in bertscore_metrics:
                if metric in stats['metrics']:
                    data = stats['metrics'][metric]
                    f.write(f"\n{metric.replace('_', ' ').title()}:\n")
                    f.write(f"  Raw Mean: {data['raw_mean']:.3f}\n")
                    f.write(f"  RAG Mean: {data['rag_mean']:.3f}\n")
                    f.write(f"  Improvement: {data['improvement_pct']:+.1f}%\n")
                    f.write(f"  Significant: {'Yes' if data['significant'] else 'No'}\n")
            
            f.write("\n" + "=" * 70 + "\n\n")
    
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
        print(f"\nOverall BERTScore F1:")
        print(f"  Raw vs Gold: {bert_data['raw_mean']:.3f} ± {bert_data['raw_std']:.3f}")
        print(f"  RAG vs Gold: {bert_data['rag_mean']:.3f} ± {bert_data['rag_std']:.3f}")
        print(f"  Improvement: {bert_data['improvement_pct']:+.1f}%")
        print(f"  Statistical significance: {'Yes' if bert_data['significant'] else 'No'}")
        print(f"  Valid BERTScore calculations: {bert_data['total_raw']}/{bert_data['total_rag']}")
    
    # Traditional metrics summary
    print(f"\nTraditional Metrics Summary:")
    significant_improvements = sum(1 for m in ['semantic_similarity', 'answer_length', 'flesch_reading_ease', 
                                              'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity']
                                  if overall_stats[m]['significant'] and overall_stats[m]['improvement_pct'] > 0)
    print(f"Significant improvements: {significant_improvements}/6 metrics")
    
    # Category summary
    print(f"\nCategory Analysis Summary:")
    for category, stats in category_stats.items():
        # BERTScore F1 improvement for each category
        bert_f1_data = stats['metrics'].get('bertscore_f1', {})
        if 'improvement_pct' in bert_f1_data:
            improvement = bert_f1_data['improvement_pct']
            print(f"{category}: BERTScore F1 improvement = {improvement:+.1f}% ({stats['n_questions']} questions)")
        else:
            print(f"{category}: No BERTScore data ({stats['n_questions']} questions)")
    
    print(f"\nAnalysis complete! Check the 'analysis_results/enhanced_plots/' directory for visualizations.")

if __name__ == "__main__":
    main() 