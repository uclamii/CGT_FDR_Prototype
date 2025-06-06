import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from bert_score import score as bert_score
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
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

def calculate_comprehensive_stats(df):
    """Calculate both overall and category statistics."""
    # Excluded answer_length as requested
    metrics = ['semantic_similarity', 'flesch_reading_ease', 
               'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity']
    bertscore_metrics = ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
    
    # Overall statistics
    overall_stats = {}
    for metric in metrics:
        raw_values = df[f'{metric}_raw'].values
        rag_values = df[f'{metric}_rag'].values
        
        raw_mean = np.mean(raw_values)
        rag_mean = np.mean(rag_values)
        improvement = rag_mean - raw_mean
        improvement_pct = (improvement / raw_mean) * 100 if raw_mean != 0 else 0
        
        t_stat, p_value = stats.ttest_rel(rag_values, raw_values)
        diff = rag_values - raw_values
        cohens_d = np.mean(diff) / np.std(diff, ddof=1)
        
        overall_stats[metric] = {
            'raw_mean': raw_mean,
            'rag_mean': rag_mean,
            'improvement_pct': improvement_pct,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05
        }
    
    # BERTScore overall
    for metric in bertscore_metrics:
        values = df[metric].dropna().values
        if len(values) > 0:
            overall_stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'total_valid': len(values)
            }
    
    # Category statistics
    category_stats = {}
    for category in QUESTION_CATEGORIES.keys():
        category_df = df[df['Category'] == category]
        if len(category_df) == 0:
            continue
            
        category_results = {'n_questions': len(category_df), 'metrics': {}}
        
        for metric in metrics:
            raw_values = category_df[f'{metric}_raw'].values
            rag_values = category_df[f'{metric}_rag'].values
            
            raw_mean = np.mean(raw_values)
            rag_mean = np.mean(rag_values)
            improvement_pct = ((rag_mean - raw_mean) / raw_mean) * 100 if raw_mean != 0 else 0
            
            if len(raw_values) > 1:
                t_stat, p_value = stats.ttest_rel(rag_values, raw_values)
            else:
                p_value = 1.0
            
            category_results['metrics'][metric] = {
                'improvement_pct': improvement_pct,
                'significant': p_value < 0.05
            }
        
        for metric in bertscore_metrics:
            values = category_df[metric].dropna().values
            if len(values) > 0:
                category_results['metrics'][metric] = {
                    'mean': np.mean(values),
                    'total_valid': len(values)
                }
        
        category_stats[category] = category_results
    
    return overall_stats, category_stats

def create_comprehensive_dashboard(df, overall_stats, category_stats):
    """Create a comprehensive dashboard with charts and tables."""
    
    # Set up the figure with a complex grid
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3)
    
    # Color scheme
    colors = {
        'raw': '#FF6B6B',
        'rag': '#4ECDC4',
        'improvement': '#45B7D1',
        'decline': '#FFA07A'
    }
    
    # Title
    fig.suptitle('CGT Chatbot Performance Analysis: RAG vs Raw Approach\nComprehensive Results Dashboard', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Overall Performance Comparison (Top Left) - Excluded answer_length
    ax1 = fig.add_subplot(gs[0, :2])
    metrics = ['semantic_similarity', 'flesch_reading_ease', 'flesch_kincaid_grade']
    metric_names = ['Semantic\nSimilarity', 'Reading\nEase', 'Grade\nLevel']
    
    raw_values = [overall_stats[m]['raw_mean'] for m in metrics]
    rag_values = [overall_stats[m]['rag_mean'] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, raw_values, width, label='Raw', color=colors['raw'], alpha=0.8)
    bars2 = ax1.bar(x + width/2, rag_values, width, label='RAG', color=colors['rag'], alpha=0.8)
    
    ax1.set_xlabel('Metrics', fontweight='bold')
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('Overall Performance Comparison', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add significance markers
    for i, metric in enumerate(metrics):
        improvement = overall_stats[metric]['improvement_pct']
        significant = overall_stats[metric]['significant']
        marker = '***' if significant and overall_stats[metric]['p_value'] < 0.001 else \
                '**' if significant and overall_stats[metric]['p_value'] < 0.01 else \
                '*' if significant and overall_stats[metric]['p_value'] < 0.05 else 'ns'
        
        color = colors['improvement'] if improvement > 0 else colors['decline']
        ax1.text(i, max(raw_values[i], rag_values[i]) * 1.05, 
                f'{improvement:+.1f}%\n{marker}', 
                ha='center', va='bottom', fontweight='bold', color=color, fontsize=9)
    
    # 2. BERTScore Distribution (Top Right)
    ax2 = fig.add_subplot(gs[0, 2:])
    bert_values = df['bertscore_f1'].dropna().values
    if len(bert_values) > 0:
        ax2.hist(bert_values, bins=15, alpha=0.7, color=colors['rag'], edgecolor='black')
        ax2.axvline(np.mean(bert_values), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(bert_values):.3f}')
        ax2.set_xlabel('BERTScore F1', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title('BERTScore F1 Distribution\n(RAG vs Raw Semantic Similarity)', fontweight='bold', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Category Performance Heatmap (Middle) - Excluded answer_length
    ax3 = fig.add_subplot(gs[1, :])
    all_metrics = ['semantic_similarity', 'flesch_reading_ease', 
                   'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity', 'bertscore_f1']
    
    categories = list(category_stats.keys())
    improvement_matrix = []
    
    for category in categories:
        row = []
        for metric in all_metrics:
            if metric in category_stats[category]['metrics']:
                if metric == 'bertscore_f1':
                    value = category_stats[category]['metrics'][metric]['mean'] * 100
                else:
                    value = category_stats[category]['metrics'][metric]['improvement_pct']
                row.append(value)
            else:
                row.append(0)
        improvement_matrix.append(row)
    
    im = ax3.imshow(improvement_matrix, cmap='RdYlBu_r', aspect='auto')
    ax3.set_xticks(range(len(all_metrics)))
    ax3.set_xticklabels([m.replace('_', '\n').title().replace('Bertscore', 'BERTScore') for m in all_metrics], 
                       rotation=45, ha='right')
    ax3.set_yticks(range(len(categories)))
    ax3.set_yticklabels([c.replace(' ', '\n') for c in categories])
    ax3.set_title('Performance by Question Category (% Improvement / BERTScore F1 × 100)', fontweight='bold', fontsize=14)
    
    # Add text annotations to heatmap
    for i in range(len(categories)):
        for j in range(len(all_metrics)):
            text = ax3.text(j, i, f'{improvement_matrix[i][j]:.1f}',
                          ha="center", va="center", color="black", fontweight='bold', fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3, orientation='horizontal', pad=0.1, shrink=0.8)
    cbar.set_label('Improvement % / BERTScore F1 × 100', fontweight='bold')
    
    # 4. Overall Statistics Table (Bottom Left) - Excluded answer_length
    ax4 = fig.add_subplot(gs[2:, :2])
    ax4.axis('tight')
    ax4.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ['Metric', 'Raw Mean', 'RAG Mean', 'Improvement %', 'P-value', 'Significant']
    
    for metric in ['semantic_similarity', 'flesch_reading_ease', 'flesch_kincaid_grade']:
        data = overall_stats[metric]
        row = [
            metric.replace('_', ' ').title(),
            f"{data['raw_mean']:.3f}",
            f"{data['rag_mean']:.3f}",
            f"{data['improvement_pct']:+.1f}%",
            f"{data['p_value']:.4f}",
            "Yes" if data['significant'] else "No"
        ]
        table_data.append(row)
    
    # BERTScore row
    if 'bertscore_f1' in overall_stats:
        bert_data = overall_stats['bertscore_f1']
        table_data.append([
            'BERTScore F1',
            'N/A',
            f"{bert_data['mean']:.3f}",
            'N/A',
            'N/A',
            f"{bert_data['total_valid']}/49"
        ])
    
    table = ax4.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if j == 4:  # P-value column
                p_val = float(table_data[i-1][4]) if table_data[i-1][4] != 'N/A' else 1.0
                if p_val < 0.05:
                    table[(i, j)].set_facecolor('#E8F8F5')
            elif j == 3:  # Improvement column
                improvement_text = table_data[i-1][3]
                if improvement_text != 'N/A' and '+' in improvement_text:
                    table[(i, j)].set_facecolor('#E8F8F5')
                elif improvement_text != 'N/A' and '-' in improvement_text:
                    table[(i, j)].set_facecolor('#FADBD8')
    
    ax4.set_title('Overall Performance Statistics', fontweight='bold', fontsize=14, pad=20)
    
    # 5. Category Summary Table (Bottom Right) - Excluded answer_length from significant count
    ax5 = fig.add_subplot(gs[2:, 2:])
    ax5.axis('tight')
    ax5.axis('off')
    
    # Category summary data
    cat_table_data = []
    cat_headers = ['Question Category', 'N Questions', 'Semantic Sim %', 'BERTScore F1', 'Significant Metrics']
    
    for category, stats in category_stats.items():
        sem_sim_improvement = stats['metrics'].get('semantic_similarity', {}).get('improvement_pct', 0)
        bert_f1 = stats['metrics'].get('bertscore_f1', {}).get('mean', 0)
        
        # Count significant metrics (excluded answer_length)
        significant_count = sum(1 for m in ['semantic_similarity', 'flesch_reading_ease', 
                                          'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity']
                              if stats['metrics'].get(m, {}).get('significant', False))
        
        row = [
            category[:15] + '...' if len(category) > 15 else category,
            str(stats['n_questions']),
            f"{sem_sim_improvement:+.1f}%",
            f"{bert_f1:.3f}",
            f"{significant_count}/5"  # Changed to 5 since we excluded answer_length
        ]
        cat_table_data.append(row)
    
    cat_table = ax5.table(cellText=cat_table_data, colLabels=cat_headers, loc='center', cellLoc='center')
    cat_table.auto_set_font_size(False)
    cat_table.set_fontsize(9)
    cat_table.scale(1, 1.8)
    
    # Style the category table
    for i in range(len(cat_headers)):
        cat_table[(0, i)].set_facecolor('#45B7D1')
        cat_table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax5.set_title('Category Performance Summary', fontweight='bold', fontsize=14, pad=20)
    
    # Add legend/notes
    legend_elements = [
        mpatches.Patch(color=colors['raw'], label='Raw Chatbot (Baseline)'),
        mpatches.Patch(color=colors['rag'], label='RAG Chatbot (Enhanced)'),
        mpatches.Patch(color='#E8F8F5', label='Significant Improvement'),
        mpatches.Patch(color='#FADBD8', label='Significant Decline')
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
              ncol=4, fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    # Save the dashboard
    plots_dir = Path(__file__).parent / "analysis_results" / "dashboard"
    plots_dir.mkdir(exist_ok=True)
    
    plt.savefig(plots_dir / "comprehensive_cgt_analysis_dashboard.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(plots_dir / "comprehensive_cgt_analysis_dashboard.pdf", 
                bbox_inches='tight', facecolor='white')
    
    return fig

def main():
    """Main function to create the comprehensive dashboard."""
    print("Creating Comprehensive CGT Chatbot Analysis Dashboard")
    print("=" * 55)
    
    # Load and process data
    df = load_and_process_data()
    
    print("Calculating comprehensive statistics...")
    overall_stats, category_stats = calculate_comprehensive_stats(df)
    
    print("Creating comprehensive dashboard...")
    fig = create_comprehensive_dashboard(df, overall_stats, category_stats)
    
    print("\nDashboard Summary:")
    print("=" * 30)
    
    # Print key findings (excluded answer_length)
    semantic_improvement = overall_stats['semantic_similarity']['improvement_pct']
    bert_f1 = overall_stats['bertscore_f1']['mean']
    
    print(f"Key Findings:")
    print(f"  • Semantic Similarity: {semantic_improvement:+.1f}% improvement")
    print(f"  • BERTScore F1: {bert_f1:.3f} (RAG vs Raw)")
    print(f"  • Categories analyzed: {len(category_stats)}")
    print(f"  • Total questions: {len(df)}")
    
    # Excluded answer_length from significant metrics count
    significant_metrics = sum(1 for m in ['semantic_similarity', 'flesch_reading_ease', 
                                        'flesch_kincaid_grade', 'sentiment_polarity', 'sentiment_subjectivity']
                            if overall_stats[m]['significant'])
    print(f"  • Significant differences: {significant_metrics}/5 metrics")
    
    print(f"\nDashboard saved to: analysis_results/dashboard/")
    print(f"Files created:")
    print(f"  • comprehensive_cgt_analysis_dashboard.png (High-res image)")
    print(f"  • comprehensive_cgt_analysis_dashboard.pdf (Publication-ready)")
    
    return fig

if __name__ == "__main__":
    main() 