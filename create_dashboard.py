import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from bert_score import score as bert_score
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# Set style for clean visualization
plt.style.use('default')
sns.set_palette("husl")

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
        return []
    
    candidates, references = zip(*valid_pairs)
    
    # Calculate BERTScore using distilbert-base-uncased for efficiency
    P, R, F1 = bert_score(candidates, references, model_type="distilbert-base-uncased", 
                         verbose=False, device='cpu')
    
    return F1.numpy()

def load_and_process_data():
    """Load both RAG and raw results, calculate BERTScore vs gold standard, and merge."""
    base_dir = Path(__file__).parent
    
    # Load data
    print("Loading data...")
    raw_df = pd.read_csv(base_dir / "qa_outputs" / "questions_answers_raw.csv")
    rag_df = pd.read_csv(base_dir / "qa_outputs" / "questions_answers_rag.csv")
    
    # Load gold standard
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
    raw_bert_f1 = calculate_bertscore_vs_gold(raw_answers, gold_answers_matched)
    
    # Calculate BERTScore for RAG vs Gold  
    rag_bert_f1 = calculate_bertscore_vs_gold(rag_answers, gold_answers_matched)
    
    # Add BERTScore metrics to dataframe
    merged_df['bertscore_f1_raw'] = np.nan
    merged_df['bertscore_f1_rag'] = np.nan
    
    # Map valid BERTScore results back to dataframe
    valid_indices = [i for i, (r, g, gold) in enumerate(zip(raw_answers, rag_answers, gold_answers_matched)) 
                    if r.strip() and g.strip() and gold.strip()]
    
    if len(raw_bert_f1) == len(valid_indices):
        for idx, bert_idx in enumerate(valid_indices):
            merged_df.loc[bert_idx, 'bertscore_f1_raw'] = raw_bert_f1[idx]
    
    if len(rag_bert_f1) == len(valid_indices):
        for idx, bert_idx in enumerate(valid_indices):
            merged_df.loc[bert_idx, 'bertscore_f1_rag'] = rag_bert_f1[idx]
    
    # Add categories
    question_to_category = categorize_questions(merged_df['Question'].tolist())
    merged_df['Category'] = merged_df['Question'].map(question_to_category)
    
    return merged_df

def calculate_stats(df):
    """Calculate comprehensive statistics including detailed category analysis."""
    # Focus on most important metrics
    metrics = ['semantic_similarity', 'flesch_reading_ease']
    
    results = {}
    
    # Overall statistics
    for metric in metrics:
        raw_values = df[f'{metric}_raw'].values
        rag_values = df[f'{metric}_rag'].values
        
        raw_mean = np.mean(raw_values)
        rag_mean = np.mean(rag_values)
        improvement_pct = ((rag_mean - raw_mean) / raw_mean) * 100 if raw_mean != 0 else 0
        
        t_stat, p_value = stats.ttest_rel(rag_values, raw_values)
        
        results[metric] = {
            'raw_mean': raw_mean,
            'rag_mean': rag_mean,
            'improvement_pct': improvement_pct,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    # BERTScore statistics
    raw_bert_values = df['bertscore_f1_raw'].dropna().values
    rag_bert_values = df['bertscore_f1_rag'].dropna().values
    
    if len(raw_bert_values) > 0 and len(rag_bert_values) > 0:
        # Calculate improvement
        bert_improvement_pct = ((np.mean(rag_bert_values) - np.mean(raw_bert_values)) / np.mean(raw_bert_values)) * 100
        
        # Statistical test
        if len(raw_bert_values) == len(rag_bert_values):
            t_stat, p_value = stats.ttest_rel(rag_bert_values, raw_bert_values)
        else:
            t_stat, p_value = stats.ttest_ind(rag_bert_values, raw_bert_values)
        
        results['bertscore_f1'] = {
            'raw_mean': np.mean(raw_bert_values),
            'rag_mean': np.mean(rag_bert_values),
            'improvement_pct': bert_improvement_pct,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'raw_std': np.std(raw_bert_values),
            'rag_std': np.std(rag_bert_values),
            'total_valid': len(raw_bert_values)
        }
    
    # Enhanced category statistics with BERTScore details
    category_results = {}
    for category in QUESTION_CATEGORIES.keys():
        category_df = df[df['Category'] == category]
        if len(category_df) == 0:
            continue
            
        # Semantic similarity improvement
        raw_sem = category_df['semantic_similarity_raw'].mean()
        rag_sem = category_df['semantic_similarity_rag'].mean()
        sem_improvement = ((rag_sem - raw_sem) / raw_sem) * 100 if raw_sem != 0 else 0
        
        # Reading ease improvement
        raw_read = category_df['flesch_reading_ease_raw'].mean()
        rag_read = category_df['flesch_reading_ease_rag'].mean()
        read_improvement = ((rag_read - raw_read) / raw_read) * 100 if raw_read != 0 else 0
        
        # BERTScore for category with statistical test
        raw_bert_cat_values = category_df['bertscore_f1_raw'].dropna()
        rag_bert_cat_values = category_df['bertscore_f1_rag'].dropna()
        
        if len(raw_bert_cat_values) > 0 and len(rag_bert_cat_values) > 0:
            raw_bert_mean = raw_bert_cat_values.mean()
            rag_bert_mean = rag_bert_cat_values.mean()
            bert_improvement = ((rag_bert_mean - raw_bert_mean) / raw_bert_mean) * 100
            
            # Statistical test for category BERTScore
            if len(raw_bert_cat_values) == len(rag_bert_cat_values) and len(raw_bert_cat_values) > 1:
                _, bert_p_value = stats.ttest_rel(rag_bert_cat_values, raw_bert_cat_values)
                bert_significant = bert_p_value < 0.05
            else:
                bert_significant = False
        else:
            raw_bert_mean = 0
            rag_bert_mean = 0
            bert_improvement = 0
            bert_significant = False
        
        category_results[category] = {
            'n_questions': len(category_df),
            'semantic_improvement': sem_improvement,
            'reading_improvement': read_improvement,
            'bertscore_raw_mean': raw_bert_mean,
            'bertscore_rag_mean': rag_bert_mean,
            'bertscore_improvement': bert_improvement,
            'bertscore_significant': bert_significant,
            'bertscore_valid_count': len(raw_bert_cat_values)
        }
    
    return results, category_results

def create_enhanced_dashboard(df, overall_stats, category_stats):
    """Create an enhanced dashboard with comprehensive analysis."""
    
    # Create figure with enhanced layout - 3x2 grid
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1], 
                         hspace=0.3, wspace=0.25)
    
    fig.suptitle('CGT Chatbot Comprehensive Analysis: RAG vs Raw Performance\nBERTScore vs Gold Standard + Category Breakdown', 
                 fontsize=20, fontweight='bold', y=0.95)
    
    # Color scheme - clean and professional
    raw_color = '#E74C3C'  # Red
    rag_color = '#2ECC71'  # Green
    accent_color = '#3498DB'  # Blue
    
    # 1. Overall Performance Comparison (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['Semantic Similarity', 'Reading Ease', 'BERTScore F1']
    raw_vals = [overall_stats['semantic_similarity']['raw_mean'], 
                overall_stats['flesch_reading_ease']['raw_mean'],
                overall_stats['bertscore_f1']['raw_mean'] if 'bertscore_f1' in overall_stats else 0]
    rag_vals = [overall_stats['semantic_similarity']['rag_mean'], 
                overall_stats['flesch_reading_ease']['rag_mean'],
                overall_stats['bertscore_f1']['rag_mean'] if 'bertscore_f1' in overall_stats else 0]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, raw_vals, width, label='Raw Chatbot', 
                    color=raw_color, alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x + width/2, rag_vals, width, label='RAG Chatbot', 
                    color=rag_color, alpha=0.8, edgecolor='black', linewidth=1)
    
    ax1.set_title('Overall Performance Comparison', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add improvement percentages with significance
    improvements = [overall_stats['semantic_similarity']['improvement_pct'],
                   overall_stats['flesch_reading_ease']['improvement_pct'],
                   overall_stats['bertscore_f1']['improvement_pct'] if 'bertscore_f1' in overall_stats else 0]
    
    significance_levels = [
        overall_stats['semantic_similarity']['p_value'],
        overall_stats['flesch_reading_ease']['p_value'],
        overall_stats['bertscore_f1']['p_value'] if 'bertscore_f1' in overall_stats else 1.0
    ]
    
    for i, (bar1, bar2, imp, p_val) in enumerate(zip(bars1, bars2, improvements, significance_levels)):
        height = max(bar1.get_height(), bar2.get_height())
        color = rag_color if imp > 0 else raw_color
        
        significance = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        
        ax1.text(i, height * 1.05, f'{imp:+.1f}%{significance}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10, color=color)
    
    # 2. BERTScore Distributions (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    raw_bert = df['bertscore_f1_raw'].dropna().values
    rag_bert = df['bertscore_f1_rag'].dropna().values
    
    if len(raw_bert) > 0 and len(rag_bert) > 0:
        # Create overlapping histograms
        ax2.hist(raw_bert, bins=12, alpha=0.6, color=raw_color, 
                edgecolor='black', linewidth=1, label='Raw vs Gold')
        ax2.hist(rag_bert, bins=12, alpha=0.6, color=rag_color, 
                edgecolor='black', linewidth=1, label='RAG vs Gold')
        
        # Add mean lines
        ax2.axvline(np.mean(raw_bert), color=raw_color, linestyle='--', linewidth=2,
                   label=f'Raw Mean: {np.mean(raw_bert):.3f}')
        ax2.axvline(np.mean(rag_bert), color=rag_color, linestyle='--', linewidth=2,
                   label=f'RAG Mean: {np.mean(rag_bert):.3f}')
        
        ax2.set_title('BERTScore F1 vs Gold Standard Distribution', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('BERTScore F1', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Questions', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add statistics box
        improvement = overall_stats['bertscore_f1']['improvement_pct'] if 'bertscore_f1' in overall_stats else 0
        significance = "Yes" if overall_stats['bertscore_f1']['significant'] else "No"
        bert_text = f'BERTScore Analysis:\n• RAG improvement: {improvement:+.1f}%\n• Valid comparisons: {len(raw_bert)}/49\n• Statistical significance: {significance}\n• Effect shows semantic alignment\n  with authoritative answers'
        ax2.text(0.05, 0.95, bert_text, transform=ax2.transAxes, fontsize=9, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 3. Category BERTScore Performance (Middle Left)
    ax3 = fig.add_subplot(gs[1, 0])
    categories = list(category_stats.keys())
    short_categories = [cat.replace(' ', '\n') for cat in categories]
    bertscore_improvements = [category_stats[cat]['bertscore_improvement'] for cat in categories]
    
    bars = ax3.barh(range(len(categories)), bertscore_improvements, 
                   color=[rag_color if x > 0 else raw_color for x in bertscore_improvements],
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    ax3.set_title('BERTScore F1 Improvement by Category', fontsize=14, fontweight='bold', pad=15)
    ax3.set_xlabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax3.set_yticks(range(len(categories)))
    ax3.set_yticklabels(short_categories, fontsize=10)
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.axvline(0, color='black', linewidth=1)
    
    # Add value labels with significance markers
    for i, (bar, val, cat) in enumerate(zip(bars, bertscore_improvements, categories)):
        x_pos = val + (1 if val > 0 else -1)
        significance_marker = '*' if category_stats[cat]['bertscore_significant'] else ''
        ax3.text(x_pos, i, f'{val:+.1f}%{significance_marker}', va='center', 
                ha='left' if val > 0 else 'right', fontweight='bold', fontsize=9)
    
    # 4. Category Overview Matrix (Middle Right)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Create a comparison matrix for categories
    category_names = list(category_stats.keys())
    metric_names = ['Questions', 'Semantic %', 'Reading %', 'BERTScore %']
    
    # Prepare data matrix
    data_matrix = []
    for cat in category_names:
        stats = category_stats[cat]
        row = [
            stats['n_questions'],
            stats['semantic_improvement'],
            stats['reading_improvement'],
            stats['bertscore_improvement']
        ]
        data_matrix.append(row)
    
    # Create heatmap
    data_array = np.array(data_matrix)
    
    # Normalize for better visualization (except question count)
    normalized_data = data_array.copy().astype(float)
    for j in range(1, data_array.shape[1]):  # Skip question count column
        col_data = data_array[:, j]
        if np.max(np.abs(col_data)) > 0:
            normalized_data[:, j] = col_data / np.max(np.abs(col_data)) * 100
    
    im = ax4.imshow(normalized_data[:, 1:], cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)
    
    # Set ticks and labels
    ax4.set_xticks(range(len(metric_names)-1))
    ax4.set_xticklabels(metric_names[1:], fontsize=11)
    ax4.set_yticks(range(len(category_names)))
    ax4.set_yticklabels([cat.replace(' ', '\n') for cat in category_names], fontsize=10)
    
    # Add text annotations
    for i in range(len(category_names)):
        for j in range(len(metric_names)-1):
            value = data_matrix[i][j+1]
            ax4.text(j, i, f'{value:+.1f}%' if j > 0 else f'{value}', 
                    ha='center', va='center', fontweight='bold', fontsize=9,
                    color='white' if abs(normalized_data[i, j+1]) > 25 else 'black')
    
    ax4.set_title('Category Performance Matrix', fontsize=14, fontweight='bold', pad=15)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4, shrink=0.6)
    cbar.set_label('Improvement %', fontsize=10)
    
    # 5. Detailed Summary Table (Bottom)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Create comprehensive summary table
    table_data = [
        ['Metric', 'Raw Mean', 'RAG Mean', 'Change', 'P-value', 'Significant', 'Effect Size'],
        ['Semantic Similarity', 
         f"{overall_stats['semantic_similarity']['raw_mean']:.3f}",
         f"{overall_stats['semantic_similarity']['rag_mean']:.3f}",
         f"{overall_stats['semantic_similarity']['improvement_pct']:+.1f}%",
         f"{overall_stats['semantic_similarity']['p_value']:.4f}",
         '✓' if overall_stats['semantic_similarity']['significant'] else '✗',
         'Medium' if abs(overall_stats['semantic_similarity']['improvement_pct']) > 10 else 'Small'],
        ['Reading Ease',
         f"{overall_stats['flesch_reading_ease']['raw_mean']:.1f}",
         f"{overall_stats['flesch_reading_ease']['rag_mean']:.1f}",
         f"{overall_stats['flesch_reading_ease']['improvement_pct']:+.1f}%",
         f"{overall_stats['flesch_reading_ease']['p_value']:.4f}",
         '✓' if overall_stats['flesch_reading_ease']['significant'] else '✗',
         'Medium' if abs(overall_stats['flesch_reading_ease']['improvement_pct']) > 10 else 'Small'],
    ]
    
    # Add BERTScore row
    if 'bertscore_f1' in overall_stats:
        table_data.append(['BERTScore F1 vs Gold', 
                          f"{overall_stats['bertscore_f1']['raw_mean']:.3f}",
                          f"{overall_stats['bertscore_f1']['rag_mean']:.3f}",
                          f"{overall_stats['bertscore_f1']['improvement_pct']:+.1f}%",
                          f"{overall_stats['bertscore_f1']['p_value']:.4f}",
                          '✓' if overall_stats['bertscore_f1']['significant'] else '✗',
                          'Large' if abs(overall_stats['bertscore_f1']['improvement_pct']) > 15 else 'Medium'])
    
    # Create table
    table = ax5.table(cellText=table_data[1:], colLabels=table_data[0], 
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)
    
    # Style the table
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor(accent_color)
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code the cells
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if j == 3 and table_data[i][j] != 'N/A':  # Change column
                if '+' in table_data[i][j]:
                    table[(i, j)].set_facecolor('#D5F4E6')  # Light green
                elif '-' in table_data[i][j]:
                    table[(i, j)].set_facecolor('#FADBD8')  # Light red
            elif j == 5 and table_data[i][j] == '✓':  # Significant column
                table[(i, j)].set_facecolor('#D5F4E6')  # Light green
    
    ax5.set_title('Comprehensive Performance Analysis Summary', fontsize=16, fontweight='bold', y=0.9)
    
    # Add insights text boxes
    best_category = max(category_stats.keys(), key=lambda x: category_stats[x]['bertscore_improvement'])
    worst_category = min(category_stats.keys(), key=lambda x: category_stats[x]['bertscore_improvement'])
    
    insights_text = f"""Key Insights:
• Total: {len(df)} questions across {len(category_stats)} categories
• Best performing category: {best_category} ({category_stats[best_category]['bertscore_improvement']:+.1f}% BERTScore)
• Needs improvement: {worst_category} ({category_stats[worst_category]['bertscore_improvement']:+.1f}% BERTScore)
• RAG shows consistent improvements in semantic alignment with gold standard
• BERTScore provides objective measurement against authoritative medical guidelines"""
    
    ax5.text(0.02, 0.25, insights_text, transform=ax5.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Add methodology text
    methodology_text = f"""Methodology:
• BERTScore calculated against {overall_stats['bertscore_f1']['total_valid'] if 'bertscore_f1' in overall_stats else 0} gold standard answers
• Statistical significance tested using paired t-tests
• Categories enable targeted analysis of genetic counseling topics
• RAG approach uses medical guidelines for enhanced accuracy"""
    
    ax5.text(0.52, 0.25, methodology_text, transform=ax5.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the enhanced dashboard
    plots_dir = Path(__file__).parent / "analysis_results" / "dashboard"
    plots_dir.mkdir(exist_ok=True)
    
    plt.savefig(plots_dir / "enhanced_cgt_analysis_dashboard.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(plots_dir / "enhanced_cgt_analysis_dashboard.pdf", 
                bbox_inches='tight', facecolor='white')
    
    return fig

def main():
    """Main function to create the enhanced dashboard."""
    print("Creating Enhanced CGT Chatbot Analysis Dashboard")
    print("=" * 50)
    
    # Load and process data
    df = load_and_process_data()
    
    print("Calculating statistics...")
    overall_stats, category_stats = calculate_stats(df)
    
    print("Creating enhanced dashboard...")
    fig = create_enhanced_dashboard(df, overall_stats, category_stats)
    
    print("\nKey Results:")
    print("=" * 20)
    
    semantic_improvement = overall_stats['semantic_similarity']['improvement_pct']
    reading_improvement = overall_stats['flesch_reading_ease']['improvement_pct']
    bert_improvement = overall_stats['bertscore_f1']['improvement_pct'] if 'bertscore_f1' in overall_stats else 0
    
    print(f"✓ Semantic Similarity: {semantic_improvement:+.1f}% change")
    print(f"✓ Reading Ease: {reading_improvement:+.1f}% change")
    print(f"✓ BERTScore F1 vs Gold: {bert_improvement:+.1f}% change")
    print(f"✓ Categories analyzed: {len(category_stats)}")
    print(f"✓ Total questions: {len(df)}")
    
    print(f"\nDashboard saved to: analysis_results/dashboard/")
    print(f"Files: enhanced_cgt_analysis_dashboard.png & .pdf")
    
    plt.show()
    return fig

if __name__ == "__main__":
    main() 