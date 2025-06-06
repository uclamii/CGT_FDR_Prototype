# Archived Scripts

This folder contains outdated scripts that have been superseded by enhanced versions. These are kept for reference but should not be used in current workflows.

## Archived Scripts and Their Replacements

### Core Chatbot Scripts
- **`chatbot_raw.py`** → Replaced by `chatbot_raw_enhanced.py`
  - Original raw chatbot implementation
  - Missing comprehensive metrics and BERTScore evaluation
  
- **`chatbot_rag.py`** → Replaced by `chatbot_rag_enhanced.py`
  - Original RAG chatbot implementation
  - Missing enhanced metrics and proper CSV formatting

### Evaluation Scripts
- **`answer_evaluator.py`** → Replaced by `enhanced_answer_evaluator.py`
  - Original basic evaluator with limited metrics
  - Missing BERTScore vs gold standard evaluation
  - Missing additional quality metrics

### Analysis Scripts
- **`analyze_by_category.py`** → Functionality incorporated into enhanced scripts
  - Category analysis functionality now built into enhanced pipeline
  - Replaced by comprehensive analysis in enhanced scripts

- **`analyze_results.py`** → Replaced by enhanced analysis pipeline
  - Large monolithic analysis script (1130+ lines)
  - Functionality distributed across enhanced scripts
  - Replaced by modular enhanced analysis system

### Dashboard Scripts  
- **`enhanced_analysis_with_bertscore.py`** → Replaced by `create_enhanced_dashboard.py`
  - Earlier version of enhanced analysis
  - Missing comprehensive visualization features
  
- **`create_comprehensive_dashboard.py`** → Replaced by `create_enhanced_dashboard.py`
  - Earlier comprehensive dashboard version
  - Missing latest enhancements and proper data handling

## Current Enhanced Pipeline

The current workflow uses these enhanced scripts:

1. **`enhanced_answer_evaluator.py`** - Comprehensive evaluation with BERTScore vs gold standard
2. **`chatbot_raw_enhanced.py`** - Enhanced raw chatbot with full metrics
3. **`chatbot_rag_enhanced.py`** - Enhanced RAG chatbot with full metrics  
4. **`regenerate_enhanced_csvs.py`** - Script to regenerate CSVs with enhanced metrics
5. **`create_enhanced_dashboard.py`** - Comprehensive 3x3 grid dashboard
6. **`create_simplified_dashboard.py`** - Clean simplified dashboard (still in use)

## Key Improvements in Enhanced Versions

- **BERTScore evaluation** against gold standard medical answers
- **Comprehensive metrics** including word count, sentence analysis, quality measures
- **Proper CSV formatting** with `_raw` and `_rag` suffixes for dashboard compatibility
- **Category classification** automatic categorization into genetic counseling domains
- **Enhanced visualizations** with statistical significance testing
- **Modular architecture** for better maintainability
- **Memory optimization** and error handling improvements

## Archive Date
Scripts archived: January 2025

## Notes
These scripts are preserved for historical reference and potential code reuse, but the enhanced versions should be used for all current and future work. 