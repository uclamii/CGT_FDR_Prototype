# CGT FDR Prototype

This repository contains a comprehensive prototype for a Cancer Genetic Testing (CGT) chatbot system that helps patients understand their genetic test results and cancer risks. The system uses advanced language models with Retrieval-Augmented Generation (RAG) to provide clear, accurate, and empathetic responses to patient questions about genetic testing and hereditary cancer syndromes.

## 🚀 Features

### **Chatbot Implementations**
- **`chatbot_raw.py`**: Direct implementation using Microsoft Phi-4 model with comprehensive evaluation
- **`chatbot_rag.py`**: RAG implementation combining Phi-4 with medical PDF guidelines
- **Memory-optimized**: Designed to prevent system crashes with efficient memory management
- **Question Categorization**: Automatic classification into 5 genetic counseling domains

### **Comprehensive Answer Evaluation & Analysis**
- **`answer_evaluator.py`**: Comprehensive evaluation metrics including:
  - **BERTScore vs Gold Standard**: Precision, recall, and F1 scores against authoritative medical answers
  - **Semantic similarity** between questions and answers using sentence transformers
  - **Readability metrics** (Flesch reading ease and grade level)
  - **Answer length analysis** (character count, word count, sentence count, average sentence length)
  - **Sentiment analysis** (polarity and subjectivity)
  - **Gold standard coverage** tracking

### **Question Categorization System**
Questions are automatically categorized into 5 genetic counseling domains:
- **Genetic Variant Interpretation** (5 questions)
- **Inheritance Patterns** (7 questions) 
- **Family Risk Assessment** (11 questions)
- **Gene-Specific Recommendations** (14 questions)
- **Support and Resources** (12 questions)

### **Data Processing & Migration Tools**
- **`regenerate_csvs.py`**: Convert existing CSV files to current format with comprehensive metrics
- **Answer cleaning utilities**: Remove formatting artifacts and improve quality
- **Re-evaluation capabilities**: Assess impact of data cleaning on metrics
- **Batch processing**: Handle large question sets efficiently

### **Analysis & Visualization**
- **Dashboard capabilities**: Interactive visualizations and statistical analysis
- **Statistical comparison**: Paired t-tests and effect size calculations between approaches
- **Category-specific analysis**: Performance metrics by question domain
- **Publication-quality plots**: Professional visualizations for research reports

## 🗂️ Architecture

### Raw Chatbot Approach 
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Question  │────▶│   Phi-4     │────▶│   Answer    │
└─────────────┘     │   Model     │     └─────────────┘
                    └─────────────┘           │
                                              ▼
                                    ┌─────────────────┐
                                    │ Comprehensive   │
                                    │ Evaluation      │
                                    │ • BERTScore     │
                                    │ • Readability   │
                                    │ • Categorization│
                                    └─────────────────┘
```

### RAG Approach 
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Question  │────▶│  Vector     │────▶│  Medical    │
│ +Category   │     │  Search     │     │ Guidelines  │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Answer    │◀────│   Phi-4     │◀────│  Context    │
└─────────────┘     │   Model     │     │   Prompt    │
      │             └─────────────┘     └─────────────┘
      │
      ▼
┌─────────────────┐     ┌─────────────┐     ┌─────────────┐
│ Comprehensive   │────▶│ Statistical │────▶│  Dashboard  │
│ Evaluation      │     │  Analysis   │     │ & Reports   │
│ • BERTScore     │     │ • t-tests   │     │ • Category  │
│ • Gold Standard │     │ • Effect    │     │ • Comparison│
│ • 16 Metrics    │     │   Size      │     │ • Visuals   │
└─────────────────┘     └─────────────┘     └─────────────┘
```

### Two-Step Answer Generation
1. **RAG Context**: Retrieves relevant medical guidelines from PDF documents
2. **LLM Knowledge**: Falls back to model's general medical knowledge if needed
3. **Quality Assurance**: Comprehensive evaluation with 16 metrics including BERTScore
4. **Gold Standard Comparison**: Evaluates against authoritative medical answers

## 📋 Requirements

### Core Dependencies
```
torch>=1.9.0
transformers>=4.20.0
langchain>=0.0.200
langchain-community>=0.0.20
sentence-transformers>=2.2.0
chromadb>=0.3.0
textstat>=0.7.0
textblob>=0.17.0
bert-score>=0.3.13
PyPDF2>=3.0.0
```

### Analysis Dependencies
```
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.9.0
plotly>=5.0.0
```

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone [repository-url]
cd CGT_FDR_Prototype
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up directory structure:**
```
CGT_FDR_Prototype/
├── guidelines/              # PDF medical guidelines (NCCN, UpToDate, etc.)
├── qa_outputs/             # Generated results with comprehensive metrics
│   ├── questions_answers_raw.csv      # Raw chatbot results
│   ├── questions_answers_rag.csv      # RAG chatbot results  
│   └── questions_answers_comparison.csv # Merged comparison for dashboard
├── config/                 # Configuration files
│   └── question_categories.json      # Question categories definition
├── helper/                 # Helper functions and utilities
│   └── clean_results.py            # Answer cleaning functionality
├── chroma_db/              # Vector database for RAG
├── archive/                # Archived older versions
├── questions.txt           # Input questions (49 total)
├── questions_answers.txt   # Gold standard answers for BERTScore
├── chatbot_raw.py         # Raw chatbot implementation
├── chatbot_rag.py         # RAG chatbot implementation (recommended)
├── answer_evaluator.py    # Comprehensive evaluation metrics
├── regenerate_csvs.py     # Convert existing CSVs to current format
└── README.md
```

## 🚀 Usage

### 1. Generate Answers

**RAG Chatbot (Recommended):**
```bash
python chatbot_rag.py
```
- Generates `qa_outputs/questions_answers_rag.csv`
- Uses PDF guidelines for context-aware responses
- Includes 16 comprehensive evaluation metrics

**Raw Chatbot (Baseline):**
```bash
python chatbot_raw.py
```
- Generates `qa_outputs/questions_answers_raw.csv`
- Direct model responses without guidelines
- Same comprehensive evaluation metrics for comparison

### 2. Convert Existing Data (if needed)

**Upgrade existing CSV files to current format:**
```bash
python regenerate_csvs.py
```
- Converts old CSV format to current format with 16 metrics
- Adds BERTScore evaluation against gold standard
- Creates comparison CSV for dashboard use

### 3. Key Output Files

| File | Description | Metrics |
|------|-------------|---------|
| `questions_answers_raw.csv` | Raw chatbot results | 16 comprehensive metrics |
| `questions_answers_rag.csv` | RAG chatbot results | 16 comprehensive metrics |  
| `questions_answers_comparison.csv` | Side-by-side comparison | Raw vs RAG with _raw/_rag suffixes |

## 📊 Evaluation Metrics

### **Quality Metrics (16 Total)**
- **Semantic Similarity**: Relevance between question and answer (0-1)
- **Answer Length**: Character count analysis  
- **Word Count**: Token-level analysis
- **Readability**: Flesch reading ease and Flesch-Kincaid grade level
- **Sentence Analysis**: Count and average sentence length
- **Sentiment**: Polarity (-1 to 1) and subjectivity (0-1)
- **BERTScore vs Gold Standard**: Precision, recall, and F1 scores
- **Gold Standard Coverage**: Boolean flag for availability of authoritative answers

### **BERTScore Evaluation**
- Uses `distilbert-base-uncased` model for semantic similarity
- Compares generated answers against gold standard medical answers from `questions_answers.txt`
- Provides precision, recall, and F1 scores for comprehensive evaluation
- Critical for grant reporting and publication requirements

### **Question Categories**
All 49 questions are automatically categorized into:
- **Genetic Variant Interpretation**: Understanding test results and implications
- **Inheritance Patterns**: Heredity, family transmission, reproductive options
- **Family Risk Assessment**: Testing recommendations for relatives
- **Gene-Specific Recommendations**: Cancer risks and screening by gene (BRCA1/2, Lynch syndrome genes)
- **Support and Resources**: Insurance, emotional support, finding care

## 🎯 Performance Optimizations

### **Memory Management**
- Optimized token generation limits (200-300 tokens)
- Memory clearing after each question
- Efficient model loading with `device_map="auto"`
- CPU-based embeddings to preserve GPU memory

### **Quality Improvements**
- Advanced prompting strategies for medical context
- Minimum answer length requirements (30+ characters)
- Improved context retrieval (5 relevant document chunks)
- Two-step generation with fallback mechanisms

### **Evaluation Features**
- BERTScore integration for semantic similarity vs gold standard
- Comprehensive readability analysis for patient communication
- Category-specific performance tracking
- Statistical significance testing capabilities

## 📈 Expected Results

Based on comprehensive testing, the RAG approach typically shows:
- **Improved BERTScore** vs gold standard medical answers
- **Better semantic similarity** (improved relevance to questions)
- **Longer, more detailed answers** with medical context
- **Better use of authoritative medical guidelines**
- **More consistent quality** across all 5 question categories
- **Higher statistical significance** in paired comparisons

## 🔧 Configuration

### **Model Settings**
```python
MODEL_NAME = "microsoft/phi-4"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BERTSCORE_MODEL = "distilbert-base-uncased"
MAX_NEW_TOKENS = 300  # Increased for detailed answers
MIN_NEW_TOKENS = 50   # Ensures comprehensive responses
TEMPERATURE = 0.7
TOP_P = 0.9
```

### **RAG Settings**
```python
CHUNK_SIZE = 1000      # Increased for better context
CHUNK_OVERLAP = 200    # Improved coherence
RETRIEVAL_K = 5        # More comprehensive document retrieval
```

### **Question Categories Configuration**
Question categories are stored in `config/question_categories.json` and can be modified to:
- Add new question categories
- Update existing questions in categories
- Modify category names

The config file structure:
```json
{
  "question_categories": {
    "Category Name": [
      "Question 1",
      "Question 2",
      ...
    ]
  }
}
```

Current categories:
- **Genetic Variant Interpretation** (5 questions)
- **Inheritance Patterns** (7 questions)
- **Family Risk Assessment** (11 questions)
- **Gene-Specific Recommendations** (14 questions)
- **Support and Resources** (12 questions)

### **Evaluation Settings**
```python
FIELDNAMES = [
    "Question", "Answer", "Category",
    "semantic_similarity", "answer_length", "word_count", "char_count",
    "flesch_reading_ease", "flesch_kincaid_grade", "sentence_count", "avg_sentence_length",
    "sentiment_polarity", "sentiment_subjectivity",
    "bertscore_precision", "bertscore_recall", "bertscore_f1", "has_gold_standard"
]
```

## 🚨 Troubleshooting

### **Memory Issues**
- Reduce `max_new_tokens` in generation settings
- Use CPU for embeddings: `model_kwargs={'device': 'cpu'}`
- Clear cache regularly: `torch.cuda.empty_cache()` or `torch.mps.empty_cache()`

### **BERTScore Issues**
- Ensure `questions_answers.txt` exists with gold standard answers
- Verify BERTScore model download: `bert-score` package
- Check GPU/CPU availability for BERTScore computation

### **Quality Issues**
- Verify PDF documents in `guidelines/` directory
- Check question format in `questions.txt` (49 questions expected)
- Review gold standard answers in `questions_answers.txt`
- Validate CSV format with 16 metrics

### **File Structure Issues**
- Run `regenerate_csvs.py` to convert old format CSVs
- Check for `_enhanced` suffix files (these are legacy versions)
- Verify `qa_outputs/` directory exists and is writable

## 📊 Data Files Structure

### **Input Files**
- `questions.txt`: 49 genetic counseling questions
- `questions_answers.txt`: Gold standard answers for BERTScore evaluation
- `guidelines/*.pdf`: Medical guidelines (NCCN, UpToDate, etc.)

### **Output Files**
- `qa_outputs/questions_answers_raw.csv`: Raw chatbot results (16 metrics)
- `qa_outputs/questions_answers_rag.csv`: RAG chatbot results (16 metrics)
- `qa_outputs/questions_answers_comparison.csv`: Merged comparison with _raw/_rag suffixes

### **Archive**
- `archive/`: Contains older versions of scripts that have been updated
- See `archive/README.md` for details on what was archived and why

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure all 16 evaluation metrics are properly implemented
5. Test with both raw and RAG approaches
6. Submit a pull request

## 📄 License

[Add your license information here]

## 🤝 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the evaluation metrics documentation
3. Verify BERTScore and gold standard setup
4. Open an issue with detailed error information and system specifications

---

**Note**: This system is designed for research and development purposes in genetic counseling and testing. The evaluation metrics, including BERTScore comparison against gold standard medical answers, are designed to meet grant reporting and publication requirements. Always consult with healthcare professionals for actual medical advice.
