# CGT FDR Prototype

This repository contains a comprehensive prototype for a Genetic Counseling and Testing (CGT) chatbot system that helps patients understand their genetic test results and cancer risks. The system uses advanced language models with Retrieval-Augmented Generation (RAG) to provide clear, accurate, and empathetic responses to patient questions about genetic testing and hereditary cancer syndromes.

## 🚀 Features

### **Chatbot Implementations**
- **`chatbot_raw.py`**: Direct implementation using Microsoft Phi-4 model
- **`chatbot_rag.py`**: RAG-enhanced implementation combining Phi-4 with medical guidelines
- **Memory-optimized**: Designed to prevent system crashes with efficient memory management

### **Answer Evaluation & Analysis**
- **`answer_evaluator.py`**: Comprehensive evaluation metrics including:
  - Semantic similarity between questions and answers
  - Readability metrics (Flesch reading ease and grade level)
  - Answer length analysis
  - Sentiment analysis (polarity and subjectivity)

### **Results Analysis Tools**
- **`analyze_results.py`**: Statistical comparison between raw and RAG approaches
  - Paired t-tests and Wilcoxon signed-rank tests
  - Effect size calculations (Cohen's d)
  - Comprehensive visualizations and reports
  - Publication-quality plots and statistical analysis

### **Data Processing**
- **Answer cleaning utilities**: Remove formatting artifacts and improve quality
- **Re-evaluation capabilities**: Assess impact of data cleaning on metrics
- **Batch processing**: Handle large question sets efficiently

## ��️ Architecture

### Raw Chatbot Approach 
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Question  │────▶│   Phi-4     │────▶│   Answer    │
└─────────────┘     │   Model     │     └─────────────┘
                    └─────────────┘           │
                                              ▼
                                        ┌─────────────┐
                                        │ Evaluation  │
                                        │  Metrics    │
                                        └─────────────┘
```

### RAG-Enhanced Approach 
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Question  │────▶│  Vector     │────▶│  Relevant   │
└─────────────┘     │  Search     │     │ Guidelines  │
                    └─────────────┘     └─────────────┘
                                              │
                                              ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Answer    │◀────│   Phi-4     │◀────│  Enhanced   │
└─────────────┘     │   Model     │     │   Prompt    │
      │             └─────────────┘     └─────────────┘
      │
      ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Evaluation  │────▶│ Statistical │────▶│   Report    │
│  Metrics    │     │  Analysis   │     │ Generation  │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Two-Step Answer Generation
1. **RAG Context**: First attempts to answer using retrieved medical guidelines
2. **LLM Knowledge**: Falls back to model's general medical knowledge if needed
3. **Quality Assurance**: Evaluates and scores all responses

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
```

### Analysis Dependencies
```
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.9.0
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
├── guidelines/              # PDF medical guidelines
├── qa_outputs/             # Generated results
├── analysis_results/       # Analysis outputs
│   ├── plots/             # Visualization files
│   ├── comparison_report.txt
│   └── summary_statistics.csv
├── questions.txt           # Input questions
├── chatbot_raw.py         # Raw chatbot
├── chatbot_rag.py         # RAG chatbot (recommended)
├── answer_evaluator.py    # Evaluation metrics
├── analyze_results.py     # Statistical analysis
└── README.md
```

## 🚀 Usage

### 1. Generate Answers

**RAG Chatbot (Recommended):**
```bash
python chatbot_rag.py
```

**Raw Chatbot (Baseline):**
```bash
python chatbot_raw.py
```

### 2. Analyze Results

**Compare RAG vs Raw Performance:**
```bash
python analyze_results.py
```

This generates:
- **Statistical comparison report**
- **Visualization plots** (box plots, scatter plots, distributions)
- **Summary CSV** with key metrics
- **Recommendations** for deployment

### 3. Key Output Files

| File | Description |
|------|-------------|
| `questions_answers_raw.csv` | Raw chatbot results |
| `questions_answers_rag.csv` | RAG chatbot results |
| `analysis_results/comparison_report.txt` | Detailed statistical analysis |
| `analysis_results/summary_statistics.csv` | Key metrics comparison |
| `analysis_results/plots/` | Visualization files |

## 📊 Evaluation Metrics

### **Quality Metrics**
- **Semantic Similarity**: Relevance between question and answer (0-1)
- **Answer Length**: Character count and word analysis
- **Readability**: Flesch reading ease and grade level
- **Sentiment**: Polarity (-1 to 1) and subjectivity (0-1)

### **Statistical Analysis**
- **Paired t-tests**: Test for significant differences
- **Wilcoxon signed-rank tests**: Non-parametric alternative
- **Effect sizes**: Cohen's d for practical significance
- **Improvement rates**: Percentage of questions with better scores

## 🎯 Performance Optimizations

### **Memory Management**
- Reduced token generation limits (200 tokens max)
- Memory clearing after each question
- Optimized model loading with `device_map="auto"`
- Conservative generation parameters

### **Quality Improvements**
- Two-step answer generation (RAG → LLM fallback)
- Enhanced prompting strategies
- Minimum answer length requirements
- Improved context retrieval (3-5 relevant documents)

## 📈 Expected Results

Based on testing, the RAG approach typically shows:
- **Improved semantic similarity** (better relevance)
- **Longer, more detailed answers**
- **Better use of medical guidelines**
- **More consistent quality across questions**

## 🔧 Configuration

### **Model Settings**
```python
MODEL_NAME = "microsoft/phi-4"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.9
```

### **RAG Settings**
```python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 3  # Number of documents to retrieve
```

## 🚨 Troubleshooting

### **Memory Issues**
- Reduce `max_new_tokens` in generation settings
- Use CPU for embeddings: `model_kwargs={'device': 'cpu'}`
- Clear cache regularly: `torch.cuda.empty_cache()`

### **Quality Issues**
- Check guideline documents in `guidelines/` directory
- Verify question format in `questions.txt`
- Review prompting strategies in the code

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🤝 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the analysis reports for insights
3. Open an issue with detailed error information

---

**Note**: This system is designed for research and development purposes. Always consult with healthcare professionals for actual medical advice.
