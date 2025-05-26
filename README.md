# CGT FDR Prototype

This repository contains a prototype for a Genetic Counseling and Testing (CGT) chatbot system that helps patients understand their genetic test results and cancer risks. The system uses advanced language models to provide clear, accurate, and empathetic responses to patient questions about genetic testing and hereditary cancer syndromes.

## Features

- **Two Chatbot Implementations**:
  - `chatbot_raw.py`: A direct implementation using the Microsoft Phi-4 model
  - `chatbot_rag.py`: A Retrieval-Augmented Generation (RAG) implementation that combines the Phi-4 model with relevant medical guidelines

- **Answer Evaluation**:
  - Semantic similarity between questions and answers
  - Readability metrics (Flesch reading ease and grade level)
  - Answer length analysis
  - Sentiment analysis (polarity and subjectivity)

## Architecture

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

### RAG Chatbot Approach
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Question  │────▶│  Document   │────▶│  Relevant   │
└─────────────┘     │  Retrieval  │     │  Context    │
                    └─────────────┘     └─────────────┘
                                              │
                                              ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Answer    │◀────│   Phi-4     │◀────│  Combined   │
└─────────────┘     │   Model     │     │  Input      │
      │             └─────────────┘     └─────────────┘
      │
      ▼
┌─────────────┐
│ Evaluation  │
│  Metrics    │
└─────────────┘
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- LangChain
- Sentence Transformers
- TextStat
- TextBlob
- ChromaDB

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd CGT_FDR_Prototype
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Set up the directory structure:
```
CGT_FDR_Prototype/
├── guidelines/           # PDF files containing medical guidelines
├── qa_outputs/          # Output CSV files with questions and answers
├── questions.txt        # Input questions file
├── chatbot_raw.py       # Raw chatbot implementation
├── chatbot_rag.py       # RAG-based chatbot implementation
└── answer_evaluator.py  # Answer evaluation module
```

## Usage

### Running the Raw Chatbot

```bash
python chatbot_raw.py
```

This will:
1. Load the Phi-4 model
2. Process questions from `questions.txt`
3. Generate answers using the model
4. Evaluate the answers using various metrics
5. Save results to `qa_outputs/questions_answers_llm.csv`

### Running the RAG Chatbot

```bash
python chatbot_rag.py
```

This will:
1. Load the Phi-4 model
2. Process and index medical guidelines from the `guidelines` directory
3. Process questions from `questions.txt`
4. Generate answers using both the guidelines and model knowledge
5. Evaluate the answers using various metrics
6. Save results to `qa_outputs/questions_answers_rag.csv`

## Output Format

Both implementations generate CSV files with the following columns:
- Question: The input question
- Answer: The generated response
- semantic_similarity: Relevance score between question and answer
- answer_length: Number of words in the answer
- flesch_reading_ease: Readability score (higher = easier to read)
- flesch_kincaid_grade: Reading grade level required
- sentiment_polarity: Sentiment score (-1 to 1)
- sentiment_subjectivity: How objective/subjective the answer is (0-1)

## Model Details

The system uses Microsoft's Phi-4 model, which is optimized for:
- Clear and accurate medical explanations
- Patient-friendly language
- Consistent and reliable responses
- Appropriate handling of sensitive medical information

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add appropriate license information]

## Contact

[Add contact information]


