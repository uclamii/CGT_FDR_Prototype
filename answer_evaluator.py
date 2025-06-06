import textstat
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score as bert_score
import numpy as np
from pathlib import Path

class EnhancedAnswerEvaluator:
    """Enhanced evaluator with BERTScore and gold standard comparison."""
    
    def __init__(self):
        """Initialize the evaluator with necessary models."""
        self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.gold_standard_answers = self.load_gold_standard()
        print(f"Loaded {len(self.gold_standard_answers)} gold standard answers")
    
    def load_gold_standard(self):
        """Load gold standard answers from questions_answers.txt."""
        base_dir = Path(__file__).parent
        gold_standard_file = base_dir / "questions_answers.txt"
        
        if not gold_standard_file.exists():
            print(f"Warning: Gold standard file not found at {gold_standard_file}")
            return {}
        
        questions_answers = {}
        
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
                questions_answers[question] = answer
        
        return questions_answers
    
    def calculate_bertscore_vs_gold(self, question: str, answer: str) -> dict:
        """Calculate BERTScore against gold standard answer."""
        bertscore_metrics = {
            'bertscore_precision': 0.0,
            'bertscore_recall': 0.0,
            'bertscore_f1': 0.0,
            'has_gold_standard': False
        }
        
        if question not in self.gold_standard_answers:
            return bertscore_metrics
        
        gold_answer = self.gold_standard_answers[question]
        
        if not answer.strip() or not gold_answer.strip():
            return bertscore_metrics
        
        try:
            # Calculate BERTScore using distilbert-base-uncased for efficiency
            P, R, F1 = bert_score([answer], [gold_answer], 
                                 model_type="distilbert-base-uncased", 
                                 verbose=False, device='cpu')
            
            bertscore_metrics.update({
                'bertscore_precision': float(P[0].item()),
                'bertscore_recall': float(R[0].item()),
                'bertscore_f1': float(F1[0].item()),
                'has_gold_standard': True
            })
            
        except Exception as e:
            print(f"Error calculating BERTScore for question '{question}': {str(e)}")
        
        return bertscore_metrics
    
    def evaluate_answer(self, question: str, answer: str) -> dict:
        """Comprehensive answer evaluation including BERTScore."""
        metrics = {}
        
        try:
            # Calculate embeddings
            question_embedding = self.embedding_model.encode([question], convert_to_tensor=True)
            answer_embedding = self.embedding_model.encode([answer], convert_to_tensor=True)
            
            # Convert to numpy for similarity calculations
            question_embedding_np = question_embedding.cpu().numpy()
            answer_embedding_np = answer_embedding.cpu().numpy()
            
            # Calculate semantic similarity between question and answer
            metrics['semantic_similarity'] = float(cosine_similarity(question_embedding_np, answer_embedding_np)[0][0])
            
            # Calculate readability metrics
            metrics['answer_length'] = len(answer.split())
            metrics['flesch_reading_ease'] = textstat.flesch_reading_ease(answer)
            metrics['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(answer)
            
            # Calculate sentiment
            sentiment = TextBlob(answer).sentiment
            metrics['sentiment_polarity'] = sentiment.polarity
            metrics['sentiment_subjectivity'] = sentiment.subjectivity
            
            # Calculate BERTScore vs gold standard
            bertscore_metrics = self.calculate_bertscore_vs_gold(question, answer)
            metrics.update(bertscore_metrics)
            
            # Additional quality metrics
            metrics['word_count'] = len(answer.split())
            metrics['char_count'] = len(answer)
            metrics['sentence_count'] = len([s for s in answer.split('.') if s.strip()])
            metrics['avg_sentence_length'] = metrics['word_count'] / max(metrics['sentence_count'], 1)
            
        except Exception as e:
            print(f"Error in metrics calculation: {str(e)}")
            # Set default values for metrics
            metrics = {
                'semantic_similarity': 0.0,
                'answer_length': 0,
                'flesch_reading_ease': 0.0,
                'flesch_kincaid_grade': 0.0,
                'sentiment_polarity': 0.0,
                'sentiment_subjectivity': 0.0,
                'bertscore_precision': 0.0,
                'bertscore_recall': 0.0,
                'bertscore_f1': 0.0,
                'has_gold_standard': False,
                'word_count': 0,
                'char_count': 0,
                'sentence_count': 0,
                'avg_sentence_length': 0
            }
        
        return metrics 