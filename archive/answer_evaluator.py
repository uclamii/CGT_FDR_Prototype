import textstat
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class AnswerEvaluator:
    """Evaluates answers using various metrics."""
    
    def __init__(self):
        """Initialize the evaluator with necessary models."""
        self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    def evaluate_answer(self, question: str, answer: str) -> dict:
        """Evaluate an answer using various metrics."""
        metrics = {}
        
        try:
            # Calculate embeddings
            question_embedding = self.embedding_model.encode([question], convert_to_tensor=True)
            answer_embedding = self.embedding_model.encode([answer], convert_to_tensor=True)
            
            # Convert to numpy for similarity calculations
            question_embedding_np = question_embedding.cpu().numpy()
            answer_embedding_np = answer_embedding.cpu().numpy()
            
            # Calculate semantic similarity
            metrics['semantic_similarity'] = float(cosine_similarity(question_embedding_np, answer_embedding_np)[0][0])
            
            # Calculate readability metrics
            metrics['answer_length'] = len(answer.split())
            metrics['flesch_reading_ease'] = textstat.flesch_reading_ease(answer)
            metrics['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(answer)
            
            # Calculate sentiment
            sentiment = TextBlob(answer).sentiment
            metrics['sentiment_polarity'] = sentiment.polarity
            metrics['sentiment_subjectivity'] = sentiment.subjectivity
            
        except Exception as e:
            print(f"Error in metrics calculation: {str(e)}")
            # Set default values for metrics
            metrics = {
                'semantic_similarity': 0.0,
                'answer_length': 0,
                'flesch_reading_ease': 0.0,
                'flesch_kincaid_grade': 0.0,
                'sentiment_polarity': 0.0,
                'sentiment_subjectivity': 0.0
            }
        
        return metrics 