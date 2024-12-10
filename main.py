import tensorflow as tf
import numpy as np
from transformers import DistilBertTokenizer
from data.data_processing import DataProcessor
from config.model_config import ModelConfig
from models.sentiment_model import EnhancedDistilBertForSentiment, ModelTrainer
from models.modelPersistence import ModelPersistence


class SentimentAnalyzer:
    """Main class for sentiment analysis using DistilBERT with enhanced features"""

    def __init__(self, model=None):
        """Initialize core components without loading data or model"""
        self.data_processor = DataProcessor()
        self.model = model
        self.tokenizer = None
        self.trainer = None
        
        # Data storage
        self.train_texts = None
        self.val_texts = None
        self.test_texts = None
        self.train_labels = None
        self.val_labels = None
        self.test_labels = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def process_data(self):
        """Load, split and preprocess training data"""
        # First load the raw data
        df = self.data_processor.load_data()
        # Then balance it
        df = self.data_processor.create_balanced_dataset(df)
        # Split data and store results in instance variables
        self.train_texts, self.val_texts, self.test_texts, self.train_labels, self.val_labels, self.test_labels = self.data_processor.split_data(df)
        # self.train_texts, self.val_texts, self.train_labels, self.val_labels = self.data_processor.split_data(df)
        # Return for convenience, but also store in instance
        return self.train_texts, self.val_texts,self.test_texts, self.train_labels, self.val_labels, self.test_labels
    

    def initialize_model(self):
        """Initialize and compile model"""
        if self.train_texts is None:
            raise ValueError("Please run process_data() before initializing the model")
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(ModelConfig.BERT_MODEL)
        self.model = EnhancedDistilBertForSentiment()
        self.trainer = ModelTrainer(self.model, self.tokenizer)

        # Prepare simplified datasets (sentiment labels only)
        self.train_dataset = self.trainer.prepare_dataset(
            self.train_texts, 
            {'sentiment': self.train_labels['sentiment']}
        )
        self.val_dataset = self.trainer.prepare_dataset(
            self.val_texts,
            {'sentiment': self.val_labels['sentiment']}
        )

        
    def predict(self, text):
        """Generate sentiment analysis for input text"""
        processed_text, _ = self.data_processor.preprocess_text(text)
        inputs = self.tokenizer(
            processed_text,
            return_tensors='tf',
            padding=True,
            truncation=True,
            max_length=ModelConfig.MAX_LENGTH
        )
        
        # Get model predictions (now returns single sentiment tensor)
        sentiment_logits = self.model(inputs)
        sentiment_probs = tf.nn.softmax(sentiment_logits, axis=-1).numpy()[0]
        
        # Map probabilities to sentiment labels
        sentiments = ['negative', 'neutral', 'positive']
        return {
            'sentiment': {label: float(prob) for label, prob in zip(sentiments, sentiment_probs)},
            'predicted': sentiments[np.argmax(sentiment_probs)]
        }
    
    def verify_model_architecture(self):
        """ Verify model architecture by running a dummy input through the model and displaying the model summary. """
        # Create dummy input
        dummy_input = self.tokenizer(
            "This is a dummy text",
            return_tensors='tf',
            padding=True,
            truncation=True,
            max_length=ModelConfig.MAX_LENGTH
        )
        
        # Run dummy input through model to get output shape
        output = self.model(dummy_input)

        # Print model summary
        print("\nModel Architecture Summary:")
        self.model.summary()
            

    def cleanup(self):
        """Clean up resources"""
        try:
            tf.keras.backend.clear_session()
            self.train_dataset = None
            self.val_dataset = None
            self.model = None
            self.trainer = None
            
        except Exception as e:
            print(f"Warning: Cleanup failed: {str(e)}")


def main():
    """Initialize model and train"""
    analyzer = None
    try:
        analyzer = SentimentAnalyzer()
        
        print("Processing data...")
        analyzer.process_data()
        
        print("Initializing model...")
        analyzer.initialize_model()
        
        print("\nTraining model...")
        history = analyzer.trainer.train(analyzer.train_dataset, analyzer.val_dataset)

        print("\nModel training complete. Use \"saveAndLoad class to save the trained model.")
        
        return history
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
        
    finally:
        if analyzer:
            analyzer.cleanup()


if __name__ == "__main__":
   main()