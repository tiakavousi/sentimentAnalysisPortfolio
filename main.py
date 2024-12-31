import tensorflow as tf
import numpy as np
from transformers import DistilBertTokenizer
from data.data_processing import DataProcessor
from config.model_config import Config
from models.sentiment_model import EnhancedDistilBertForSentiment, ModelTrainer
from models.modelPersistence import ModelPersistence


class SentimentAnalyzer:
    """
        Manages sentiment analysis pipeline using DistilBERT with enhanced features.
        Handles data processing, model initialization, training, and prediction
    """
    def __init__(self, model=None):
        """
            Initialize analyzer components without loading data or model.
            
            Args:
            model: Optional pre-trained model instance
        """
        self.data_processor = DataProcessor()
        self.model = model
        self.tokenizer = None
        self.trainer = None
        
        # Data storage
        self.processed_data = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
   
    def process_data(self):
        """
        Process raw data using DataProcessor.
        
        Returns:
            Dict containing processed dataframes and model inputs
        """
        processed_data = self.data_processor.prepare_data()
        self.processed_data = processed_data
        return processed_data
    

    def initialize_model(self):
        """
        Initialize DistilBERT model, tokenizer, and prepare training datasets.
        Creates train and validation datasets from processed data.
        """
        self.tokenizer = DistilBertTokenizer.from_pretrained(Config.BERT_MODEL)
        self.model = EnhancedDistilBertForSentiment()
        self.trainer = ModelTrainer(self.model, self.tokenizer)

        # Extract data from processed_data
        train_texts = self.processed_data['dataframes']['train']['processed_text'].to_numpy()
        val_texts = self.processed_data['dataframes']['val']['processed_text'].to_numpy()
        
        train_labels = {'sentiment': self.processed_data['model_inputs'][1]['sentiment']}
        val_labels = {'sentiment': self.processed_data['model_inputs'][3]['sentiment']}

        # Prepare datasets
        self.train_dataset = self.trainer.prepare_dataset(train_texts, train_labels)
        self.val_dataset = self.trainer.prepare_dataset(val_texts, val_labels)
        
    def predict(self, text):
        """
        Analyze sentiment and sarcasm in input text.
        
        Args:
            text: String to analyze
            
        Returns:
            Dict containing sentiment probabilities, predicted sentiment,
            and sarcasm detection results
        """
        processed_text = self.data_processor.preprocess_text(text)
        inputs = self.tokenizer(
            processed_text,
            return_tensors='tf',
            padding=True,
            truncation=True,
            max_length=Config.MAX_LENGTH
         )
            
        # Get model predictions
        sentiment_logits = self.model(inputs)
        sentiment_probs = tf.nn.softmax(sentiment_logits, axis=-1).numpy()[0]
        
        # Map probabilities to sentiment labels
        sentiments = ['negative', 'neutral', 'positive']
        
        # Detect sarcasm using the data processor's detector
        _, sarcasm_info = self.data_processor.sarcasm_detector.detect_sarcasm(processed_text)
        
        return {
            'sentiment': {label: float(prob) for label, prob in zip(sentiments, sentiment_probs)},
            'predicted': sentiments[np.argmax(sentiment_probs)],
            'sarcasm_detected': bool(sarcasm_info is not None),
            'sarcasm_type': sarcasm_info if sarcasm_info else None
        }
    
    def verify_model_architecture(self):
        """
            Verify model architecture using dummy input and display model summary.
            Builds model by running a test prediction.
        """        # Create dummy input
        dummy_input = self.tokenizer(
            "This is a dummy text",
            return_tensors='tf',
            padding=True,
            truncation=True,
            max_length=Config.MAX_LENGTH
        )
        
        # Convert to proper format for model input
        features = {
            'input_ids': tf.convert_to_tensor(dummy_input['input_ids'], dtype=tf.int32),
            'attention_mask': tf.convert_to_tensor(dummy_input['attention_mask'], dtype=tf.int32)
        }
        
        # Run dummy input through model to build it
        _ = self.model(features)

        # Now we can print the summary
        print("\nModel Architecture Summary:")
        self.model.summary()
            

    def cleanup(self):
        """
        Release memory and clear TensorFlow session.
        Called after training or when shutting down the analyzer.
        """        
        try:
            tf.keras.backend.clear_session()
            self.train_dataset = None
            self.val_dataset = None
            self.model = None
            self.trainer = None
            self.processed_data = None
            
        except Exception as e:
            print(f"Warning: Cleanup failed: {str(e)}")


def main():
    """
    Main entry point for training the sentiment analyzer.
    Processes data, initializes model, trains, and performs cleanup.
    
    Returns:
        Training history if successful, None if training fails
    """    
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