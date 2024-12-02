import tensorflow as tf
from transformers import DistilBertTokenizer
import numpy as np
from data.data_processing import DataProcessor
from config.model_config import ModelConfig
from models.sentiment_model import EnhancedDistilBertForSentiment, ModelTrainer


class SentimentAnalyzer:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def process_data(self):
        # Load and process data
        df = self.data_processor.load_data()
        train_texts, val_texts, train_labels, val_labels = self.data_processor.split_data(df)
        
        # Process some example texts to test preprocessing
        test_texts = [
            "Great service and amazing food!",
            "Terrible experience, would not recommend.",
            "The food was okay, but the service could be better.",
            "Yeah right, like that's going to work...",
            "Thanks a lot... now everything is broken 🙄",
            "Obviously this is the best restaurant ever..."
        ]
        
        print("\nPreprocessing Examples:")
        for text in test_texts:
            processed_text = self.data_processor.preprocess_text(text)
            print(f"\nOriginal: {text}")
            print(f"Processed: {processed_text}")
        
        return train_texts, val_texts, train_labels, val_labels
        
    def initialize_model(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained(ModelConfig.BERT_MODEL)
        self.model = EnhancedDistilBertForSentiment()
        self.trainer = ModelTrainer(self.model, self.tokenizer)

        # Compile model
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            ModelConfig.LEARNING_RATE,
            decay_steps=ModelConfig.DECAY_STEPS,
            decay_rate=ModelConfig.DECAY_RATE
        )

         # Define losses for each output
        losses = {
            'sentiment': 'sparse_categorical_crossentropy',
            'sarcasm': 'binary_crossentropy',
            'negation': 'binary_crossentropy',
            'polarity': 'mse'  # for multipolarity score
        }
        # Define metrics for each output
        metrics = {
            'sentiment': ['accuracy'],
            'sarcasm': ['accuracy'],
            'negation': ['accuracy'],
            'polarity': ['mae']
        }

        # Define loss weights
        loss_weights = {
            'sentiment': 1.0,
            'sarcasm': 0.5,
            'negation': 0.5,
            'polarity': 0.3
        }

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=losses,
            metrics=metrics,
            loss_weights=loss_weights
        )

    def train(self):
        """
        Train the model using processed data with multiple outputs.
        Returns training history for visualization.
        """
        # Load and process data
        df = self.data_processor.load_data()
        train_texts, val_texts, train_labels, val_labels = self.data_processor.split_data(df)
        
        # Prepare datasets and train
        train_dataset = self.trainer.prepare_dataset(train_texts, train_labels)
        val_dataset = self.trainer.prepare_dataset(val_texts, val_labels)
        
        # Train using ModelTrainer
        return self.trainer.train(train_dataset, val_dataset)


    def predict(self, text):
        processed_text = self.data_processor.preprocess_text(text)
        inputs = self.tokenizer(
            processed_text,
            return_tensors='tf',
            padding=True,
            truncation=True,
            max_length=ModelConfig.MAX_LENGTH
        )
        
        predictions = self.model(inputs)

        # Process sentiment probabilities
        sentiment_probs = tf.nn.softmax(predictions['sentiment'], axis=-1).numpy()[0]

        # Process other outputs
        sarcasm_prob = tf.nn.sigmoid(predictions['sarcasm']).numpy()[0][0]
        negation_prob = tf.nn.sigmoid(predictions['negation']).numpy()[0][0]
        polarity_score = predictions['polarity'].numpy()[0][0]
        
        # Calculate multipolarity based on sentiment distribution and polarity score
        is_multipolar = (np.max(sentiment_probs) < 0.6) or (polarity_score > 0.5)
    
        return {
            'sentiment': {
                'negative': float(sentiment_probs[0]),
                'neutral': float(sentiment_probs[1]),
                'positive': float(sentiment_probs[2])
            },
            'sarcasm': {
                'probability': float(sarcasm_prob),
                'detected': sarcasm_prob > 0.5 or '_SARC' in processed_text
            },
            'negation': {
                'probability': float(negation_prob),
                'detected': negation_prob > 0.5 or '_NEG' in processed_text
            },
            'multipolarity': {
                'score': float(polarity_score),
                'is_multipolar': bool(is_multipolar)
            },
            'processed_text': processed_text
        }
    


    @staticmethod
    def demonstrate_marking():
        analyzer = SentimentAnalyzer()
        analyzer.initialize_model()
        
        test_cases = [
            # Negation examples
            "I do not like this restaurant",
            "The food wasn't very good",
            
            # Sarcasm examples
            "Yeah right, this is the best service ever...",
            "Thanks a lot for ruining my evening!!!",
            
            # Multipolar examples
            "The food was amazing but the service was terrible",
            "Great atmosphere, decent food, horrible prices",
            
            # Combined examples
            "This couldn't possibly be any better... 🙄",
            "The food was good but I'm not coming back"
        ]
        
        print("Sentiment Analysis Demonstration:\n")
        for text in test_cases:
            results = analyzer.predict(text)
            print(f"\nOriginal:  {text}")
            print(f"Processed: {results['processed_text']}")
            print(f"Analysis Results:")
            print(f"- Sentiment: {max(results['sentiment'].items(), key=lambda x: x[1])[0]}")
            print(f"- Sarcasm Detected: {results['sarcasm']['detected']}")
            print(f"- Negation Detected: {results['negation']['detected']}")
            print(f"- Multipolarity: {results['multipolarity']['is_multipolar']}")
            print("-" * 80)


def main():
    analyzer = SentimentAnalyzer()
    analyzer.initialize_model()
    
    # Train model
    history = analyzer.train()
    AnalysisUtils.plot_training_history(history)
    
    # Test predictions
    test_reviews = [
        "Great service and amazing food!",
        "Terrible experience, would not recommend.",
        "The food was okay, but the service could be better."
    ]
    
    for review in test_reviews:
        results = analyzer.predict(review)
        print(f"\nReview: {review}\nAnalysis: {results}")


if __name__ == "__main__":
   main()