import tensorflow as tf
from transformers import DistilBertTokenizer
import numpy as np

class SentimentAnalyzer:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model = None
        self.tokenizer = None
        self.trainer = None

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
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self):
        # Load and process data
        df = self.data_processor.load_data()
        train_texts, val_texts, train_labels, val_labels = self.data_processor.split_data(df)
        
        # Prepare datasets
        train_dataset = self.trainer.prepare_dataset(train_texts, train_labels)
        val_dataset = self.trainer.prepare_dataset(val_texts, val_labels)
        
        # Train model
        history = self.trainer.train(train_dataset, val_dataset)
        return history

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
        probs = tf.nn.softmax(predictions, axis=-1).numpy()[0]
        
        return {
            'negative': float(probs[0]),
            'neutral': float(probs[1]),
            'positive': float(probs[2]),
            'has_sarcasm': '_SARC' in processed_text,
            'has_negation': '_NEG' in processed_text,
            'is_multipolar': np.max(probs) < 0.6
        }

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