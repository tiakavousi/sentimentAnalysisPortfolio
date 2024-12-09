import tensorflow as tf
import numpy as np
import os
import json
from transformers import DistilBertTokenizer
from data.data_processing import DataProcessor
from config.model_config import ModelConfig
from models.sentiment_model import EnhancedDistilBertForSentiment, ModelTrainer


class SentimentAnalyzer:
    """Main class for sentiment analysis using DistilBERT with enhanced features"""

    def __init__(self):
        """Initialize core components without loading data or model"""
        self.data_processor = DataProcessor()
        self.model = None
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


    def _get_project_root(self):
        """Get the project root directory from any location"""
        if os.path.basename(os.getcwd()) == 'notebooks':
            # If we're in the notebooks directory, go up one level
            return os.path.dirname(os.getcwd())
        return os.getcwd()

    def load_saved_model(self, epoch, model_dir=None):
        """
        Load a previously saved model with weights and configuration.

        Args:
            epoch (int): The training epoch to load
            model_dir (str): Directory containing saved model files

        Returns:
            tuple: (loaded_model, training_history)
        """
        try:
            # Handle model directory path
            if model_dir is None:
                model_dir = os.path.join(os.getcwd(), 'saved_models')

            print(f"Loading model from directory: {model_dir}")

            # 1. Load model info and configuration
            model_info_path = os.path.join(
                model_dir, f"model_info_epoch{epoch}.json")
            if not os.path.exists(model_info_path):
                raise FileNotFoundError(
                    f"Model info file not found at: {model_info_path}")

            with open(model_info_path, 'r') as f:
                self.model_info = json.load(f)
                print("Loaded model configuration and training history")

            # 2. Initialize tokenizer
            try:
                self.tokenizer = DistilBertTokenizer.from_pretrained(
                    'distilbert-base-uncased')
                print("Initialized tokenizer")
            except Exception as e:
                raise RuntimeError(f"Failed to load tokenizer: {str(e)}")

            # 3. Initialize and build model
            try:
                self.model = EnhancedDistilBertForSentiment()

                # Build model with dummy input
                dummy_input = {
                    'input_ids': tf.zeros((1, ModelConfig.MAX_LENGTH), dtype=tf.int32),
                    'attention_mask': tf.zeros((1, ModelConfig.MAX_LENGTH), dtype=tf.int32)
                }
                _ = self.model(dummy_input)
                print("Initialized model architecture")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize model: {str(e)}")

            # 4. Load weights
            weights_path = os.path.join(
                model_dir, f"enhanced_distilbert_epoch{epoch}_weights")
            if not os.path.exists(weights_path + '.index'):
                raise FileNotFoundError(
                    f"Model weights not found at: {weights_path}")

            try:
                self.model.load_weights(weights_path)
                print("Loaded model weights successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to load model weights: {str(e)}")

            # 5. Compile model
            try:
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=ModelConfig.LEARNING_RATE),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                print("Model compiled successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to compile model: {str(e)}")

            print("\nModel loaded successfully!")
            print(f"Original training epochs: {self.model_info['epoch']}")
            print(f"Final validation accuracy: {
                  self.model_info['history']['val_accuracy'][-1]:.4f}")

            return self.model, self.model_info['history']

        except Exception as e:
            print(f"\nError loading model: {str(e)}")
            print("Load location attempted: ", model_dir)
            raise
    
    def save_enhanced_model(self, epoch, history, model_dir=None):
        """
        Save the model weights, configuration, and training history.
        Only called manually after training is complete.

        Args:
            epoch (int): Final training epoch
            history: Training history object
            model_dir (str): Directory to save model files. If None, uses default location
        """
        if self.model is None:
            raise ValueError("Model must be initialized before saving")

        if model_dir is None:
            model_dir = os.path.join(self._get_project_root(), 'saved_models')

        try:
            os.makedirs(model_dir, exist_ok=True)

            # 1. Save model weights
            weights_path = os.path.join(
                model_dir, f"enhanced_distilbert_epoch{epoch}_weights")
            self.model.save_weights(weights_path)
            print("Model weights saved successfully")

            # 2. Save the DistilBERT configuration
            try:
                distilbert_config = self.model.distilbert.config
                distilbert_config_path = os.path.join(
                    model_dir, f"distilbert_config_epoch{epoch}.json")
                distilbert_config.to_json_file(distilbert_config_path)
                print("DistilBERT configuration saved successfully")
            except Exception as e:
                print(f"Warning: Failed to save DistilBERT config: {str(e)}")

            # 3. Save model parameters and history
            model_params = {
                'lstm_units': ModelConfig.LSTM_UNITS,
                'feature_dim': ModelConfig.FEATURE_DIM,
                'fusion_layers': ModelConfig.FUSION_LAYERS,
                'dropout_rates': ModelConfig.DROPOUT_RATES,
                'num_classes': ModelConfig.NUM_CLASSES
            }

            # Convert history to regular Python types
            try:
                converted_history = {
                    'loss': [float(v) for v in history.history['loss']],
                    'val_loss': [float(v) for v in history.history['val_loss']],
                    'accuracy': [float(v) for v in history.history['accuracy']],
                    'val_accuracy': [float(v) for v in history.history['val_accuracy']]
                }
            except Exception as e:
                print(f"Warning: Error converting history: {str(e)}")
                converted_history = {}

            # 4. Create and save info file
            save_info = {
                'epoch': int(epoch),
                'history': converted_history,
                'model_parameters': model_params,
                'weights_path': weights_path
            }

            info_path = os.path.join(
                model_dir, f"model_info_epoch{epoch}.json")
            with open(info_path, 'w') as f:
                json.dump(save_info, f)
            print("Model info saved successfully")

            print(f"\nAll model data saved successfully in directory: {
                  model_dir}")

        except Exception as e:
            print(f"Error during model saving: {str(e)}")
            print("Save location attempted: ", model_dir)
            raise


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

        print("\nSaving the trained model...")
        analyzer.save_enhanced_model(epoch=ModelConfig.EPOCHS, history=history)

        print("\nModel training complete. Use \"03_LoadAndTestTrainedModel.ipynb\" notebook for final evaluation.")
        
        return history
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
        
    finally:
        if analyzer:
            analyzer.cleanup()


if __name__ == "__main__":
   main()