import tensorflow as tf
import os
import json
import sys
import transformers  # Import transformers directly
from transformers import DistilBertTokenizer
from models.sentiment_model import EnhancedDistilBertForSentiment
from config.model_config import ModelConfig

# If you need to debug the location, use this instead:
print("Transformers package location:", transformers.__file__)

class ModelPersistence():
    @staticmethod
    def save_model(epoch, history, model, model_dir=None):
        """
        Save the model weights, configuration, and training history.
        Only called manually after training is complete.

        Args:
            epoch (int): Final training epoch
            history: Training history object
            model: The model to save
            model_dir (str): Directory to save model files. If None, uses default location
        """
        if model_dir is None:
            model_dir='/Users/tayebekavousi/Desktop/sentimentAnalysisPortfolio/saved_models'
            
        try:
            os.makedirs(model_dir, exist_ok=True)

            # 1. Save the entire model including optimizer state
            model_path = os.path.join(model_dir, f"enhanced_distilbert_epoch{epoch}")
            model.save(model_path, save_format='tf')
            print("Full model saved successfully")

            # 2. Save the DistilBERT configuration
            try:
                distilbert_config = model.distilbert.config
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
                'model_path': model_path
            }

            info_path = os.path.join(model_dir, f"model_info_epoch{epoch}.json")
            with open(info_path, 'w') as f:
                json.dump(save_info, f)
            print("Model info saved successfully")

            print(f"\nAll model data saved successfully in directory: {model_dir}")

        except Exception as e:
            print(f"Error during model saving: {str(e)}")
            print("Save location attempted: ", model_dir)
            raise

    @staticmethod
    def load_model(epoch, model_dir=None):
        if model_dir is None:
            model_dir='/Users/tayebekavousi/Desktop/sentimentAnalysisPortfolio/saved_models'

        try:
            # Load model info
            info_path = os.path.join(model_dir, f"model_info_epoch{epoch}.json")
            with open(info_path, 'r') as f:
                model_info = json.load(f)

            # Initialize model (no parameters needed)
            model = EnhancedDistilBertForSentiment()

            # Load weights
            weights_path = model_info['weights_path']
            model.load_weights(weights_path)
            print("Model weights loaded successfully")

            return model, model_info['history']

        except Exception as e:
            print(f"Error during model loading: {str(e)}")
            print("Load location attempted: ", model_dir)
            raise


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
                model_dir='/Users/tayebekavousi/Desktop/sentimentAnalysisPortfolio/saved_models'

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
        




    