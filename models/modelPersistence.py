import datetime
import tensorflow as tf
import os
import json
import sys
import transformers  # Import transformers directly
from transformers import DistilBertTokenizer
from models.sentiment_model import EnhancedDistilBertForSentiment
from config.model_config import Config

# If you need to debug the location, use this instead:
print("Transformers package location:", transformers.__file__)

class ModelPersistence():
    @staticmethod
    def save_model(epoch, history, model, tokenizer, model_dir, version):
        """Save the trained model and associated artifacts"""
        save_dir = os.path.join(model_dir, f"model_v{version}_epoch{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, "full_model")
        tf.keras.models.save_model(model, model_path)
        
        # Save tokenizer
        tokenizer_path = os.path.join(save_dir, "tokenizer")
        tokenizer.save_pretrained(tokenizer_path)
        
        # Save config
        config = {
            'model_params': {
                'bert_model': Config.BERT_MODEL,
                'lstm_units': Config.LSTM_UNITS,
                'feature_dim': Config.FEATURE_DIM,
                'fusion_layers': Config.FUSION_LAYERS,
                'dropout_rates': Config.DROPOUT_RATES,
                'num_classes': Config.NUM_CLASSES
            },
            'training_params': {
                'max_length': Config.MAX_LENGTH,
                'batch_size': Config.BATCH_SIZE,
                'learning_rate': Config.LEARNING_RATE,
                'num_epochs': Config.EPOCHS
            },
            'performance': {
                'final_train_accuracy': history.history['accuracy'][-1],
                'final_val_accuracy': history.history['val_accuracy'][-1],
                'best_val_accuracy': max(history.history['val_accuracy'])
            }
        }
        
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved to: {save_dir}")
        return save_dir


   
    @staticmethod
    def load_model(model_path, return_config=True):
        """Load a trained model and associated artifacts
        
        Args:
            model_path (str): Path to the saved model directory
            return_config (bool): Whether to return the saved configuration
            
        Returns:
            tuple: (loaded_model, loaded_tokenizer, config) if return_config=True
                (loaded_model, loaded_tokenizer) if return_config=False
                
        Raises:
            FileNotFoundError: If model directory or required files are missing
            ValueError: If loaded model/config is incompatible
        """
        try:
            # Validate directory exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model directory not found: {model_path}")
                
            # Load model
            model_full_path = os.path.join(model_path, "full_model")
            if not os.path.exists(model_full_path):
                raise FileNotFoundError(f"Model not found in directory: {model_full_path}")
                
            print("Loading model architecture and weights...")
            loaded_model = tf.keras.models.load_model(model_full_path)
            
            # Load tokenizer
            tokenizer_path = os.path.join(model_path, "tokenizer")
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"Tokenizer not found in directory: {tokenizer_path}")
                
            print("Loading tokenizer...")
            loaded_tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
            
            # Load config if requested
            if return_config:
                config_path = os.path.join(model_path, "config.json")
                if not os.path.exists(config_path):
                    raise FileNotFoundError(f"Config file not found: {config_path}")
                    
                print("Loading model configuration...")
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                # Validate configuration
                required_keys = ['model_params', 'training_params', 'performance']
                if not all(key in config for key in required_keys):
                    raise ValueError("Invalid config file: missing required sections")
                    
                # Verify model architecture matches config
                model_config = config['model_params']
                if loaded_model.get_layer('bi_lstm').units != model_config['lstm_units']:
                    raise ValueError("Loaded model architecture doesn't match saved configuration")
                    
                return loaded_model, loaded_tokenizer, config
                
            return loaded_model, loaded_tokenizer
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise