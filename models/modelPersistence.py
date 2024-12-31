from config.model_config import Config
from models.sentiment_model import EnhancedDistilBertForSentiment
import tensorflow as tf
import os
import json
from transformers import DistilBertTokenizer


class ModelPersistence():
    @staticmethod
    def save_model_state(epoch, history, model, tokenizer, model_dir, version):
        """
        Save model state including architecture, weights, and custom objects.
        
        Args:
            epoch (int): Current epoch number
            history: Training history object
            model: The trained model
            tokenizer: The tokenizer used
            model_dir (str): Directory to save the model
            version (str): Version string
        """
        
        # Create save directory
        save_dir = os.path.join(model_dir, f"model_v{version}_epoch{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Save model with full architecture and weights
        model_path = os.path.join(save_dir, "full_model")
        
        # Save model configuration (architecture)
        model_config = model.get_config()
        config_path = os.path.join(save_dir, "model_architecture.json")
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # Save full model including weights
        try:
            # First try saving with model.save()
            model.save(model_path, save_format='tf', include_optimizer=True)
        except Exception as e:
            print(f"Warning: Standard save failed ({str(e)}), trying alternative method...")
            # Fallback: Save weights separately if full save fails
            weights_path = os.path.join(save_dir, "model_weights")
            model.save_weights(weights_path, save_format='tf')
        
        # 2. Save tokenizer
        tokenizer_path = os.path.join(save_dir, "tokenizer")
        tokenizer.save_pretrained(tokenizer_path)
        
        # 3. Save training history - convert tensors to floats
        history_dict = {}
        for key, value in history.history.items():
            history_dict[key] = [float(val.numpy()) if tf.is_tensor(val) 
                            else float(val) 
                            for val in value]
        
        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)


        print(f"Model state saved successfully to: {save_dir}")
        return save_dir

    
    @staticmethod
    def load_model_state(model_path):
        """
        Load a saved model state including architecture, weights, tokenizer, and history.
        
        Args:
            model_path (str): Path to the saved model directory
            
        Returns:
            dict: Dictionary containing loaded model components
                'model': The loaded model with weights
                'tokenizer': The loaded tokenizer
                'history': Training history
                'config': Model configuration
        """
        print("\nLoading model state...")
        
        try:
            # 1. Load model configuration
            print("Loading model configuration...")
            with open(os.path.join(model_path, "model_architecture.json"), 'r') as f:
                model_config = json.load(f)
                
            # 2. Initialize model from config
            print("Initializing model architecture...")
            model = EnhancedDistilBertForSentiment()
            # Create dummy input to build the model
            dummy_input = {
                'input_ids': tf.zeros((1, Config.MAX_LENGTH), dtype=tf.int32),
                'attention_mask': tf.zeros((1, Config.MAX_LENGTH), dtype=tf.int32)
            }
            _ = model(dummy_input)  # Build the model
            
            # 3. Load model weights
            print("Loading model weights...")
            model.load_weights(os.path.join(model_path, "full_model", "variables", "variables"))
            
            # 4. Load tokenizer
            print("Loading tokenizer...")
            tokenizer = DistilBertTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))
            
            # 5. Load training history
            print("Loading training history...")
            with open(os.path.join(model_path, "training_history.json"), 'r') as f:
                history_dict = json.load(f)
            
            # Create a history object
            class History:
                def __init__(self, history_dict):
                    self.history = history_dict
            history = History(history_dict)
            
            print("Model state loaded successfully!")
            
            return {
                'model': model,
                'tokenizer': tokenizer,
                'history': history,
                'config': model_config
            }
            
        except Exception as e:
            raise Exception(f"Error loading model state: {str(e)}")