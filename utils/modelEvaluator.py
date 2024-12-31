import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from config.model_config import Config
from utils.testcases import TestCases


class ModelEvaluator:
    """
    Evaluates machine learning model performance through metrics, visualizations and testing.
    
    Attributes:
        history: Training history containing metrics per epoch
        model: Trained model instance
        trainer: Model trainer instance handling model training with custom learning rate scheduling and callbacks.
        label_map: Dictionary mapping label indices to human-readable names
        edge_cases: Collection of test cases for edge case evaluation
    """
    def __init__(self, history, model, trainer):
        """Initializes evaluator with model artifacts and configuration."""
        self.history = history
        self.model = model
        self.trainer = trainer
        self.label_map = Config.LABEL_MAP
        self.edge_cases = TestCases.edge_cases
    
    def plot_training_history(self):
        """
        Plot training & validation loss and accuracy over epochs
        
        Args:
            history: History object returned by model.fit()
        """
        # Extract metrics from history
        metrics = ['loss', 'accuracy']
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Model Training History', fontsize=14)
        
        for idx, metric in enumerate(metrics):
            # Plot training and validation metrics
            train_values = self.history.history[metric]
            val_values = self.history.history[f'val_{metric}']
            epochs = range(1, len(train_values) + 1)
            
            axes[idx].plot(epochs, train_values, 'bo-', label=f'Training {metric}')
            axes[idx].plot(epochs, val_values, 'ro-', label=f'Validation {metric}')
            
            # Customize plot
            axes[idx].set_title(f'Model {metric.capitalize()}')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric.capitalize())
            axes[idx].legend()
            axes[idx].grid(True)
            
            # Set y-axis limits for accuracy between 0 and 1
            if metric == 'accuracy':
                axes[idx].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.show()
        
        # Print final metrics
        print("\nFinal Training Metrics:")
        for metric in metrics:
            print(f"Training {metric}: {self.history.history[metric][-1]:.4f}")
            print(f"Validation {metric}: {self.history.history[f'val_{metric}'][-1]:.4f}")
    
    def evaluate_test_set(self, test_texts, test_labels, batch_size=32):
        """
        Evaluate model performance on test set.
        
        Args:
            test_texts: Array of test text samples
            test_labels: Dictionary containing 'sentiment' labels
            batch_size: Batch size for prediction
        """
        # Prepare features
        features = self.trainer.prepare_dataset(test_texts)
        
        # Generate predictions
        predictions = []
        for i in range(0, len(test_texts), batch_size):
            batch_features = {
                'input_ids': features['input_ids'][i:i+batch_size],
                'attention_mask': features['attention_mask'][i:i+batch_size]
            }
            batch_predictions = self.model(batch_features, training=False)
            predictions.append(batch_predictions)
        
        # Combine predictions
        predictions = tf.concat(predictions, axis=0).numpy()
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = test_labels['sentiment']
        
        # Calculate metrics
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        class_report = classification_report(
            true_labels, predicted_labels,
            target_names=list(self.label_map.values()),
            output_dict=True
        )
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=list(self.label_map.values()),
            yticklabels=list(self.label_map.values())
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # Print metrics
        print("\nTest Set Metrics:")
        print(f"Overall Accuracy: {class_report['accuracy']:.4f}")
        print(f"Macro Avg F1-Score: {class_report['macro avg']['f1-score']:.4f}")
        print("\nClass-wise Metrics:")
        for label in self.label_map.values():
            print(f"\n{label}:")
            print(f"Precision: {class_report[label]['precision']:.4f}")
            print(f"Recall: {class_report[label]['recall']:.4f}")
            print(f"F1-Score: {class_report[label]['f1-score']:.4f}")
        
        return {
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }
    
    def test_edge_cases(self):
        """
        Test model performance on edge cases.
        
        Args:
            edge_cases: List of dictionaries with format:
                      [{'text': 'sample text', 'expected': 'Positive'}, ...]
        """
        texts = [case['text'] for case in self.edge_cases]
        expected_labels = [list(self.label_map.values()).index(case['expected']) 
                         for case in self.edge_cases]
        
        # Prepare features
        features = self.trainer.prepare_dataset(np.array(texts))
        
        # Get predictions
        predictions = self.model(features, training=False)
        predicted_labels = np.argmax(predictions.numpy(), axis=1)
        
        # Calculate and display results
        print("\nEdge Case Analysis:")
        for i, case in enumerate(self.edge_cases):
            pred_label = self.label_map[predicted_labels[i]]
            print(f"\nTest Case {i+1}:")
            print(f"Text: {case['text']}")
            print(f"Expected: {case['expected']}")
            print(f"Predicted: {pred_label}")
            print(f"Correct: {'✓' if pred_label == case['expected'] else '✗'}")
        
        # Calculate accuracy
        accuracy = np.mean(predicted_labels == expected_labels)
        print(f"\nOverall Edge Case Accuracy: {accuracy:.4f}")
        
        return {
            'predictions': predictions.numpy(),
            'accuracy': accuracy
        }


    
    
    def test_user_input(self, text):
        """
        Test model with a single user input.
        
        Args:
            text: String of text to analyze
        """
        # Prepare features
        features = self.trainer.prepare_dataset(np.array([text]))
        
        # Get prediction
        prediction = self.model(features, training=False)
        predicted_probs = tf.nn.softmax(prediction, axis=-1).numpy()[0]
        predicted_label = self.label_map[np.argmax(predicted_probs)]
        
        # Display results
        print("\nSentiment Analysis Results:")
        print(f"Input Text: {text}")
        print(f"\nPredicted Sentiment: {predicted_label}")
        print("\nClass Probabilities:")
        for label, prob in zip(self.label_map.values(), predicted_probs):
            print(f"{label}: {prob:.4f}")
        
        return {
            'sentiment': predicted_label,
            'probabilities': dict(zip(self.label_map.values(), predicted_probs))
        }