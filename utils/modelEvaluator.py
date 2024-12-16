import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """Handles model evaluation and performance visualization"""
    
    def __init__(self, model, trainer):
        self.model = model
        self.trainer = trainer
        self.label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    def evaluate_model(self, texts, labels, batch_size=32):
        """Evaluate model on test data"""
        print(f"Starting evaluation on {len(texts)} examples...")
        
        # Prepare features
        print("Preparing input features...")
        features = self.trainer.prepare_dataset(texts)
        print("Features prepared successfully")
        
        # Generate predictions in batches
        print("\nGenerating predictions...")
        all_predictions = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_features = {
                'input_ids': features['input_ids'][i:i+batch_size],
                'attention_mask': features['attention_mask'][i:i+batch_size]
            }
            predictions = self.model(batch_features, training=False)
            all_predictions.append(predictions)
            
            current_batch = (i + batch_size) // batch_size
            print(f"Progress: {current_batch}/{total_batches} batches ({(current_batch/total_batches*100):.1f}%)")
        
        print("\nCalculating metrics...")
        predictions = tf.concat(all_predictions, axis=0).numpy()
        predicted_labels = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)
        true_labels = labels['sentiment']

        print("Computing class-wise metrics...")
        # Calculate class-wise accuracy
        class_accuracies = {}
        target_names = ['Negative', 'Neutral', 'Positive']
        for i, class_name in enumerate(target_names):
            class_mask = np.array(true_labels) == i
            class_accuracies[class_name] = np.mean(np.array(predicted_labels)[class_mask] == i)
        
        # Calculate prediction distribution
        unique_labels, counts = np.unique(predicted_labels, return_counts=True)
        prediction_distribution = {
            target_names[label]: {
                'count': count,
                'percentage': (count/len(predicted_labels)*100)
            }
            for label, count in zip(unique_labels, counts)
        }
        
        # Compile all metrics
        metrics = {
            'predictions': predictions,
            'predicted_labels': predicted_labels,
            'true_labels': true_labels,
            'confidence_scores': confidence_scores,
            'accuracy': np.mean(true_labels == predicted_labels),
            'class_accuracies': class_accuracies,
            'confusion_matrix': confusion_matrix(true_labels, predicted_labels),
            'classification_report': classification_report(
                true_labels, predicted_labels,
                target_names=target_names,
                output_dict=True
            ),
            'prediction_distribution': prediction_distribution,
            'avg_confidence': np.mean(confidence_scores)
        }
        
        return metrics

    def visualize_performance(self, metrics, history):
        """
        Visualize model performance with proper label handling.
        
        Args:
            metrics (dict): Dictionary containing evaluation metrics
            history (dict): Model training history
        """
        plt.figure(figsize=(15, 5))
        
        # Training history plot
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Prediction distribution plot
        plt.subplot(1, 2, 2)
        dist = metrics['prediction_distribution']
        
        # Get actual keys from the distribution dictionary
        available_classes = list(dist.keys())
        
        # Create the bar plot using available classes
        percentages = [dist[cls]['percentage'] for cls in available_classes]
        plt.bar(available_classes, percentages)
        plt.title('Prediction Distribution')
        plt.ylabel('Percentage')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Print numerical metrics
        print("\nModel Performance Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print("\nClass-wise Metrics:")
        for cls in available_classes:
            print(f"\n{cls}:")
            print(f"Precision: {metrics['class_metrics'][cls]['precision']:.4f}")
            print(f"Recall: {metrics['class_metrics'][cls]['recall']:.4f}")
            print(f"F1-Score: {metrics['class_metrics'][cls]['f1_score']:.4f}")