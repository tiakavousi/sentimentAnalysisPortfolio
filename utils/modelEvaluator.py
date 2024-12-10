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

    def visualize_performance(self, metrics, history=None):
        """Visualize model performance metrics"""
        if history:
            self._plot_training_history(history)
        
        target_names = ['Negative', 'Neutral', 'Positive']
        
        # First set: Main evaluation metrics
        plt.figure(figsize=(15, 5))
        
        # 1. Confusion Matrix
        plt.subplot(1, 3, 1)
        sns.heatmap(
            metrics['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 2. Prediction Confidence
        plt.subplot(1, 3, 2)
        sns.histplot(metrics['confidence_scores'], bins=20)
        plt.axvline(
            metrics['avg_confidence'],
            color='r',
            linestyle='--',
            label=f"Mean: {metrics['avg_confidence']:.3f}"
        )
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.legend()
        
        # 3. Class-wise Accuracies
        plt.subplot(1, 3, 3)
        class_accuracies = [metrics['class_accuracies'][cls] for cls in target_names]
        plt.bar(target_names, class_accuracies)
        plt.title('Class-wise Accuracy')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
        
        # Second set: Detailed performance metrics
        plt.figure(figsize=(15, 5))
        
        # 1. Class-wise Performance Metrics
        plt.subplot(1, 2, 1)
        report = metrics['classification_report']
        perf_metrics = ['precision', 'recall', 'f1-score']
        x = np.arange(len(perf_metrics))
        width = 0.25
        
        for i, cls in enumerate(target_names):
            # Use original case for accessing report
            scores = [report[cls][m] for m in perf_metrics]
            plt.bar(x + i*width, scores, width, label=cls)
        
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title('Class-wise Performance Metrics')
        plt.xticks(x + width, perf_metrics)
        plt.legend()
        
        # 2. Prediction Distribution
        plt.subplot(1, 2, 2)
        dist = metrics['prediction_distribution']
        plt.bar(target_names, [dist[cls]['percentage'] for cls in target_names])
        plt.title('Prediction Distribution')
        plt.ylabel('Percentage')
        
        plt.tight_layout()
        plt.show()
        
        # Print metrics report
        print("\n=== Evaluation Metrics ===")
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Average Confidence: {metrics['avg_confidence']:.4f}")
        
        print("\nClass-wise Performance:")
        for cls in target_names:
            print(f"\n{cls}:")
            print(f"Accuracy: {metrics['class_accuracies'][cls]:.4f}")
            print(f"Precision: {report[cls]['precision']:.4f}")
            print(f"Recall: {report[cls]['recall']:.4f}")
            print(f"F1-score: {report[cls]['f1-score']:.4f}")
        
        print("\nPrediction Distribution:")
        for cls in target_names:
            dist = metrics['prediction_distribution'][cls]
            print(f"{cls}: {dist['count']} ({dist['percentage']:.2f}%)")

    def _plot_training_history(self, history):
        """Plot training history metrics"""
        history_dict = history.history if hasattr(history, 'history') else history
        
        plt.figure(figsize=(15, 5))
        
        # 1. Loss curves
        plt.subplot(1, 3, 1)
        plt.plot(history_dict['loss'], label='Training Loss', marker='o')
        plt.plot(history_dict['val_loss'], label='Validation Loss', marker='o')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Accuracy curves
        plt.subplot(1, 3, 2)
        plt.plot(history_dict['accuracy'], label='Training Accuracy', marker='o')
        plt.plot(history_dict['val_accuracy'], label='Validation Accuracy', marker='o')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Learning rate
        if 'lr' in history_dict:
            plt.subplot(1, 3, 3)
            plt.plot(history_dict['lr'], marker='o')
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print training summary
        print("\n=== Training Summary ===")
        print(f"Epochs trained: {len(history_dict['loss'])}")
        print(f"Best validation loss: {min(history_dict['val_loss']):.4f}")
        print(f"Best validation accuracy: {max(history_dict['val_accuracy']):.4f}")