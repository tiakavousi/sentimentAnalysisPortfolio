import json
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from config.model_config import Config


class ModelEvaluator:    
    def __init__(self, model, trainer):
        self.model = model
        self.trainer = trainer
        self.label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    def evaluate_model(self, texts, labels, batch_size = Config.BATCH_SIZE):
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
        
        # Calculate metrics
        print("\nCalculating metrics...")
        predictions = tf.concat(all_predictions, axis=0).numpy()
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = labels['sentiment']
        
        # Calculate comprehensive metrics
        classification_rep = classification_report(
            true_labels, predicted_labels,
            target_names=list(self.label_map.values()),
            output_dict=True
        )
        
        # Print detailed metrics
        print("\n=== Overall Metrics ===")
        print(f"Accuracy: {classification_rep['accuracy']:.4f}")
        print(f"Macro Avg F1-Score: {classification_rep['macro avg']['f1-score']:.4f}")
        print(f"Weighted Avg F1-Score: {classification_rep['weighted avg']['f1-score']:.4f}")
        
        print("\n=== Class-wise Metrics ===")
        for label in self.label_map.values():
            print(f"\n{label}:")
            print(f"Precision: {classification_rep[label]['precision']:.4f}")
            print(f"Recall: {classification_rep[label]['recall']:.4f}")
            print(f"F1-Score: {classification_rep[label]['f1-score']:.4f}")
            print(f"Support: {classification_rep[label]['support']}")
        
        # Return comprehensive metrics
        return {
            'predictions': predictions,
            'predicted_labels': predicted_labels,
            'true_labels': true_labels,
            'confusion_matrix': confusion_matrix(true_labels, predicted_labels),
            'classification_report': classification_rep,
            'overall_accuracy': classification_rep['accuracy'],
            'class_metrics': {label: classification_rep[label] for label in self.label_map.values()}
        }


    def visualize_performance(self, history)
        with open(os.path.join(model_path, "config.json"), 'r') as f:
        config = json.load(f)
        metrics, predictions = evaluate_model(analyzer)  # Using your existing evaluate_model function
        plt.figure(figsize=(10, 8))
        plot_confusion_matrix(metrics)
        plot_performance_metrics(metrics)
        print_detailed_metrics(metrics)

    def plot_training_history(history):
        """Plot training metrics history"""
        plt.figure(figsize=(15, 5))
        
        # Plot loss
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 3, 2)
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Accuracy Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot learning rate if available
        if 'lr' in history.history:
            plt.subplot(1, 3, 3)
            plt.plot(history.history['lr'])
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
        
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(metrics):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        sns.heatmap(
            metrics['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[label_map[i] for i in range(3)],
            yticklabels=[label_map[i] for i in range(3)]
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def plot_performance_metrics(metrics):
        """Plot detailed performance metrics"""
        plt.figure(figsize=(15, 10))
        
        # Performance by category
        plt.subplot(2, 1, 1)
        categories = ['Negative', 'Neutral', 'Positive']
        report = metrics['classification_report']
        
        metrics_data = {
            'Precision': [report[cat]['precision'] for cat in categories],
            'Recall': [report[cat]['recall'] for cat in categories],
            'F1-Score': [report[cat]['f1-score'] for cat in categories]
        }
        
        x = np.arange(len(categories))
        width = 0.25
        
        plt.bar(x - width, metrics_data['Precision'], width, label='Precision')
        plt.bar(x, metrics_data['Recall'], width, label='Recall')
        plt.bar(x + width, metrics_data['F1-Score'], width, label='F1-Score')
        
        plt.xlabel('Sentiment Category')
        plt.ylabel('Score')
        plt.title('Performance Metrics by Category')
        plt.xticks(x, categories)
        plt.legend()
        
        # Sarcasm impact
        plt.subplot(2, 1, 2)
        sarcasm_metrics = [
            metrics['non_sarcastic_accuracy'],
            metrics['sarcastic_accuracy'],
            metrics['accuracy']
        ]
        plt.bar(['Non-Sarcastic', 'Sarcastic', 'Overall'],
                sarcasm_metrics,
                color=['skyblue', 'lightcoral', 'lightgreen'])
        plt.title('Accuracy by Sarcasm Type')
        plt.ylabel('Accuracy')
        
        plt.tight_layout()
        plt.show()

    def print_detailed_metrics(metrics):
        """Print detailed evaluation metrics"""
        print("\nDetailed Performance Metrics:")
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Sarcastic Text Accuracy: {metrics['sarcastic_accuracy']:.4f}")
        print(f"Non-sarcastic Text Accuracy: {metrics['non_sarcastic_accuracy']:.4f}")
        
        categories = ['Negative', 'Neutral', 'Positive']
        report = metrics['classification_report']
        
        for category in categories:
            print(f"\n{category} Metrics:")
            print(f"Precision: {report[category]['precision']:.4f}")
            print(f"Recall: {report[category]['recall']:.4f}")
            print(f"F1-score: {report[category]['f1-score']:.4f}")


    def test_sentiment_edge_cases(self, analyzer):
        test_cases = {
            "Positive with Negation": [
                ("I can't deny that this place is amazing. Not a single thing wrong with the food or service!", "Double Negative -> Positive"),
                ("Never had a bad experience here. The food isn't anything less than perfect.", "Negated Negative"),
                ("Not once have I been disappointed by their service. The staff isn't unfriendly at all.", "Complex Negation")
            ],
            
            "Positive with Sarcasm": [
                ("Yeah right, like this restaurant could possibly get ANY better! *chef's kiss* Absolutely perfect!", "Exaggerated Praise"),
                ("Oh sure, just RUIN my diet with your impossibly delicious desserts! How dare you be this good!", "Mock Complaint"),
                ("Just what I needed - another restaurant to be obsessed with! 🙄 Now I'll have to keep coming back!", "Fake Annoyance")
            ],
            
            "Positive with Multipolarity": [
                ("The wait was long but honestly worth every minute. Amazing food and exceptional service!", "Contrast Resolution"),
                ("Small portions and pricey, but the taste makes up for everything. Will definitely return!", "Trade-off Acceptance"),
                ("Noisy atmosphere but incredible food and the best service I've had in years.", "Mixed with Positive Dominance")
            ],
            
            "Negative with Negation": [
                ("The food isn't good at all. Not worth the price and I won't be returning.", "Direct Negative"),
                ("I couldn't find anything special about this place. The service wasn't even close to acceptable.", "Compound Negative"),
                ("Not once did they get our order right. The manager wasn't helpful either.", "Sequential Negative")
            ],
            
            "Negative with Sarcasm": [
                ("Oh fantastic, another overpriced meal with cold food. Just what I was hoping for! 🙄", "Mock Enthusiasm"),
                ("Wow, amazing how they consistently manage to mess up a simple order. Such talent! 😒", "Ironic Praise"),
                ("Five stars for teaching me the true meaning of patience! Best 2-hour wait ever! 🙃", "Exaggerated Compliment")
            ],
            
            "Negative with Multipolarity": [
                ("Great location but terrible food and even worse service. Definitely not returning.", "Location vs Experience"),
                ("Beautiful decor, shame about the rude staff and inedible food.", "Aesthetics vs Function"),
                ("Nice ambiance but overpriced food and disappointing service ruined the experience.", "Environment vs Service")
            ],
            
            "Neutral with Negation": [
                ("The food isn't amazing but isn't terrible either. Just an average experience.", "Balanced Negation"),
                ("Not the best, not the worst. Wouldn't go out of my way to return.", "Double Neutral Negation"),
                ("Can't say it was great, can't say it was bad. Just okay.", "Negated Extremes")
            ],
            
            "Neutral with Sarcasm": [
                ("Well, that was... an experience. I guess that's one way to run a restaurant! 🤔", "Ambiguous Evaluation"),
                ("'Interesting' take on Italian food. Very... unique interpretation! 😏", "Noncommittal Sarcasm"),
                ("Such a... memorable experience. Definitely different from what I expected! 🫤", "Understated Sarcasm")
            ],
            
            "Neutral with Multipolarity": [
                ("Good food but slow service. Bad parking but nice location. Evens out I guess.", "Balanced Trade-offs"),
                ("Some dishes were great, others terrible. Service varied. Hard to form an opinion.", "Mixed Experience"),
                ("Excellent appetizers, mediocre mains, poor desserts. Averages out to okay.", "Quality Variation")
            ],
            
        }
        
        # Collect all texts and their expected sentiments
        all_texts = []
        true_sentiments = []
        
        print("\n=== Testing Edge Cases ===\n")
        for category, cases in test_cases.items():
            print(f"\n{category}:")
            for text, case_type in cases:
                prediction = analyzer.predict(text)
                print(f"\nCase Type: {case_type}")
                print(f"Text: {text}")
                print(f"Prediction: {prediction}")
                
                all_texts.append(text)
                sentiment = category.split()[0]
                true_sentiments.append(list(self.label_map.values()).index(sentiment))
        
        # Convert lists to numpy arrays before evaluation
        all_texts = np.array(all_texts)
        true_sentiments = np.array(true_sentiments)
        
        # Evaluate edge cases
        print("\n=== Edge Cases Evaluation ===")
        edge_case_metrics = self.evaluate_model(
            all_texts, 
            {'sentiment': true_sentiments}
        )
        
        # Plot confusion matrix for edge cases
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            edge_case_metrics['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=list(self.label_map.values()),
            yticklabels=list(self.label_map.values())
        )
        plt.title('Edge Cases Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return edge_case_metrics


    def predict_sentiment(self, analyzer, text):
        prediction = analyzer.predict(text)
        
        print("\n=== Sentiment Analysis Results ===")
        print(f"\nInput Text: {text}")
        print("\nPrediction:", json.dumps(prediction, indent=2))
        
        return prediction
    
    def visualize_results(training_history, test_metrics, label_map):
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Training History Plot
        ax1 = plt.subplot(2, 3, 1)
        metrics = {
            'Training Accuracy': training_history.get('final_train_accuracy', 0),
            'Validation Accuracy': training_history.get('final_val_accuracy', 0),
            'Best Validation': training_history.get('best_val_accuracy', 0)
        }
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        bars = plt.bar(range(len(metrics)), list(metrics.values()), color=colors)
        plt.xticks(range(len(metrics)), list(metrics.keys()), rotation=45)
        plt.title('Model Training Performance', pad=20)
        plt.ylabel('Accuracy')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        # 2. Test Set Performance by Class
        ax2 = plt.subplot(2, 3, 2)
        categories = list(label_map.values())
        metrics_data = {
            'Precision': [test_metrics['classification_report'][cat]['precision'] for cat in categories],
            'Recall': [test_metrics['classification_report'][cat]['recall'] for cat in categories],
            'F1-Score': [test_metrics['classification_report'][cat]['f1-score'] for cat in categories]
        }
        
        x = np.arange(len(categories))
        width = 0.25
        
        for i, (metric, values) in enumerate(metrics_data.items()):
            bars = plt.bar(x + i*width - width, values, width, label=metric)
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', rotation=0)
        
        plt.title('Test Set Performance by Class', pad=20)
        plt.xlabel('Sentiment Category')
        plt.ylabel('Score')
        plt.xticks(x, categories)
        plt.legend(loc='upper right')
        
        # 3. Confusion Matrix
        ax3 = plt.subplot(2, 3, 3)
        im = plt.imshow(test_metrics['confusion_matrix'], cmap='Blues')
        
        # Add numbers to confusion matrix
        for i in range(len(categories)):
            for j in range(len(categories)):
                text = plt.text(j, i, test_metrics['confusion_matrix'][i, j],
                            ha="center", va="center", color="black")
        
        plt.title('Confusion Matrix on Test Set', pad=20)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(range(len(categories)), categories)
        plt.yticks(range(len(categories)), categories)
        plt.colorbar(im)
        
        # 4. Detailed Metrics Summary
        ax4 = plt.subplot(2, 1, 2)
        ax4.axis('off')
        
        report = test_metrics['classification_report']
        metrics_text = f"""
        Overall Performance Metrics:
        
        Training Metrics:
        • Final Training Accuracy: {training_history.get('final_train_accuracy', 0):.3f}
        • Final Validation Accuracy: {training_history.get('final_val_accuracy', 0):.3f}
        • Best Validation Accuracy: {training_history.get('best_val_accuracy', 0):.3f}
        
        Test Set Metrics:
        • Overall Accuracy: {report['accuracy']:.3f}
        • Macro Avg F1-Score: {report['macro avg']['f1-score']:.3f}
        • Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.3f}
        
        Class-wise Performance (Test Set):
        • Negative - Precision: {report['Negative']['precision']:.3f}, Recall: {report['Negative']['recall']:.3f}
        • Neutral  - Precision: {report['Neutral']['precision']:.3f}, Recall: {report['Neutral']['recall']:.3f}
        • Positive - Precision: {report['Positive']['precision']:.3f}, Recall: {report['Positive']['recall']:.3f}
        """
        plt.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace')
        
        plt.tight_layout()
        return fig

    def performance_visualizer():
        # Load config
        with open(os.path.join(model_path, "config.json"), 'r') as f:
            config = json.load(f)

        performance = config.get('performance', {})
        training_params = config.get('training_params', {})

        # Plot final metrics as bar chart
        plt.figure(figsize=(12, 5))

        # Model Accuracy Comparison
        plt.subplot(1, 2, 1)
        accuracies = [
            performance.get('final_train_accuracy', 0),
            performance.get('final_val_accuracy', 0),
            performance.get('best_val_accuracy', 0)
        ]
        labels = ['Training\nAccuracy', 'Final Val.\nAccuracy', 'Best Val.\nAccuracy']
        colors = ['#2ecc71', '#3498db', '#e74c3c']

        plt.bar(labels, accuracies, color=colors)
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.0)  # Set y-axis from 0 to 1
        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center')

        # Training Parameters
        plt.subplot(1, 2, 2)
        plt.axis('off')
        info_text = (
            f"Training Parameters:\n"
            f"Epochs: {training_params.get('num_epochs', 'N/A')}\n"
            f"Batch Size: {training_params.get('batch_size', 'N/A')}\n"
            f"Learning Rate: {training_params.get('learning_rate', 'N/A')}\n"
            f"Max Length: {training_params.get('max_length', 'N/A')}\n\n"
            f"Model Performance:\n"
            f"Final Training Acc: {performance.get('final_train_accuracy', 'N/A'):.3f}\n"
            f"Final Val. Acc: {performance.get('final_val_accuracy', 'N/A'):.3f}\n"
            f"Best Val. Acc: {performance.get('best_val_accuracy', 'N/A'):.3f}"
        )
        plt.text(0.1, 0.5, info_text, fontsize=10, va='center')

        plt.tight_layout()
        plt.show()

        # Print detailed metrics
        print("\n=== Model Performance Summary ===")
        print(f"Training Accuracy: {performance.get('final_train_accuracy', 'N/A'):.3f}")
        print(f"Validation Accuracy: {performance.get('final_val_accuracy', 'N/A'):.3f}")
        print(f"Best Validation Accuracy: {performance.get('best_val_accuracy', 'N/A'):.3f}")

        print("\n=== Training Parameters ===")
        for param, value in training_params.items():
            print(f"{param}: {value}")        