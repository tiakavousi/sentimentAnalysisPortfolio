import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

class AnalysisUtils:
    @staticmethod
    def plot_training_history(history):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def analyze_predictions(predictions, texts):
        sentiments = ['Negative', 'Neutral', 'Positive']
        for text, probs in zip(texts, predictions):
            sentiment_idx = np.argmax(probs)
            confidence = probs[sentiment_idx]
            print(f"\nText: {text}")
            print(f"Predicted sentiment: {sentiments[sentiment_idx]} ({confidence:.2%})")
            print("Probability distribution:")
            for sentiment, prob in zip(sentiments, probs):
                print(f"{sentiment}: {prob:.2%}")