import tensorflow as tf
from transformers import DistilBertTokenizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import random
from wordcloud import WordCloud
import re
from sklearn.metrics import confusion_matrix, classification_report
from data.data_processing import DataProcessor, TextSignals

class SentimentAnalysisVisualizer:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model = None
        self.tokenizer = None
        self.trainer = None
            
    @staticmethod
    def analyze_ratings_distribution(df):
        """Analyze and display the distribution of ratings"""
        
        print("\nRating Distribution:")
        total_samples = len(df)
        
        for rating in sorted(df['label'].unique()):
            count = len(df[df['label'] == rating])
            percentage = (count / total_samples) * 100
            print(f"Rating {rating}: {count:,} reviews ({percentage:.1f}%)")
    

    @staticmethod
    def analyze_sentiment_distribution(df):
        """Analyze and display sentiment distribution"""
        # Calculate sentiment distribution
        sentiment_counts = df['sentiment'].value_counts().sort_index()
        
        # Get actual sentiment values from the data
        unique_sentiments = sorted(df['sentiment'].unique())
        
        # Create labels based on actual values
        sentiment_labels = {}
        for sentiment in unique_sentiments:
            if sentiment < 0:
                sentiment_labels[sentiment] = "Negative"
            elif sentiment > 0:
                sentiment_labels[sentiment] = "Positive"
            else:
                sentiment_labels[sentiment] = "Neutral"
        
        print("\nSentiment Distribution:")
        for sentiment in unique_sentiments:
            print(f"{sentiment_labels[sentiment]}: {sentiment_counts[sentiment]:,}")
    
    @staticmethod
    def analyze_text_lengths(texts):
        """Analyze and plot text length distribution by words"""
        # Count words
        word_lengths = [len(text.split()) for text in texts]
        
        # Create figure
        plt.figure(figsize=(10, 5))
        
        # Word length distribution
        sns.histplot(word_lengths, bins=50)
        plt.title('Distribution of Text Lengths (Words)')
        plt.xlabel('Number of Words')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nText Length Statistics:")
        print("\nWord counts:")
        print(f"Average length: {np.mean(word_lengths):.1f} words")
        print(f"Median length: {np.median(word_lengths):.1f} words")
        print(f"Max length: {max(word_lengths)} words")
        print(f"Min length: {min(word_lengths)} words")
    

    @staticmethod
    def analyze_token_lengths(encoded_data, quantile=0.95):
        """
        Analyze and visualize the distribution of token lengths
        
        Parameters:
        -----------
        encoded_data : dict
            Dictionary containing 'input_ids' and 'attention_mask' tensors
        quantile : float
            Quantile to use for length suggestion (default: 0.95)
        """
        # Get token lengths directly from attention mask
        token_lengths = tf.reduce_sum(encoded_data['attention_mask'], axis=1).numpy()
        
        # Calculate statistics
        mean_len = np.mean(token_lengths)
        median_len = np.median(token_lengths)
        max_len = max(token_lengths)
        q95_len = np.quantile(token_lengths, quantile)
        
        print(f"Token Length Statistics:")
        print(f"Mean: {mean_len:.1f}")
        print(f"Median: {median_len:.1f}")
        print(f"95th percentile: {q95_len:.1f}")
        print(f"Max: {max_len}")
        
        # Plot distribution
        plt.figure(figsize=(10, 5))
        sns.histplot(token_lengths, bins=50)
        plt.axvline(q95_len, color='r', linestyle='--', label=f'{quantile*100}th percentile')
        plt.title('Distribution of Token Lengths')
        plt.xlabel('Number of Tokens')
        plt.ylabel('Count')
        plt.legend()
        plt.show()
        
        return int(q95_len)  # Return suggested MAX_LENGTH
    
    @staticmethod
    def visualize_sentiment_wordclouds(df, min_word_length=2):
        """
        Generate and display word clouds for each sentiment class using the balanced dataset.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The balanced dataset containing 'text' and 'sentiment' columns
        min_word_length : int, optional
            Minimum length of words to include in the word cloud (default=3)
        """
        def preprocess_for_wordcloud(text):
            # Convert to lowercase and remove special characters
            text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
            # Remove extra whitespace
            text = ' '.join(text.split())
            # Filter out short words
            words = [word for word in text.split() if len(word) >= min_word_length]
            return ' '.join(words)
        
        # Set up the plot
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        
        # Generate word cloud for each sentiment
        for sentiment in range(3):
            # Get texts for current sentiment
            texts = df[df['sentiment'] == sentiment]['text']
            
            # Combine all texts and preprocess
            combined_text = ' '.join(texts.apply(preprocess_for_wordcloud))
            
            # Create and generate word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                colormap='viridis',
                random_state=42
            ).generate(combined_text)
            
            # Plot the word cloud
            axes[sentiment].imshow(wordcloud, interpolation='bilinear')
            axes[sentiment].set_title(f'{sentiment_labels[sentiment]} Sentiment', fontsize=14, pad=20)
            axes[sentiment].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print most common words for each sentiment
        print("\nMost Common Words by Sentiment:")
        for sentiment in range(3):
            texts = df[df['sentiment'] == sentiment]['text']
            words = ' '.join(texts.apply(preprocess_for_wordcloud)).split()
            word_freq = Counter(words).most_common(10)
            
            print(f"\n{sentiment_labels[sentiment]}:")
            for word, freq in word_freq:
                print(f"  {word}: {freq}")


    @staticmethod
    def visualize_training_history(history):
        """Visualize model training metrics."""
        plt.figure(figsize=(15, 5))

        # Loss plots
        plt.subplot(1, 3, 1)
        for metric in ['loss', 'sentiment_loss', 'sarcasm_loss', 'negation_loss', 'polarity_loss']:
            if metric in history:
                plt.plot(history[metric], label=metric)
        plt.title('Model Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Accuracy plots
        plt.subplot(1, 3, 2)
        for metric in ['sentiment_accuracy', 'sarcasm_accuracy', 'negation_accuracy']:
            if metric in history:
                plt.plot(history[metric], label=metric)
        plt.title('Model Accuracies')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # MAE for polarity
        plt.subplot(1, 3, 3)
        if 'polarity_mae' in history:
            plt.plot(history['polarity_mae'], label='polarity_mae')
        plt.title('Polarity MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Print final metrics
        print("\nFinal Training Metrics:")
        for metric, values in history.items():
            print(f"{metric}: {values[-1]:.4f}")
