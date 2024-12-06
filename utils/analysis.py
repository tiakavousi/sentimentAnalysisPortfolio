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


    @staticmethod
    def get_random_words():
        """Return dictionaries of random words for template filling."""
        return {
            'item': random.choice(["product", "device", "appliance", "service", "system", "solution"]),
            'items': random.choice(["products", "devices", "appliances", "services", "systems", "solutions"]),
            'aspect1': random.choice(["user interface", "build quality", "performance", "reliability", "functionality"]),
            'aspect2': random.choice(["customer support", "design", "efficiency", "durability", "features"]),
            'aspect3': random.choice(["setup process", "maintenance", "compatibility", "user experience", "integration"]),
            'aspect4': random.choice(["value proposition", "performance stability", "quality control", "overall design"]),
            'feature': random.choice(["AI capabilities", "smart features", "automation system", "connectivity options"]),
            'pos_adj1': random.choice(["outstanding", "exceptional", "remarkable", "impressive", "excellent"]),
            'pos_adj2': random.choice(["innovative", "well-designed", "intuitive", "reliable", "efficient"]),
            'neg_adj1': random.choice(["disappointing", "frustrating", "unreliable", "problematic", "inadequate"]),
            'neg_adj2': random.choice(["defective", "cumbersome", "inefficient", "poorly designed", "unstable"]),
            'duration': random.choice(["three months", "six weeks", "several months", "a few weeks", "extensive testing"]),
            'price': random.choice(["$500", "$299", "$1000", "$750", "$150"]),
            'neg_aspect': random.choice(["initial setup", "learning curve", "price point", "minor issues"]),
            'pos_detail': random.choice([
                "The attention to detail is evident in every aspect.",
                "Every feature has been thoughtfully implemented.",
                "The quality is apparent from the moment you start using it."
            ]),
            'neg_detail': random.choice([
                "dealing with constant crashes and errors",
                "struggling with basic functionality",
                "waiting endlessly for responses"
            ]),
            'neutral_detail': random.choice([
                "Some features work well while others need improvement.",
                "The performance varies depending on usage.",
                "Your experience may vary based on specific needs."
            ]),
            'pos_conclusion': random.choice([
                "Highly recommended for anyone in the market.",
                "Definitely worth the investment.",
                "A stellar example of quality and innovation."
            ]),
            'neg_conclusion': random.choice([
                "Save your money and look elsewhere.",
                "Not worth the frustration or cost.",
                "A disappointing experience overall."
            ]),
            'neutral_conclusion': random.choice([
                "Consider your specific needs before purchasing.",
                "Might work for some, but not for others.",
                "An average option in this category."
            ])
        }

    @staticmethod
    def generate_review_templates():
        """Generate templates for different sentiment categories with complex features."""
        templates = {
            'positive': [
                "Despite initial concerns about {aspect1}, I was absolutely blown away by this {item}. The {aspect2} exceeded all expectations, and even though I've tried many similar {items} before, this one stands out. The {aspect3} was {pos_adj1}, the {aspect4} was {pos_adj2}, and I particularly enjoyed the {feature}. {pos_conclusion}",
                "I have to admit, I was skeptical at first about the {item}, especially given my past experiences. However, after {duration} of use, I'm thoroughly impressed. Not only is the {aspect1} {pos_adj1}, but the {aspect2} is also {pos_adj2}. {pos_detail} The {aspect3} works flawlessly, and the {feature} is an added bonus. {pos_conclusion}"
            ],
            'negative': [
                "I really wanted to love this {item}, but unfortunately, it falls short in almost every aspect. Sure, the {aspect1} might be {pos_adj1}, but that doesn't make up for the terrible {aspect2}. Despite what the marketing claims, the {aspect3} is {neg_adj1}, and the {aspect4} is even worse. {neg_conclusion}",
                "Don't waste your money on this {item}! While everyone raves about the {aspect1}, I found it to be completely {neg_adj1}. The {aspect2} stopped working after {duration}, the {aspect3} is {neg_adj2}, and customer service was no help at all. {neg_detail} Save yourself the trouble and look elsewhere. {neg_conclusion}"
            ],
            'neutral': [
                "The {item} has both pros and cons worth considering. While the {aspect1} is {pos_adj1} and the {aspect2} is decent, the {aspect3} leaves room for improvement. After {duration} of use, I can say it's neither exceptional nor terrible. {neutral_detail} The {aspect4} is average at best. {neutral_conclusion}",
                "I have mixed feelings about this {item}. On one hand, the {aspect1} is {pos_adj1} and the {aspect2} works well enough. However, the {aspect3} is {neg_adj1}, and the {feature} isn't anything special. {neutral_detail} For the {price}, it's exactly what you'd expect. {neutral_conclusion}"
            ]
        }
        return templates

    @staticmethod
    def generate_review(sentiment, templates):
        """Generate a single review with given sentiment."""
        template = random.choice(templates[sentiment])
        words = SentimentAnalysisVisualizer.get_random_words()
        review = template.format(**words)
        return review