import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import re
from data.data_processing import DataProcessor, TextSignals

class DataVisualizer:
    def __init__(self, data_processor=None):
        self.data_processor = data_processor if data_processor is not None else DataProcessor()
            
    @staticmethod
    def analyze_ratings_distribution(df):
        """Analyze and display the distribution of ratings"""
        print("\nRating Distribution:")
        total_samples = len(df)
        
        if 'label' in df.columns:
            for rating in sorted(df['label'].unique()):
                count = len(df[df['label'] == rating])
                percentage = (count / total_samples) * 100
                print(f"Rating {rating}: {count:,} reviews ({percentage:.1f}%)")
        elif 'sentiment' in df.columns:
            sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
            for sentiment in sorted(df['sentiment'].unique()):
                count = len(df[df['sentiment'] == sentiment])
                percentage = (count / total_samples) * 100
                print(f"{sentiment_map[sentiment]}: {count:,} reviews ({percentage:.1f}%)")
    
    @staticmethod
    def analyze_sentiment_distribution(df):
        """Analyze and display sentiment distribution including sarcasm info"""
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        
        print("\nSentiment Distribution:")
        sentiment_counts = df['sentiment'].value_counts().sort_index()
        for sentiment, count in sentiment_counts.items():
            print(f"{sentiment_map[sentiment]}: {count:,}")
        
        if 'is_sarcastic' in df.columns:
            print("\nSarcasm Distribution:")
            sarcasm_counts = df['is_sarcastic'].value_counts()
            print(f"Sarcastic: {sarcasm_counts.get(True, 0):,}")
            print(f"Non-sarcastic: {sarcasm_counts.get(False, 0):,}")
            
            # Cross-tabulation of sentiment and sarcasm
            print("\nSentiment-Sarcasm Distribution:")
            cross_tab = pd.crosstab(df['sentiment'], df['is_sarcastic'])
            for sentiment in sorted(df['sentiment'].unique()):
                sarcastic = cross_tab.loc[sentiment, True] if True in cross_tab.columns else 0
                non_sarcastic = cross_tab.loc[sentiment, False] if False in cross_tab.columns else 0
                print(f"{sentiment_map[sentiment]}:")
                print(f"  Sarcastic: {sarcastic:,}")
                print(f"  Non-sarcastic: {non_sarcastic:,}")
    
    @staticmethod
    def analyze_text_lengths(texts):
        """Analyze and plot text length distribution by words"""
        # Use processed text if available
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        word_lengths = [len(str(text).split()) for text in texts]
        
        plt.figure(figsize=(10, 5))
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
        token_lengths = tf.reduce_sum(encoded_data['attention_mask'], axis=1).numpy()
        
        mean_len = np.mean(token_lengths)
        median_len = np.median(token_lengths)
        max_len = max(token_lengths)
        q95_len = np.quantile(token_lengths, quantile)
        
        print(f"Token Length Statistics:")
        print(f"Mean: {mean_len:.1f}")
        print(f"Median: {median_len:.1f}")
        print(f"95th percentile: {q95_len:.1f}")
        print(f"Max: {max_len}")
        
        plt.figure(figsize=(10, 5))
        sns.histplot(token_lengths, bins=50)
        plt.axvline(q95_len, color='r', linestyle='--', label=f'{quantile*100}th percentile')
        plt.title('Distribution of Token Lengths')
        plt.xlabel('Number of Tokens')
        plt.ylabel('Count')
        plt.legend()
        plt.show()
        
        return int(q95_len)
    
    @staticmethod
    def visualize_wordclouds(df, min_word_length=2):
        """Generate and display word clouds for each sentiment class"""
        def preprocess_for_wordcloud(text):
            text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
            text = ' '.join(text.split())
            words = [word for word in text.split() if len(word) >= min_word_length]
            return ' '.join(words)
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        
        for sentiment in range(3):
            texts = df[df['sentiment'] == sentiment]['text']
            combined_text = ' '.join(texts.apply(preprocess_for_wordcloud))
            
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                colormap='viridis',
                random_state=42
            ).generate(combined_text)
            
            axes[sentiment].imshow(wordcloud, interpolation='bilinear')
            axes[sentiment].set_title(f'{sentiment_labels[sentiment]} Sentiment', fontsize=14, pad=20)
            axes[sentiment].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("\nMost Common Words by Sentiment:")
        for sentiment in range(3):
            texts = df[df['sentiment'] == sentiment]['text']
            words = ' '.join(texts.apply(preprocess_for_wordcloud)).split()
            word_freq = Counter(words).most_common(10)
            
            print(f"\n{sentiment_labels[sentiment]}:")
            for word, freq in word_freq:
                print(f"  {word}: {freq}")


    @staticmethod
    def analyze_text_signals(df):
        """
        Analyze and visualize the distribution of sarcasm and polarity in the dataset
        """
        plt.figure(figsize=(15, 5))
        
        # 1. Sarcasm Distribution
        plt.subplot(1, 3, 1)
        sarcasm_counts = df['is_sarcastic'].value_counts()
        sns.barplot(x=['Not Sarcastic', 'Sarcastic'], y=sarcasm_counts.values)
        plt.title('Distribution of Sarcasm')
        plt.ylabel('Count')
        
        # 2. Sarcasm by Sentiment
        plt.subplot(1, 3, 2)
        sentiment_sarcasm = pd.crosstab(
            df['sentiment'],
            df['is_sarcastic'],
            normalize='index'
        ) * 100
        sentiment_sarcasm.plot(kind='bar')
        plt.title('Sarcasm Distribution by Sentiment')
        plt.xlabel('Sentiment Class')
        plt.ylabel('Percentage')
        plt.legend(['Not Sarcastic', 'Sarcastic'])
        plt.xticks(range(3), ['Negative', 'Neutral', 'Positive'])
        
        # 3. Polarity Score Distribution
        plt.subplot(1, 3, 3)
        sns.histplot(df['polarity_score'], bins=20)
        plt.title('Distribution of Polarity Scores')
        plt.xlabel('Polarity Score')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistical analysis
        total_reviews = len(df)
        sarcastic_count = df['is_sarcastic'].sum()
        
        print(f"\n=== Text Signal Analysis ===")
        print(f"Total Reviews: {total_reviews:,}")
        
        print(f"\nSarcasm Detection:")
        print(f"- Sarcastic reviews: {sarcastic_count:,} ({(sarcastic_count/total_reviews*100):.2f}%)")
        print(f"- Non-sarcastic reviews: {total_reviews - sarcastic_count:,} "
              f"({((total_reviews - sarcastic_count)/total_reviews*100):.2f}%)")
        
        print("\nPolarity Score Statistics:")
        print(f"- Mean: {df['polarity_score'].mean():.3f}")
        print(f"- Median: {df['polarity_score'].median():.3f}")
        print(f"- Std Dev: {df['polarity_score'].std():.3f}")


    @staticmethod
    def display_processed_reviews(df, num_samples=5):
        """Display a sample of reviews with their associated labels and metrics."""
        try:
            samples = df.sample(n=min(num_samples, len(df)), random_state=42)
            
            for idx, row in samples.iterrows():
                sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
                sentiment_label = sentiment_map.get(row['sentiment'], 'Unknown')
                
                print(f"\n{'='*80}")
                print(f"Review #{idx}")
                print(f"\nOriginal Text:")
                print(f"{row['text'][:200]}..." if len(row['text']) > 200 else row['text'])
                
                if 'processed_text' in row:
                    print(f"\nProcessed Text:")
                    print(f"{row['processed_text'][:200]}..." if len(row['processed_text']) > 200 else row['processed_text'])
                
                print(f"\nLabels and Metrics:")
                print(f"- Sentiment: {row['sentiment']} ({sentiment_label})")
                print(f"- Sarcasm Detected: {row['is_sarcastic']}")
                print(f"- Polarity Score: {row['polarity_score']:.3f}")
                
        except Exception as e:
            print(f"Error displaying reviews: {str(e)}")
            print("Please ensure the DataFrame contains required columns: 'text', 'sentiment', 'is_sarcastic', 'polarity_score'")