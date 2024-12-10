import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import re
from data.data_processing import DataProcessor

class DataVisualizer:
    def __init__(self):
        self.data_processor = DataProcessor()
            
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
    def visualize_wordclouds(df, min_word_length=2):
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
    def analyze_text_signals(df):
        """
        Analyze and visualize the distribution of sarcasm and negation in the dataset
        and their relationship with sentiment classes.
        
        Args:
            df: DataFrame containing the reviews with 'text' column
        """
        # Process all texts to get sarcasm and negation info
        import pandas as pd
        results = []
        processor = DataProcessor()
        
        print("Processing texts... This may take a while...\n")
        
        for text in df['text']:
            processed_text, is_sarcastic = processor.preprocess_text(text)
            has_negation = any(neg in processed_text for neg in processor.sarcasm_detector.strong_markers)
            results.append({
                'sarcastic': is_sarcastic,
                'has_negation': has_negation
            })
        
        # Convert results to DataFrame for easy analysis
        results_df = pd.DataFrame(results)
        
        # Create visualization of distributions and relationships
        plt.figure(figsize=(15, 10))
        
        # 1. Overall Distribution of Signals
        plt.subplot(2, 2, 1)
        signal_counts = pd.DataFrame({
            'Signal': ['Sarcasm', 'Negation'],
            'Count': [results_df['sarcastic'].sum(), results_df['has_negation'].sum()]
        })
        sns.barplot(x='Signal', y='Count', data=signal_counts)
        plt.title('Distribution of Text Signals')
        
        # 2. Signal Co-occurrence
        plt.subplot(2, 2, 2)
        cross_tab = pd.crosstab(results_df['sarcastic'], results_df['has_negation'])
        sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Signal Co-occurrence')
        plt.xlabel('Has Negation')
        plt.ylabel('Is Sarcastic')
        
        # 3. Signals by Sentiment (if sentiment column exists)
        if 'sentiment' in df.columns:
            plt.subplot(2, 2, 3)
            df_with_signals = df.copy()
            df_with_signals['sarcastic'] = results_df['sarcastic']
            df_with_signals['has_negation'] = results_df['has_negation']
            
            # Create grouped bar plot
            sentiment_signals = pd.crosstab(
                df_with_signals['sentiment'],
                [df_with_signals['sarcastic']], 
                normalize='index'
            ) * 100
            sentiment_signals.plot(kind='bar', ax=plt.gca())
            plt.title('Sarcasm Distribution by Sentiment')
            plt.xlabel('Sentiment Class')
            plt.ylabel('Percentage')
            plt.legend(['Not Sarcastic', 'Sarcastic'])
            plt.xticks(range(3), ['Negative', 'Neutral', 'Positive'])
            
            # 4. Negation by Sentiment
            plt.subplot(2, 2, 4)
            sentiment_negation = pd.crosstab(
                df_with_signals['sentiment'],
                [df_with_signals['has_negation']], 
                normalize='index'
            ) * 100
            sentiment_negation.plot(kind='bar', ax=plt.gca())
            plt.title('Negation Distribution by Sentiment')
            plt.xlabel('Sentiment Class')
            plt.ylabel('Percentage')
            plt.legend(['No Negation', 'Has Negation'])
            plt.xticks(range(3), ['Negative', 'Neutral', 'Positive'])
        
        plt.tight_layout()
        plt.show()
        
        # Print statistical analysis
        total_reviews = len(df)
        sarcastic_count = results_df['sarcastic'].sum()
        negation_count = results_df['has_negation'].sum()
        
        print(f"\n=== Text Signal Analysis ===")
        print(f"Total Reviews Analyzed: {total_reviews:,}")
        
        print(f"\nSarcasm Detection:")
        print(f"- Reviews with sarcasm: {sarcastic_count:,} ({(sarcastic_count/total_reviews*100):.2f}%)")
        print(f"- Reviews without sarcasm: {total_reviews - sarcastic_count:,} "
            f"({((total_reviews - sarcastic_count)/total_reviews*100):.2f}%)")
        
        print(f"\nNegation Analysis:")
        print(f"- Reviews with negation: {negation_count:,} ({(negation_count/total_reviews*100):.2f}%)")
        print(f"- Reviews without negation: {total_reviews - negation_count:,} "
            f"({((total_reviews - negation_count)/total_reviews*100):.2f}%)")
        
        print("\nSignal Co-occurrence:")
        print(cross_tab)
        
        if 'sentiment' in df.columns:
            print("\nDistribution across Sentiment Classes:")
            sentiment_distribution = pd.crosstab(
                df_with_signals['sentiment'],
                [df_with_signals['sarcastic'], df_with_signals['has_negation']]
            )
            print("\nCounts for each combination (sentiment, sarcasm, negation):")
            print(sentiment_distribution)


    @staticmethod
    def display_processed_reviews(df, num_samples=5):
        """
        Display a sample of reviews with their associated labels.
        
        Args:
            df (pandas.DataFrame): DataFrame containing the reviews and labels
            num_samples (int): Number of reviews to display (default: 5)
        """
        try:
            # Create sample with all required fields
            samples = df.sample(n=min(num_samples, len(df)), random_state=42)
            
            for idx, row in samples.iterrows():
                # Initialize DataProcessor once outside the loop
                processor = DataProcessor()
                
                # Process the text to get sarcasm info
                processed_text, is_sarcastic = processor.preprocess_text(row['text'])
                
                # Calculate polarity score
                polarity_score = processor._calculate_polarity_score(row['text'])
                
                # Check for negation
                has_negation = '_NEG_' in processed_text
                
                # Get sentiment label
                sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
                sentiment_label = sentiment_map.get(row['sentiment'], 'Unknown')
                
                # Print formatted output
                print(f"\n{'='*80}")
                print(f"Review #{idx}")
                print(f"\nOriginal Text:")
                print(f"{row['text'][:200]}..." if len(row['text']) > 200 else row['text'])
                print(f"\nProcessed Text:")
                print(f"{processed_text[:200]}..." if len(processed_text) > 200 else processed_text)
                print(f"\nLabels:")
                print(f"- Sentiment: {row['sentiment']} ({sentiment_label})")
                print(f"- Sarcasm Detected: {is_sarcastic}")
                print(f"- Contains Negation: {has_negation}")
                print(f"- Polarity Score: {polarity_score:.2f}")
                
        except Exception as e:
            print(f"Error displaying reviews: {str(e)}")
            print("Please ensure the DataFrame contains 'text' and 'sentiment' columns")