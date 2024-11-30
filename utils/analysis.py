import tensorflow as tf
from transformers import DistilBertTokenizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data.data_processing import DataProcessor
from collections import Counter
import random

class SentimentAnalyzer:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def analyze_text_lengths(self, texts):
        """Analyze and plot text length distribution"""
        lengths = [len(text.split()) for text in texts]
        
        plt.figure(figsize=(10, 5))
        sns.histplot(lengths, bins=50)
        plt.title('Distribution of Text Lengths')
        plt.xlabel('Number of Words')
        plt.ylabel('Count')
        plt.show()
        
        print(f"\nText Length Statistics:")
        print(f"Average length: {np.mean(lengths):.1f} words")
        print(f"Median length: {np.median(lengths):.1f} words")
        print(f"Max length: {max(lengths)} words")
        print(f"Min length: {min(lengths)} words")
    
    def analyze_preprocessing(self, sample_size=10):
        """Display detailed preprocessing examples"""
        print("\nDetailed Preprocessing Analysis:")
        
        # Sample different types of texts
        test_texts = [
            "Great service and amazing food!",
            "Terrible experience, would not recommend.",
            "The food was okay, but the service could be better.",
            "Yeah right, like that's going to work...",
            "Thanks a lot... now everything is broken 🙄",
            "Obviously this is the best restaurant ever...",
            "I can't believe how bad this was!",
            "This couldn't possibly be any worse...",
            "The food wasn't very good, but the service was nice",
            "Never going back there again!"
        ]
        
        for i, text in enumerate(test_texts, 1):
            processed = self.data_processor.preprocess_text(text)
            print(f"\nExample {i}:")
            print(f"Original:  {text}")
            print(f"Processed: {processed}")
            
            # Analyze changes
            print("Changes detected:")
            if '_NEG' in processed:
                print("- Negation marking added")
            if '_SARC' in processed:
                print("- Sarcasm detected")
            if 'MULTI_EXCLAIM' in processed or 'ELLIPSIS' in processed:
                print("- Special tokens replaced")
            print("-" * 80)
    
    def analyze_class_distribution(self, labels):
        """Visualize class distribution"""
        plt.figure(figsize=(8, 5))
        sns.countplot(x=labels)
        plt.title('Distribution of Sentiment Classes')
        plt.xlabel('Sentiment (0: Negative, 1: Neutral, 2: Positive)')
        plt.ylabel('Count')
        plt.show()
        
        # Print class distribution
        class_dist = Counter(labels)
        total = len(labels)
        print("\nClass Distribution:")
        for sentiment, count in class_dist.items():
            print(f"Class {sentiment}: {count} samples ({count/total*100:.1f}%)")
    
    def process_data(self, verbose=True):
        """Process data with detailed analysis"""
        print("Loading and processing data...")
        df = self.data_processor.load_data()
        
        if verbose:
            print(f"\nTotal samples: {len(df)}")
            print("\nSample texts from each class:")
            for sentiment in range(3):
                samples = df[df['sentiment'] == sentiment]['text'].sample(3).tolist()
                print(f"\nClass {sentiment} samples:")
                for i, sample in enumerate(samples, 1):
                    print(f"{i}. {sample[:100]}...")
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = self.data_processor.split_data(df)
        
        if verbose:
            print("\nData Split Statistics:")
            print(f"Training samples: {len(train_texts)}")
            print(f"Validation samples: {len(val_texts)}")
            
            # Analyze text lengths
            print("\nAnalyzing text lengths...")
            self.analyze_text_lengths(train_texts)
            
            # Show class distribution
            print("\nAnalyzing class distribution...")
            self.analyze_class_distribution(train_labels)
            
            # Show preprocessing examples
            print("\nAnalyzing preprocessing...")
            self.analyze_preprocessing()
        
        return train_texts, val_texts, train_labels, val_labels
    
    def analyze_tokenization(self, texts, sample_size=5):
        """Analyze tokenization results"""
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        print("\nTokenization Analysis:")
        for i, text in enumerate(random.sample(texts.tolist(), sample_size), 1):
            tokens = self.tokenizer.tokenize(text)
            print(f"\nExample {i}:")
            print(f"Original text: {text[:100]}...")
            print(f"Tokenized ({len(tokens)} tokens): {tokens[:20]}...")
            print("-" * 80)
        
        # Analyze token length distribution
        token_lengths = [len(self.tokenizer.tokenize(text)) for text in texts]
        
        plt.figure(figsize=(10, 5))
        sns.histplot(token_lengths, bins=50)
        plt.title('Distribution of Token Lengths')
        plt.xlabel('Number of Tokens')
        plt.ylabel('Count')
        plt.show()
        
        print(f"\nToken Length Statistics:")
        print(f"Average tokens per text: {np.mean(token_lengths):.1f}")
        print(f"Median tokens per text: {np.median(token_lengths):.1f}")
        print(f"Max tokens: {max(token_lengths)}")
        print(f"Min tokens: {min(token_lengths)}")

def main():
    analyzer = SentimentAnalyzer()
    
    # Process data with detailed analysis
    train_texts, val_texts, train_labels, val_labels = analyzer.process_data(verbose=True)
    
    # Analyze tokenization
    analyzer.analyze_tokenization(train_texts)
    
    print("\nAnalysis complete! The processed data is ready for model training.")

if __name__ == "__main__":
    main()