from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

class SarcasmDetector:
    def __init__(self):
         # Initialize the sarcasm detector with strong markers, context-dependent 
        # markers, punctuation signals, and emoji signals that indicate sarcasm.
        
        # Strong sarcasm indicators (high confidence)
        self.strong_markers = [
            'yeah right',
            'suuure',
            'riiight',
            'shock horror',
            'surprise surprise',
            'oh great job',
            'just what I needed',
            'big surprise'
        ]
        
        # Context-dependent markers (need additional signals)
        self.contextual_markers = {
            'thanks a lot': {
                'negative_signals': ['but', 'for nothing', 'now', "didn't", 'not'],
                'punctuation': ['!!!', '...', '!?'],
                'emoji': ['🙄', '😒', ':/']
            },
            'thank you so much': {
                'negative_signals': ['but', 'for nothing', "didn't", 'not'],
                'punctuation': ['!!!', '...', '!?'],
                'emoji': ['🙄', '😒', ':/']
            },
            'obviously': {
                'negative_signals': ['not', "didn't", 'never', 'but'],
                'punctuation': ['...', '!?'],
                'emoji': ['🙄', '😒']
            }
        }
        
        # Punctuation that might indicate sarcasm when combined with other signals
        self.punctuation_signals = ['!!!', '...', '!?', '??']
        
        # Emojis that might indicate sarcasm when combined with other signals
        self.emoji_signals = ['🙄', '😒', '😏', ':/']

    def detect_sarcasm(self, text):
        # Detect sarcasm in the given text by checking for strong markers 
        # and contextual markers combined with additional signals.
        text_lower = text.lower()
        
        # Check for strong markers (these alone indicate sarcasm)
        for marker in self.strong_markers:
            if marker in text_lower:
                return True, marker
        
        # Check for contextual markers
        for marker, signals in self.contextual_markers.items():
            if marker in text_lower:
                # Look for negative context signals
                has_negative = any(signal in text_lower for signal in signals['negative_signals'])
                has_punctuation = any(punct in text for punct in signals['punctuation'])
                has_emoji = any(emoji in text for emoji in signals['emoji'])
                
                # Require at least two signals for contextual markers
                if sum([has_negative, has_punctuation, has_emoji]) >= 2:
                    return True, f"{marker} (contextual)"
                
        return False, None

    def process_text(self, text):
        # Process the text to detect sarcasm and mark it if found.
        is_sarcastic, marker = self.detect_sarcasm(text)
        if is_sarcastic:
            text += f" _SARC_{marker}"
        return text, is_sarcastic

class DataProcessor:
    def __init__(self):
        # Initialize the data processor with contraction mappings, special token 
        # mappings, negation words, and a sarcasm detector instance.
        self.contraction_map = {
            "can't": "cannot",
            "won't": "will not",
            "n't": " not",
            "'re": " are",
            "'s": " is",
            "'d": " would",
            "'ll": " will",
            "'ve": " have",
            "'m": " am",
            "it's": "it is",
            "let's": "let us",
            "who's": "who is",
            "what's": "what is",
            "there's": "there is",
            "we're": "we are",
            "they're": "they are",
            "you're": "you are",
            "he's": "he is",
            "she's": "she is",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "hasn't": "has not",
            "haven't": "have not",
            "doesn't": "does not",
            "don't": "do not",
            "didn't": "did not",
            "shouldn't": "should not",
            "wouldn't": "would not",
            "couldn't": "could not",
            "mightn't": "might not",
            "mustn't": "must not",
        }
        self.special_tokens = {
            '!!!': ' MULTI_EXCLAIM ',
            '...': ' ELLIPSIS ',
            '??': ' MULTI_QUESTION ',
            ':)': ' HAPPY_EMOJI ',
            ':(': ' SAD_EMOJI ',
            ':/': ' SKEPTICAL_EMOJI '
        }
        self.negation_words = ['not', 'never', "n't", 'no', 'neither', 'nor']
        self.sarcasm_detector = SarcasmDetector()


    def _calculate_polarity_score(self, text):
        # Calculate a polarity score (0 to 1) indicating the degree of mixed sentiment in the text,
        # considering contrasting sentiment markers, 'but' clauses, and negations.
        processed_text = self.preprocess_text(text)
        
        # Check for contrasting sentiment markers
        positive_markers = ['good', 'great', 'excellent', 'amazing', 'wonderful']
        negative_markers = ['bad', 'terrible', 'horrible', 'awful', 'poor']
        
        # Count positive and negative markers
        pos_count = sum(marker in processed_text.lower() for marker in positive_markers)
        neg_count = sum(marker in processed_text.lower() for marker in negative_markers)
        
        # Check for 'but' clauses which often indicate mixed sentiment
        has_but = 'but' in processed_text.lower()
        
        # Check for negation which might flip sentiment
        has_negation = '_NEG_' in processed_text
        
        # Calculate score (0 to 1, higher means more mixed/polar)
        if has_but:
            score = 0.7  # Base score for contrasting statements
        else:
            score = 0.3  # Base score for regular statements
        
        # Adjust score based on sentiment markers
        if pos_count > 0 and neg_count > 0:
            score += 0.3  # Increase score for mixed sentiment
        
        if has_negation:
            score += 0.2  # Increase score for negation
            
        return min(score, 1.0)  # Cap score at 1.0
    
    
    def load_data(self, samples_per_class=2000):
        # Load the Yelp Review dataset, balance the classes by sampling an equal 
        # number of reviews per sentiment class, and shuffle the data.
        dataset = load_dataset("yelp_review_full")
        df = pd.DataFrame(dataset['train'])
        df['sentiment'] = df['label'].apply(
            lambda x: 0 if x <= 1 else (1 if x == 2 else 2)
        )
        
        df_balanced = pd.DataFrame()
        for sentiment in range(3):
            class_data = df[df['sentiment'] == sentiment].sample(
                n=samples_per_class, 
                random_state=42
            )
            df_balanced = pd.concat([df_balanced, class_data])
        
        return df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    def split_data(self, df, test_size=0.1):
        # Split the data into train and validation sets, stratifying by sentiment.
        # Return the text and label dictionaries for each split.
        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42,
            stratify=df['sentiment']
        )
        train_texts = train_df['text']
        val_texts = val_df['text']
        # Create label dictionaries
        train_labels = {
            'sentiment': train_df['sentiment'],
            'sarcasm': train_df['text'].apply(lambda x: '_SARC_' in x),
            'negation': train_df['text'].apply(lambda x: '_NEG_' in x),
            'polarity': train_df['text'].apply(self._calculate_polarity_score)
        }
        
        val_labels = {
            'sentiment': val_df['sentiment'],
            'sarcasm': val_df['text'].apply(lambda x: '_SARC_' in x),
            'negation': val_df['text'].apply(lambda x: '_NEG_' in x),
            'polarity': val_df['text'].apply(self._calculate_polarity_score)
        }
        return train_texts, val_texts, train_labels, val_labels
    
    def preprocess_text(self, text):
        # Preprocess the text by expanding contractions, handling special tokens,
        # marking negations, and detecting sarcasm using the enhanced detector.

        # Expand contractions
        for contraction, expanded in self.contraction_map.items():
            text = text.replace(contraction, expanded)
        
        # Handle special tokens
        for token, replacement in self.special_tokens.items():
            text = text.replace(token, replacement)
        
        # Mark negations
        for word in self.negation_words:
            text = text.replace(f'{word} ', f'{word}_NEG ')
        
        # Detect sarcasm using the enhanced detector
        processed_text, is_sarcastic = self.sarcasm_detector.process_text(text)
        
        return processed_text

    def process_batch(self, texts):
        """Process a batch of texts with detailed analysis"""
        # Process a batch of texts with detailed analysis, including sarcasm detection,
        # negation marking, and special token handling. Return the processed texts
        # and analysis counts.
        processed_texts = []
        analysis = {
            'sarcasm_count': 0,
            'negation_count': 0,
            'special_tokens_count': 0
        }
        
        for text in texts:
            # Process text and detect sarcasm
            processed_text, is_sarcastic = self.sarcasm_detector.process_text(text)
            
            # Apply other preprocessing
            processed_text = self.preprocess_text(processed_text)
            
            # Update analysis
            if is_sarcastic:
                analysis['sarcasm_count'] += 1
            if any(neg in processed_text for neg in self.negation_words):
                analysis['negation_count'] += 1
            if any(token in text for token in self.special_tokens):
                analysis['special_tokens_count'] += 1
                
            processed_texts.append(processed_text)
            
        return processed_texts, analysis