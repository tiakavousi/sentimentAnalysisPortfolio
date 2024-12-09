from datasets import load_dataset
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from config.model_config import ModelConfig


class TextSignals:
    PUNCTUATION = ['!!!', '...', '!?', '??']
    EMOJI = ['🙄', '😒', '😏', ':/']
    NEGATION_WORDS = ['not', 'never', "n't", 'no', 'neither', 'nor']
    
    # URL regex pattern
    URL_PATTERN = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    WWW_PATTERN = r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    SPECIAL_TOKENS = {
        '!!!': ' MULTI_EXCLAIM ',
        '...': ' ELLIPSIS ',
        '??': ' MULTI_QUESTION ',
        ':/': ' SKEPTICAL_EMOJI ',
        ':|': ' NEUTRAL_EMOJI ',
        ':)': ' HAPPY_EMOJI ',
        ':))': ' VERY_HAPPY_EMOJI ',
        ':(': ' SAD_EMOJI ',
        ':((': ' VERY_SAD_EMOJI ',
        ';)': ' WINK_EMOJI ',
        '🙄': ' EYE_ROLL_EMOJI ',
        '😒': ' UNAMUSED_EMOJI ',
        '😏': ' SMIRK_EMOJI ',
        '😤': ' FRUSTRATED_EMOJI ',
        '🙃': ' UPSIDE_DOWN_EMOJI ',
        '😑': ' EXPRESSIONLESS_EMOJI ',
        '😐': ' NEUTRAL_FACE_EMOJI ',
        '🤔': ' THINKING_EMOJI ',
        '🫤': ' DIAGONAL_MOUTH_EMOJI ',
        '😅': ' SWEAT_SMILE_EMOJI ',
        '😂': ' LAUGH_TEARS_EMOJI ',
        '🤣': ' ROFL_EMOJI ',
        '👍': ' THUMBS_UP_EMOJI ',
        '👎': ' THUMBS_DOWN_EMOJI ',
    }

    CONTRACTION_MAP = {
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


    @staticmethod
    def clean_urls(text):
        """Remove URLs from text."""
        # Remove http(s) URLs
        text = re.sub(TextSignals.URL_PATTERN, ' ', text)
        # Remove www. URLs
        text = re.sub(TextSignals.WWW_PATTERN, ' ', text)
        # Clean up any extra whitespace
        text = ' '.join(text.split())
        return text


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
            'just what i needed',
            'big surprise'
        ]
        
        # Context-dependent markers
        self.contextual_markers = {
            'thanks a lot': {
                'negative_signals': ['but', 'for nothing', 'now', "didn't", 'not'],
                'punctuation': TextSignals.PUNCTUATION,
                'emoji': TextSignals.EMOJI
            },
            'thank you so much': {
                'negative_signals': ['but', 'for nothing', "didn't", 'not'],
                'punctuation': TextSignals.PUNCTUATION,
                'emoji': TextSignals.EMOJI
            },
            'obviously': {
                'negative_signals': ['not', "didn't", 'never', 'but'],
                'punctuation': TextSignals.PUNCTUATION,
                'emoji': TextSignals.EMOJI
            }
        }

    def detect_sarcasm(self, text):
            # Expect already lowercased text
            for marker in self.strong_markers:
                if marker in text:
                    return True, marker
                    
            for marker, signals in self.contextual_markers.items():
                if marker in text:
                    has_negative = any(signal in text for signal in signals['negative_signals'])
                    has_punctuation = any(punct in text for punct in TextSignals.PUNCTUATION)
                    has_emoji = any(emoji in text for emoji in TextSignals.EMOJI)
                    
                    if sum([has_negative, has_punctuation, has_emoji]) >= 2:
                        return True, f"{marker} (contextual)"
                        
            return False, None


class DataProcessor:
    def __init__(self):
        self.sarcasm_detector = SarcasmDetector()

    def _calculate_polarity_score(self, text):
        # Calculate a polarity score (0 to 1) indicating the degree of mixed sentiment in the text,
        # considering contrasting sentiment markers, 'but' clauses, and negations.
        processed_text, _ = self.preprocess_text(text)
        
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
    
    
    
    def load_data(self):
        """Load the Yelp Review dataset and convert to DataFrame"""
        dataset = load_dataset(ModelConfig.YELP_DATASET)

        df = pd.DataFrame(dataset['train'])
        
        # adding sentiment column based on label
        df['sentiment'] = df['label'].apply(
            lambda x: 0 if x <= 1 else (1 if x == 2 else 2)
        )
        return df
    
    
    def create_balanced_dataset(self, df, samples_per_class=ModelConfig.SAMPLES_PER_CLASS):
        """Create a balanced dataset with equal samples per sentiment class"""
        df_balanced = pd.DataFrame()
        for sentiment in range(3):
            class_data = df[df['sentiment'] == sentiment].sample(
                n=samples_per_class, 
                random_state=42
            )
            df_balanced = pd.concat([df_balanced, class_data])
        
        # Shuffle the final dataset
        return df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    


    def split_data(self, df, val_size=0.2, test_size=0.1):
        """
        Split data into train, validation and test sets using a 7:2:1 ratio.
        
        Args:
            df: Input DataFrame
            val_size: Fraction of data for validation (default 0.2 for 20%)
            test_size: Fraction of data for testing (default 0.1 for 10%)
        
        Returns:
            Tuple of (train_texts, val_texts, test_texts, train_labels, val_labels, test_labels)
        """
        # First split off the test set (10%)
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42,
            stratify=df['sentiment']
        )
        
        # Then split the remaining data into train and validation
        # For remaining 90%, we want a 7:2 split (approximately 77.8% : 22.2% of remaining data)
        effective_val_size = val_size / (1 - test_size)  # This will be 0.22222...
        
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=effective_val_size,
            random_state=42,
            stratify=train_val_df['sentiment']
        )
        
        # Print split sizes to verify distribution
        print(f"Training set size: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Validation set size: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
        print(f"Test set size: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        
        # Convert to numpy arrays
        train_texts = train_df['text'].to_numpy()
        val_texts = val_df['text'].to_numpy()
        test_texts = test_df['text'].to_numpy()
        
        def create_label_dict(df):
            return {
                'sentiment': df['sentiment'].to_numpy(),
                'sarcasm': df['text'].apply(lambda x: '_SARC_' in x).to_numpy(),
                'negation': df['text'].apply(lambda x: '_NEG_' in x).to_numpy(),
                'polarity': df['text'].apply(self._calculate_polarity_score).to_numpy()
            }
        
        train_labels = create_label_dict(train_df)
        val_labels = create_label_dict(val_df)
        test_labels = create_label_dict(test_df)
        
        return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels

    
    
    def preprocess_text(self, text):
        # Clean URLs first
        text = TextSignals.clean_urls(text)
        
        # Convert to lowercase
        processed_text = text.lower() 
        
        # Expand contractions
        for contraction, expanded in TextSignals.CONTRACTION_MAP.items():
            processed_text = processed_text.replace(contraction, expanded)
        
        # Handle special tokens
        for token, replacement in TextSignals.SPECIAL_TOKENS.items():
            processed_text = processed_text.replace(token, replacement)
        
        # Mark negations
        for word in TextSignals.NEGATION_WORDS:
            processed_text = processed_text.replace(f'{word} ', f'{word}_NEG ')
        
        # Detect sarcasm
        is_sarcastic, marker = self.sarcasm_detector.detect_sarcasm(processed_text)
        if is_sarcastic:
            processed_text += f" _SARC_{marker}"
        
        return processed_text, is_sarcastic
    
    

    def process_batch(self, texts):
        processed_texts = []
        analysis = {
            'sarcasm_count': 0,
            'negation_count': 0,
            'special_tokens_count': 0,
            'url_count': 0 
        }
        
        for text in texts:
            # Count URLs before removing them
            url_count = len(re.findall(TextSignals.URL_PATTERN, text))
            url_count += len(re.findall(TextSignals.WWW_PATTERN, text))
            analysis['url_count'] += url_count
            
            # Process with consistent lowercase handling
            processed_text, is_sarcastic = self.preprocess_text(text)
            
            # All comparisons now use lowercase text
            if is_sarcastic:
                analysis['sarcasm_count'] += 1

            if any(neg in processed_text for neg in TextSignals.NEGATION_WORDS):
                analysis['negation_count'] += 1
    
            if any(token in processed_text for token in TextSignals.SPECIAL_TOKENS):
                analysis['special_tokens_count'] += 1
                
            processed_texts.append(processed_text)
            
        return processed_texts, analysis