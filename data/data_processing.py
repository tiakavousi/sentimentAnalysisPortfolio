# ✅ Load and preprocess full dataset
# ✅ Detect natural sarcasm before splitting
# ✅ Split data
# ✅ Balance classes while preserving natural sarcasm
# ✅ Add synthetic sarcasm only to training set
# ✅ Prepare model inputs

from datasets import load_dataset
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from config.model_config import Config
import random
from typing import Optional, Tuple, Dict


class SarcasmAugmenter:
    """
    Augments text data with synthetic sarcastic variants.
    
    Generates sarcastic text using predefined patterns, intensifiers, and emojis
    while maintaining sentiment context. Focuses on converting negative sentiment
    to sarcastic positive expressions.
    """
    def __init__(self):
        """Initialize sarcasm patterns and modifiers for text augmentation."""
        self.sarcasm_patterns = {
            'positive_to_negative': [
                # Disbelief patterns
                "Oh wow, {text} ... just brilliant",
                "Sure, {text} ... because that makes total sense",
                "Yeah right, {text}",
                "How wonderful, {text}",
                "Just what everyone needed, {text}",
                "Because obviously {text}",
                
                # Exaggerated praise
                "I'm truly amazed that {text}",
                "This is pure genius: {text}",
                "Simply masterful how {text}",
                "What a breakthrough: {text}",
                "Revolutionary idea: {text}",
                
                # Mock enthusiasm
                "I'm jumping for joy that {text}",
                "My life is complete now that {text}",
                "Finally, my prayers are answered: {text}",
                "This changes everything: {text}",
                
                # Feigned gratitude
                "Thanks for showing us that {text}",
                "So grateful to learn that {text}",
                "What a blessing that {text}",
                
                # Ironic observations
                "Clearly the best part is how {text}",
                "Nothing could possibly go wrong when {text}",
                "Absolutely foolproof: {text}",
                "Can't possibly fail: {text}",
                
                # Mock agreement
                "Oh totally, because {text}",
                "Makes perfect sense that {text}",
                "I completely understand why {text}",
                
                # Exaggerated impact
                "This will definitely solve everything: {text}",
                "World-changing revelation: {text}",
                "History in the making: {text}"
            ],
            'negative_to_positive': [
                # False enthusiasm
                "This is totally the best when {text}",
                "I'm thrilled that {text}",
                "Nothing better than when {text}",
                "So happy that {text}",
                "Living the dream when {text}",
                
                # Mock optimism
                "Looking forward to more of {text}",
                "Can't wait for next time {text}",
                "Already excited about {text}",
                "The joy of experiencing {text}",
                
                # Fake contentment
                "Just what I always wanted: {text}",
                "Exactly how I hoped {text}",
                "Perfect, simply perfect how {text}",
                
                # False appreciation
                "Really appreciate how {text}",
                "So blessed to experience {text}",
                "Lucky us, getting to see {text}",
                
                # Mock satisfaction
                "Absolutely satisfying when {text}",
                "Couldn't ask for more than {text}",
                "Just what the doctor ordered: {text}",
                
                # Feigned excitement
                "The highlight of my day: {text}",
                "Simply cannot contain my joy that {text}",
                "Exactly what we needed: {text}",
                
                # Mock relief
                "Such a relief that {text}",
                "Thank goodness for {text}",
                "Finally, salvation: {text}"
            ]
        }
        
        self.intensifiers = [
            # Standard intensifiers
            "absolutely", "totally", "completely", "utterly",
            "definitely", "certainly", "surely", "obviously",
            
            # Added intensifiers
            "undoubtedly", "positively", "unquestionably", "indisputably",
            "without a doubt", "beyond question", "clearly", "plainly",
            "unmistakably", "undeniably", "genuinely", "truly",
            "perfectly", "entirely", "thoroughly", "hundred percent"
        ]
        
        self.sarcastic_endings = [
            # Direct contradiction
            "... NOT!", "... right.", "... sure.", 
            "... whatever.", "... if you say so.",
            "... like that's gonna work.",
            
            # Added endings
            "... as if!", "... yeah right.",
            "... in your dreams.", "... good luck with that.",
            "... I'm sure that'll work out great.",
            "... what could possibly go wrong?",
            "... because that makes sense.",
            "... genius plan right there.",
            "... totally believable.",
            "... seems legit.",
            "... that'll end well.",
            "... brilliant strategy.",
            "... way to go.",
            "... nailed it."
        ]
        
        self.emojis = [
            " _EYE_ROLL_EMOJI ", " _UNAMUSED_EMOJI ",
            " _SMIRK_EMOJI ", " _UPSIDE_DOWN_EMOJI ",
            " _THINKING_EMOJI ", " _EXPRESSIONLESS_EMOJI ",
            " _SIDE_EYE_EMOJI ", " _RAISED_EYEBROW_EMOJI ",
            " _STRAIGHT_FACE_EMOJI ", " _SMIRKING_EMOJI "
        ]

    def _create_sarcastic_variant(self, text: str, sentiment: int) -> Tuple[str, bool]:
        """
        Create a sarcastic variant of input text based on sentiment.
        
        Args:
            text: Input text to convert
            sentiment: Sentiment label (0=negative, 1=neutral, 2=positive)
            
        Returns:
            Tuple of (augmented text, success flag)
        """        
        if sentiment != 0:
            return text, False
    
        if sentiment == 0:  # negative
            pattern_key = 'negative_to_positive'
            
        try:
            pattern = random.choice(self.sarcasm_patterns[pattern_key])
            
            if random.random() < 0.5:
                text = f"{random.choice(self.intensifiers)} {text}"
            
            augmented = pattern.format(text=text)
            
            if random.random() < 0.3:
                augmented += random.choice(self.sarcastic_endings)
            
            if random.random() < 0.4:
                augmented += random.choice(self.emojis)
                
            return augmented, True
            
        except Exception as e:
            print(f"Failed to create sarcastic variant: {str(e)}")
            return text, False



    def create_balanced_sarcastic_dataset(self, df: pd.DataFrame, sarcasm_ratio: float = 0.4) -> pd.DataFrame:
        """
        Create balanced dataset with mix of natural and synthetic sarcasm.
        
        Args:
            df: Input DataFrame containing text and sentiment labels
            sarcasm_ratio: Target ratio of sarcastic samples per class
            
        Returns:
            Balanced DataFrame with original and synthetic sarcastic samples
        """
        balanced_data = []
        samples_per_class = Config.SAMPLES_PER_CLASS
        target_sarcastic = int(samples_per_class * sarcasm_ratio)
        
        print(f"Creating balanced dataset with:")
        print(f"- {samples_per_class} reviews per class")
        print(f"- {target_sarcastic} sarcastic reviews per class")
        
        for sentiment in [0, 1, 2]:
            class_data = df[df['sentiment'] == sentiment]
            class_samples = []
            
            # First, preserve naturally sarcastic samples
            natural_sarcastic = class_data[class_data['is_sarcastic'] == True]
            for _, row in natural_sarcastic.iterrows():
                class_samples.append({
                    'text': row['text'],
                    'processed_text': row['processed_text'],
                    'sentiment': row['sentiment'],
                    'is_sarcastic': True,
                    'sarcasm_source': 'natural'
                })
            
            # Add synthetic sarcasm to reach target
            if sentiment == 0:
                remaining_sarcastic_needed = target_sarcastic - len(natural_sarcastic)
                if remaining_sarcastic_needed > 0:
                    reviews_to_augment = class_data[
                        ~class_data.index.isin(natural_sarcastic.index) & 
                        (class_data['is_sarcastic'] == False)
                    ].sample(
                        n=min(len(class_data), remaining_sarcastic_needed),
                        random_state=Config.RANDOM_SEED
                    )
                    
                    for _, row in reviews_to_augment.iterrows():
                        augmented, success = self._create_sarcastic_variant(
                            row['processed_text'],
                            row['sentiment']
                        )
                        if success:
                            class_samples.append({
                                'text': row['text'],
                                'processed_text': augmented,
                                'sentiment': row['sentiment'],
                                'is_sarcastic': True,
                                'sarcasm_source': 'augmentation'
                            })
                
            # Fill remaining with non-sarcastic samples
            remaining_needed = samples_per_class - len(class_samples)
            if remaining_needed > 0:
                non_sarcastic_samples = class_data[
                    ~class_data.index.isin(natural_sarcastic.index) &
                    (class_data['is_sarcastic'] == False)
                ].sample(
                    n=min(remaining_needed, len(class_data)),
                    random_state=Config.RANDOM_SEED,
                    replace=True
                )
                
                for _, row in non_sarcastic_samples.iterrows():
                    class_samples.append({
                        'text': row['text'],
                        'processed_text': row['processed_text'],
                        'sentiment': row['sentiment'],
                        'is_sarcastic': False,
                        'sarcasm_source': 'none'
                    })
            
            balanced_data.extend(class_samples)
        
        balanced_df = pd.DataFrame(balanced_data)
        
        # Print distribution statistics
        print("\nFinal distribution:")
        for sentiment in [0, 1, 2]:
            sentiment_data = balanced_df[balanced_df['sentiment'] == sentiment]
            natural = sentiment_data[sentiment_data['sarcasm_source'] == 'natural']
            augmented = sentiment_data[sentiment_data['sarcasm_source'] == 'augmentation']
            print(f"\nSentiment {sentiment}:")
            print(f"- Total samples: {len(sentiment_data)}")
            print(f"- Natural sarcasm: {len(natural)}")
            print(f"- Augmented sarcasm: {len(augmented)}")
            print(f"- Total sarcastic: {len(natural) + len(augmented)} ({(len(natural) + len(augmented))/len(sentiment_data)*100:.1f}%)")
        
        return balanced_df.sample(frac=1, random_state=Config.RANDOM_SEED).reset_index(drop=True)


class TextSignals:
    """
    Defines text preprocessing constants and patterns.
    
    Contains mappings for punctuation, emojis, contractions, and special tokens
    used in text normalization and feature extraction.
    """
    PUNCTUATION = ['!!!', '...', '!?', '??']
    EMOJI = ['🙄', '😒', '😏', ':/']
    # NEGATION_WORDS = ['not', 'never', "n't", 'no', 'neither', 'nor']
    
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
        """
        Remove URLs from text using regex patterns.
        
        Args:
            text: Input text containing potential URLs
            
        Returns:
            Text with URLs removed and whitespace normalized
        """
        # Remove http(s) URLs
        text = re.sub(TextSignals.URL_PATTERN, ' ', text)
        # Remove www. URLs
        text = re.sub(TextSignals.WWW_PATTERN, ' ', text)
        # Clean up any extra whitespace
        text = ' '.join(text.split())
        return text


class SarcasmDetector:
    """
    Detects natural sarcasm in text using linguistic markers.
    
    Uses both strong markers for direct sarcasm detection and contextual
    markers that consider surrounding signals like punctuation and emojis.
    """
    def __init__(self):
        """Initialize sarcasm detection markers and rules."""
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

    def detect_sarcasm(self, text)-> Tuple[bool, Optional[str]]:
        """
        Detect sarcasm in input text.
        
        Checks for both strong sarcasm markers and contextual indicators
        using punctuation, emojis, and negative signals.
        
        Args:
            text: Preprocessed text to analyze
            
        Returns:
            Tuple of (is_sarcastic flag, detected marker or None)
        """

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
    """
    Main data processing pipeline for sentiment analysis dataset.
    
    Handles dataset loading, preprocessing, splitting, balancing, and
    sarcasm augmentation. Prepares data in format required by model.
    """
    def __init__(self):
        """Initialize preprocessing components."""
        self.sarcasm_detector = SarcasmDetector()
        self.augmenter = SarcasmAugmenter()

    #1 load_data() - Loads Yelp dataset
    def load_data(self) -> pd.DataFrame:
        """
        Load and preprocess Yelp review dataset.
        
        Loads raw data, applies text preprocessing, detects natural sarcasm,
        and assigns sentiment labels.
        
        Returns:
            DataFrame with processed texts and initial labels
        """
        # Load dataset
        dataset = load_dataset(Config.YELP_DATASET)
        df = pd.DataFrame(dataset['train'])
        
        # Preprocess all texts first
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Detect natural sarcasm
        df['is_sarcastic'] = df['processed_text'].apply(
            lambda x: self.sarcasm_detector.detect_sarcasm(x)[0]
        )
        
        # Add sentiment labels
        df['sentiment'] = df['label'].apply(lambda x: 0 if x < 2 else (1 if x == 2 else 2))
        
        return df
    
    #2 split_data() - Splits into train/val/test
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation and test sets.
        
        Performs stratified split based on sentiment labels to maintain
        class distribution across splits.
        
        Args:
            df: Input DataFrame to split
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_val_df, test_df = train_test_split(
            df,
            test_size = Config.TEST_SPLIT,
            random_state = Config.RANDOM_SEED,
            stratify = df['sentiment']
        )
        
        val_size = Config.VALIDATION_SPLIT / (1 - Config.TEST_SPLIT)

        train_df, val_df = train_test_split(
            train_val_df,
            test_size = val_size,
            random_state = Config.RANDOM_SEED,
            stratify = train_val_df['sentiment']
        )
        
        return train_df, val_df, test_df
    
    #3 _balance_classes() - Balances class distribution
    def _balance_classes(self, df: pd.DataFrame, samples_per_class: int = Config.SAMPLES_PER_CLASS) -> pd.DataFrame:
        """
        Balance class distribution while preserving natural sarcasm.
        
        Ensures equal samples per sentiment class while prioritizing
        retention of naturally sarcastic samples.
        
        Args:
            df: Input DataFrame to balance
            samples_per_class: Target number of samples per class
            
        Returns:
            Balanced DataFrame with preserved natural sarcasm
        """
        df_balanced = pd.DataFrame()
        
        for sentiment in range(3):
            sentiment_data = df[df['sentiment'] == sentiment]
            natural_sarcastic = sentiment_data[sentiment_data['is_sarcastic'] == True]
            non_sarcastic = sentiment_data[sentiment_data['is_sarcastic'] == False]
            
            remaining_needed = samples_per_class - len(natural_sarcastic)
            if remaining_needed > 0:
                sampled_non_sarcastic = non_sarcastic.sample(
                    n = min(len(non_sarcastic), remaining_needed),
                    replace = len(non_sarcastic) < remaining_needed,
                    random_state = Config.RANDOM_SEED
                )
                sampled_data = pd.concat([natural_sarcastic, sampled_non_sarcastic])
            else:
                sampled_data = natural_sarcastic.sample(n=samples_per_class, random_state=Config.RANDOM_SEED)
                
            df_balanced = pd.concat([df_balanced, sampled_data])
        
        return df_balanced.sample(frac=1, random_state=Config.RANDOM_SEED).reset_index(drop=True)
    
    
    #4 _process_split() - Processes each split
    def _process_split(self, df: pd.DataFrame, apply_augmentation: bool) -> pd.DataFrame:
        """
        Process individual data split with optional augmentation.
        
        Applies sarcasm augmentation to training data if specified,
        and generates polarity scores for all samples.
        
        Args:
            df: DataFrame split to process
            apply_augmentation: Whether to apply sarcasm augmentation
            
        Returns:
            Processed DataFrame with all required features
        """
        # Apply augmentation only to train set
        if apply_augmentation and Config.SARCASM_RATIO > 0:
            df = self.augmenter.create_balanced_sarcastic_dataset(df)
        
        # Generate final labels
        processed_data = []
        for _, row in df.iterrows():
            processed_row = {
                'text': row['text'],
                'processed_text': row['processed_text'],
                'sentiment': row['sentiment'],
                'is_sarcastic': row['is_sarcastic'],
                'polarity_score': self._calculate_polarity_score(row['processed_text'])
            }
            processed_data.append(processed_row)
        
        return pd.DataFrame(processed_data)
    

    #5 prepare_data() - Main orchestrator
    def prepare_data(self)->Tuple[np.ndarray, Dict, np.ndarray, Dict, np.ndarray, Dict]:
        """
        Prepare complete dataset through full processing pipeline.
        
        Orchestrates the entire data preparation process:
        1. Loads and preprocesses raw data
        2. Balances classes
        3. Splits data
        4. Processes each split
        5. Prepares model inputs
        
        Returns:
            Dict containing processed DataFrames and model inputs
        """
        print(f"Creating dataset with {Config.SAMPLES_PER_CLASS} samples per class")
    
        # 1. Load raw data
        raw_df = self.load_data()
        # 2. Balance classes first
        balanced_df = self._balance_classes(raw_df)
        
        # 3. Split first (before any balancing or augmentation)
        train_df, val_df, test_df = self.split_data(balanced_df)
        
        # 4. Process each split with appropriate augmentation
        train_processed = self._process_split(train_df, apply_augmentation=True)
        val_processed = self._process_split(val_df, apply_augmentation=False)
        test_processed = self._process_split(test_df, apply_augmentation=False)
        
        # 5. Prepare model inputs
        model_inputs = self._prepare_model_inputs(train_processed, val_processed, test_processed)
    
        return {
            'dataframes': {
                'train': train_processed,
                'val': val_processed,
                'test': test_processed
            },
            'model_inputs': model_inputs
        }
        

    #6 _prepare_model_inputs() - Converts to model format
    def _prepare_model_inputs(self, train_df, val_df, test_df)-> Tuple[np.ndarray, Dict, np.ndarray, Dict, np.ndarray, Dict]:
        """
        Convert processed DataFrames to model input format.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of numpy arrays and label dictionaries for each split
        """
        def prepare_split(df):
            return (
                df['processed_text'].to_numpy(),
                {
                    'sentiment': df['sentiment'].to_numpy(),
                    'sarcasm': df['is_sarcastic'].to_numpy(),
                    'polarity': df['polarity_score'].to_numpy()
                }
            )
        
        return (*prepare_split(train_df), 
                *prepare_split(val_df), 
                *prepare_split(test_df))
    

    #7 preprocess_text() - Text preprocessing utility
    def preprocess_text(self, text: str) -> str:
        """
        Apply full text preprocessing pipeline.
        
        Cleans URLs, expands contractions, normalizes case,
        and handles special tokens and emojis.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text ready for model
        """
        # Clean URLs
        text = TextSignals.clean_urls(text)
        processed_text = text.lower()
        
        # Expand contractions
        for contraction, expanded in TextSignals.CONTRACTION_MAP.items():
            processed_text = processed_text.replace(contraction, expanded)
        
        # Handle special tokens
        for token, replacement in TextSignals.SPECIAL_TOKENS.items():
            processed_text = processed_text.replace(token, replacement)
        
        return processed_text
    
    #8 _calculate_polarity_score
    def _calculate_polarity_score(self, text: str) -> float:
        """
        Calculate polarity score for text.
        
        Uses presence of positive/negative markers and
        'but' constructions to compute polarity.
        
        Args:
            text: Preprocessed text to analyze
            
        Returns:
            Polarity score between 0 and 1
        """        
        positive_markers = ['good', 'great', 'excellent', 'amazing', 'wonderful']
        negative_markers = ['bad', 'terrible', 'horrible', 'awful', 'poor']
        
        pos_count = sum(marker in text for marker in positive_markers)
        neg_count = sum(marker in text for marker in negative_markers)
        has_but = 'but' in text
        
        score = 0.7 if has_but else 0.3
        if pos_count > 0 and neg_count > 0:
            score += 0.3
            
        return min(score, 1.0)