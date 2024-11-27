from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self):
        self.contraction_map = {
            "can't": "cannot",
            "won't": "will not",
            "n't": " not",
            "'re": " are",
            "'s": " is",
            # ... (rest of contractions)
        }
        self.special_tokens = {
            '!!!': ' MULTI_EXCLAIM ',
            '...': ' ELLIPSIS ',
            '??': ' MULTI_QUESTION ',
            # ... (rest of special tokens)
        }
        self.negation_words = ['not', 'never', "n't", 'no', 'neither', 'nor']
        self.sarcasm_markers = ['obviously', 'clearly', 'surely', 'right', 'sure']

    def load_data(self, samples_per_class=2000):
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
        return train_test_split(
            df['text'], 
            df['sentiment'],
            test_size=test_size,
            random_state=42,
            stratify=df['sentiment']
        )

    def preprocess_text(self, text):
        for contraction, expanded in self.contraction_map.items():
            text = text.replace(contraction, expanded)
        
        for token, replacement in self.special_tokens.items():
            text = text.replace(token, replacement)
        
        for word in self.negation_words:
            text = text.replace(f'{word} ', f'{word}_NEG ')
        
        for marker in self.sarcasm_markers:
            text = text.replace(f'{marker} ', f'{marker}_SARC ')
        
        return text