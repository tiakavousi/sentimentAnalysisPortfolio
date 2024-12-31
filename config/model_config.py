from dataclasses import dataclass

@dataclass
class Config:
    # Dataset settings
    YELP_DATASET = "yelp_review_full"
    RANDOM_SEED: int = 42
    TRAIN_SPLIT: float = 0.7
    VALIDATION_SPLIT: float = 0.15
    TEST_SPLIT: float = 0.15
    NUM_CLASSES = 3
    SAMPLES_PER_CLASS = 1500
    SHUFFLE_BUFFER_SIZE = min(10000, SAMPLES_PER_CLASS * NUM_CLASSES)
    SARCASM_RATIO = 0.4
    LABEL_MAP = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}


    # Base model settings
    BERT_MODEL = 'distilbert-base-uncased'
    MAX_LENGTH = 192
    LSTM_UNITS = 128 
    FEATURE_DIM = 64 
    FUSION_LAYERS = [192, 96]  # Reduced from [256, 128]
    DROPOUT_RATES = [0.5, 0.6]
    NUM_HEAD = 4

    # Training settings
    BATCH_SIZE = 32  # Increased from 16 for better generalization
    LEARNING_RATE = 2e-5
    DECAY_RATE = 0.97 
    WARMUP_STEPS = 500
    DECAY_STEPS = 3000  
    EPOCHS = 15
    EARLY_STOPPING_PATIENCE = 3 
    EARLY_STOPPING_MIN_DELTA = 0.01  
    LOSS = 'sparse_categorical_crossentropy'

    # Model evaluation
    METRICS = ['accuracy']
