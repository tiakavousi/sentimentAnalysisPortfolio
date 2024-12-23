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
    SAMPLES_PER_CLASS = 1000
    SHUFFLE_BUFFER_SIZE = min(10000, SAMPLES_PER_CLASS * NUM_CLASSES)
    SARCASM_RATIO = 0.4


    # Base model settings
    BERT_MODEL = 'distilbert-base-uncased'
    MAX_LENGTH = 128  # Reduced from 192
    LSTM_UNITS = 96  # Reduced from 128
    FEATURE_DIM = 48  # Reduced from 64
    FUSION_LAYERS = [192, 96]  # Reduced from [256, 128]
    DROPOUT_RATES = [0.5, 0.6]
    NUM_HEAD = 4

    # Training settings
    BATCH_SIZE = 32  # Increased from 16 for better generalization
    LEARNING_RATE = 1e-5  # Reduced from 2e-5
    DECAY_STEPS = 1000
    DECAY_RATE = 0.95  # More gradual decay from 0.9
    EPOCHS = 10
    EARLY_STOPPING_PATIENCE = 2  # Reduced from 3
    EARLY_STOPPING_MIN_DELTA = 0.01  # Increased from 0.005
    LOSS = 'sparse_categorical_crossentropy'

    # Model evaluation
    METRICS = ['accuracy']
