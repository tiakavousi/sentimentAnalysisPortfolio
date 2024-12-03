class ModelConfig:
    BERT_MODEL = 'distilbert-base-uncased'
    MAX_LENGTH = 200
    LSTM_UNITS = 156
    ATTENTION_DIM = 96
    FUSION_LAYERS = [320, 160]
    DROPOUT_RATES = [0.25, 0.35]
    NUM_CLASSES = 3
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-5
    DECAY_STEPS = 1000
    DECAY_RATE = 0.9
    EPOCHS = 5
    SHUFFLE_BUFFER_SIZE = 10000
    SAMPLES_PER_CLASS = 2000
    # CHUNK_SIZE = 1000

    VALIDATION_SPLIT = 0.1  # 10% for validation
    EARLY_STOPPING_PATIENCE = 2  # Stop if no improvement for 3 epochs

    SPECIAL_TOKENS = {
        'SARC': '_SARC_',
        'NEG': '_NEG_',
        'MULTI_EXCLAIM': 'MULTI_EXCLAIM',
        'ELLIPSIS': 'ELLIPSIS',
        'MULTI_QUESTION': 'MULTI_QUESTION'
    }
    
    # Output dimensions for different heads
    SENTIMENT_CLASSES = 3  # positive, neutral, negative
    AUXILIARY_OUTPUTS = {
        'sarcasm': 1,      # binary classification
        'negation': 1,     # binary classification
        'polarity': 1      # continuous score for multipolarity
    }