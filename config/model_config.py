class ModelConfig:
    # Base model settings
    BERT_MODEL = 'distilbert-base-uncased'
    MAX_LENGTH = 256 
    # training stopped at epoch 2, overfitting problem
    # LSTM_UNITS = 156
    # FEATURE_DIM = 64
    # FUSION_LAYERS = [320, 160]
    # DROPOUT_RATES = [0.25, 0.35]

    LSTM_UNITS = 64
    FEATURE_DIM = 32
    FUSION_LAYERS = [160, 80]
    DROPOUT_RATES = [0.4, 0.5] 
    NUM_CLASSES = 3
    
    # Training settings
    BATCH_SIZE = 32
    # LEARNING_RATE = 1e-5
    LEARNING_RATE = 5e-6
    DECAY_STEPS = 2000
    DECAY_RATE = 0.9
    EPOCHS = 10
    SHUFFLE_BUFFER_SIZE = 5000
    SAMPLES_PER_CLASS = 1000
    VALIDATION_SPLIT = 0.1
    # EARLY_STOPPING_PATIENCE = 2
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_MIN_DELTA = 0.01
    
    # Model evaluation
    LOSS = 'sparse_categorical_crossentropy'
    METRICS = ['accuracy']
    
    # Dataset paths
    YELP_DATASET = "yelp_review_full"
    SARC_DATASET = "sarcasm_dataset" 