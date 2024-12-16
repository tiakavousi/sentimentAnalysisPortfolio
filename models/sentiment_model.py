import tensorflow as tf
from transformers import TFDistilBertModel
from transformers import DistilBertTokenizer
from config.model_config import Config

class EnhancedDistilBertForSentiment(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.distilbert = TFDistilBertModel.from_pretrained(Config.BERT_MODEL)
        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(Config.LSTM_UNITS, return_sequences=True)
        )
        
        # Initialize Layer Normalization layers
        self.lstm_norm = tf.keras.layers.LayerNormalization()
        self.sarcasm_norm = tf.keras.layers.LayerNormalization()
        self.polarity_norm = tf.keras.layers.LayerNormalization()
        
        # Feature extraction layers
        self._init_feature_layers()
        
        # Feature fusion layers
        self._init_fusion_layers()
        
        # Final classifier
        self.final_classifier = tf.keras.layers.Dense(Config.NUM_CLASSES, activation='softmax', name='sentiment')

    def _init_feature_layers(self):
        """Initialize separate feature extraction layers"""
        # Sarcasm detection branch
        self.sarcasm_dense = tf.keras.Sequential([
            tf.keras.layers.Dense(Config.FEATURE_DIM, activation='relu'),
            tf.keras.layers.Dropout(Config.DROPOUT_RATES[0])
        ])
        self.sarcasm_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=Config.NUM_HEAD, key_dim=Config.FEATURE_DIM
        )
        
        
        # Polarity analysis branch
        self.polarity_dense = tf.keras.Sequential([
            tf.keras.layers.Dense(Config.FEATURE_DIM, activation='relu'),
            tf.keras.layers.Dropout(Config.DROPOUT_RATES[0])
        ])
        self.polarity_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=Config.NUM_HEAD, key_dim=Config.FEATURE_DIM
        )

    
    def _init_fusion_layers(self):
        self.fusion_dense1 = tf.keras.layers.Dense(Config.FUSION_LAYERS[0], activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(Config.DROPOUT_RATES[0])
        self.fusion_dense2 = tf.keras.layers.Dense(Config.FUSION_LAYERS[1], activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(Config.DROPOUT_RATES[1])

    def call(self, inputs, training=False):
        # BERT embeddings
        sequence_output = self.distilbert(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            training=training
        )[0]
        
        # LSTM processing
        lstm_output = self.lstm(sequence_output)
        lstm_output = self.lstm_norm(lstm_output)

        # Feature extraction with attention
        sarcasm_features = self.sarcasm_dense(lstm_output)
        sarcasm_attended = self.sarcasm_attention(
            sarcasm_features, sarcasm_features, sarcasm_features
        )
        sarcasm_attended = self.sarcasm_norm(sarcasm_attended)
        
        polarity_features = self.polarity_dense(lstm_output)
        polarity_attended = self.polarity_attention(
            polarity_features, polarity_features, polarity_features
        )
        polarity_attended = self.polarity_norm(polarity_attended)


        # Pool features
        sarcasm_pooled = tf.reduce_mean(sarcasm_attended, axis=1)
        polarity_pooled = tf.reduce_mean(polarity_attended, axis=1)
        
        # Concatenate features
        combined = tf.concat([
            sarcasm_pooled, 
            polarity_pooled
        ], axis=-1)
        
        # Feature fusion
        fused = self.fusion_dense1(combined)   # Transform data
        fused = self.dropout1(fused, training=training)  # Regularize first transformation
        fused = self.fusion_dense2(fused)      # Second transformation
        fused = self.dropout2(fused, training=training)  # Regularize second transformation
        
        # Final sentiment prediction
        sentiment = self.final_classifier(fused)
        
        return sentiment
    

class ModelTrainer:

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer or DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        if model is not None:
            self._compile_model()


    def _compile_model(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=self._create_learning_rate_schedule(),
                weight_decay=0.01,
                clipnorm=1.0
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    
    def _create_learning_rate_schedule(self):
        """Create learning rate schedule"""
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=Config.LEARNING_RATE,
            decay_steps=Config.DECAY_STEPS,
            decay_rate=Config.DECAY_RATE
        )
    
    
    def prepare_dataset(self, texts, labels=None):
        encoded = self.tokenizer(
            texts.tolist(),
            padding='max_length',
            truncation=True,
            max_length=Config.MAX_LENGTH,
            return_tensors='tf'
        )
        
        features = {
            'input_ids': tf.convert_to_tensor(encoded['input_ids'], dtype=tf.int32),
            'attention_mask': tf.convert_to_tensor(encoded['attention_mask'], dtype=tf.int32)
        }
        
        if labels is None:
            return features
            
        return tf.data.Dataset.from_tensor_slices((
            features, 
            labels['sentiment']
        )).batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


    def train(self, train_dataset, val_dataset):
        """Train the model"""
        # Create strategy for better error handling
        strategy = tf.distribute.get_strategy()
        
        with strategy.scope():
            history = self.model.fit(
                train_dataset,
                validation_data = val_dataset,
                epochs = Config.EPOCHS,
                callbacks = self.get_callbacks()
            )
            
        return history
    

    # factor=0.5, # patience=1,# min_lr=1e-6
    def get_callbacks(self):
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=Config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                min_delta=Config.EARLY_STOPPING_MIN_DELTA
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                # factor=0.2,  # More aggressive reduction (from 0.5)
                factor=0.5,
                patience=2,  # Increase from 1
                # min_lr=1e-7  # Lower minimum learning rate
                min_lr=5e-7 
            )
        ]