import tensorflow as tf
from transformers import TFDistilBertModel
from transformers import DistilBertTokenizer
from config.model_config import ModelConfig

class EnhancedDistilBertForSentiment(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.distilbert = TFDistilBertModel.from_pretrained(ModelConfig.BERT_MODEL)
        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(ModelConfig.LSTM_UNITS, return_sequences=True)
        )
        
        # Feature extraction layers
        self._init_feature_layers()
        
        # Feature fusion layers
        self._init_fusion_layers()
        
        # Final classifier
        self.final_classifier = tf.keras.layers.Dense(ModelConfig.NUM_CLASSES, activation='softmax', name='sentiment')

    def _init_feature_layers(self):
        """Initialize separate feature extraction layers"""
        # Sarcasm detection branch
        self.sarcasm_dense = tf.keras.layers.Dense(ModelConfig.FEATURE_DIM, activation='relu')
        self.sarcasm_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=ModelConfig.FEATURE_DIM
        )
        
        # Negation detection branch
        self.negation_dense = tf.keras.layers.Dense(ModelConfig.FEATURE_DIM, activation='relu')
        self.negation_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=ModelConfig.FEATURE_DIM
        )
        
        # Polarity analysis branch
        self.polarity_dense = tf.keras.layers.Dense(ModelConfig.FEATURE_DIM, activation='relu')
        self.polarity_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=ModelConfig.FEATURE_DIM
        )

    def _init_fusion_layers(self):
        """Initialize feature fusion layers"""
        self.fusion_dense1 = tf.keras.layers.Dense(ModelConfig.FUSION_LAYERS[0], activation='relu')
        self.fusion_dense2 = tf.keras.layers.Dense(ModelConfig.FUSION_LAYERS[1], activation='relu')
        self.dropout = tf.keras.layers.Dropout(ModelConfig.DROPOUT_RATES[0])

    def call(self, inputs, training=False):
        # BERT embeddings
        sequence_output = self.distilbert(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            training=training
        )[0]
        
        # LSTM processing
        lstm_output = self.lstm(sequence_output)
        
        # Feature extraction with attention
        sarcasm_features = self.sarcasm_dense(lstm_output)
        sarcasm_attended = self.sarcasm_attention(
            sarcasm_features, sarcasm_features, sarcasm_features
        )
        
        negation_features = self.negation_dense(lstm_output)
        negation_attended = self.negation_attention(
            negation_features, negation_features, negation_features
        )
        
        polarity_features = self.polarity_dense(lstm_output)
        polarity_attended = self.polarity_attention(
            polarity_features, polarity_features, polarity_features
        )
        
        # Pool features
        sarcasm_pooled = tf.reduce_mean(sarcasm_attended, axis=1)
        negation_pooled = tf.reduce_mean(negation_attended, axis=1)
        polarity_pooled = tf.reduce_mean(polarity_attended, axis=1)
        
        # Concatenate features
        combined = tf.concat([
            sarcasm_pooled, 
            negation_pooled, 
            polarity_pooled
        ], axis=-1)
        
        # Feature fusion
        fused = self.fusion_dense1(combined)
        fused = self.dropout(fused, training=training)
        fused = self.fusion_dense2(fused)
        
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
            initial_learning_rate=ModelConfig.LEARNING_RATE,
            decay_steps=ModelConfig.DECAY_STEPS,
            decay_rate=ModelConfig.DECAY_RATE
        )
    
    
    def prepare_dataset(self, texts, labels=None):
        encoded = self.tokenizer(
            texts.tolist(),
            padding='max_length',
            truncation=True,
            max_length=ModelConfig.MAX_LENGTH,
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
        )).batch(ModelConfig.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


    def train(self, train_dataset, val_dataset):
        """Train the model"""
        # Create strategy for better error handling
        strategy = tf.distribute.get_strategy()
        
        with strategy.scope():
            history = self.model.fit(
                train_dataset,
                validation_data = val_dataset,
                epochs = ModelConfig.EPOCHS,
                callbacks = self.get_callbacks()
            )
            
        return history
    

    # factor=0.5, # patience=1,# min_lr=1e-6
    def get_callbacks(self):
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=ModelConfig.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                min_delta=ModelConfig.EARLY_STOPPING_MIN_DELTA
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,  # More aggressive reduction (from 0.5)
                patience=2,  # Increase from 1
                min_lr=1e-7  # Lower minimum learning rate
            )
        ]