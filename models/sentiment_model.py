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
        
        # Combine attention layers into a single block for clarity
        self._init_attention_layers()
        
        # Initialize shared layers
        self._init_shared_layers()
        
        # Initialize task-specific layers
        self._init_task_layers()
  
    def _init_attention_layers(self):
        """Initialize attention mechanism layers"""
        self.attention_query = tf.keras.layers.Dense(ModelConfig.ATTENTION_DIM)
        self.attention_key = tf.keras.layers.Dense(ModelConfig.ATTENTION_DIM)
        self.attention_value = tf.keras.layers.Dense(ModelConfig.ATTENTION_DIM)

    def _init_shared_layers(self):
        """Initialize shared processing layers"""
        self.fusion_dense1 = tf.keras.layers.Dense(ModelConfig.FUSION_LAYERS[0], activation='relu')
        self.fusion_dense2 = tf.keras.layers.Dense(ModelConfig.FUSION_LAYERS[1], activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(ModelConfig.DROPOUT_RATES[0])
        self.dropout2 = tf.keras.layers.Dropout(ModelConfig.DROPOUT_RATES[1])

    def _init_task_layers(self):
        """Initialize task-specific output layers"""
        self.sentiment_classifier = tf.keras.layers.Dense(ModelConfig.NUM_CLASSES, activation='softmax', name='sentiment')
        self.sarcasm_detector = tf.keras.layers.Dense(1, activation='sigmoid', name='sarcasm')
        self.negation_detector = tf.keras.layers.Dense(1, activation='sigmoid', name='negation')
        self.polarity_scorer = tf.keras.layers.Dense(1, name='polarity')


    def attention(self, query, key, value, training=False):
        """Compute attention scores and weighted sum"""
        attention_weights = tf.matmul(query, key, transpose_b=True)
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)
        return tf.matmul(attention_weights, value)
    
    def call(self, inputs, training=False):
        # Get BERT embeddings
        sequence_output = self.distilbert(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            training=training
        )[0]
        
        # Process sequence with LSTM
        lstm_output = self.lstm(sequence_output)
        lstm_output = self.dropout1(lstm_output, training=training)
        
        # Compute attention
        query = self.attention_query(lstm_output)
        key = self.attention_key(lstm_output)
        value = self.attention_value(lstm_output)
        attention_output = self.attention(query, key, value, training)
        
        # Global pooling and shared features
        pooled = tf.reduce_mean(attention_output, axis=1)
        fused = self.fusion_dense1(pooled)
        fused = self.fusion_dense2(fused)
        shared_features = self.dropout2(fused, training=training)

        # Task-specific outputs
        return {
            'sentiment': self.sentiment_classifier(shared_features),
            'sarcasm': self.sarcasm_detector(shared_features),
            'negation': self.negation_detector(shared_features),
            'polarity': self.polarity_scorer(shared_features)
        }


class ModelTrainer:

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer or DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        if model is not None:
            self._compile_model()

    def _compile_model(self):
        """Compile model with optimizer and loss functions"""
        if self.model is None:
            raise ValueError("Cannot compile None model")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self._create_learning_rate_schedule()
            ),
            loss=ModelConfig.LOSSES,
            metrics=ModelConfig.METRICS,
            loss_weights=ModelConfig.LOSS_WEIGHTS
        )

    def _create_learning_rate_schedule(self):
        """Create learning rate schedule"""
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=ModelConfig.LEARNING_RATE,
            decay_steps=ModelConfig.DECAY_STEPS,
            decay_rate=ModelConfig.DECAY_RATE
        )
    

    def prepare_dataset(self, texts, labels=None):
        """Prepare dataset with efficient preprocessing"""

        # Tokenize texts
        encoded = self.tokenizer(
            texts.tolist(),
            padding='max_length',
            truncation=True,
            max_length=ModelConfig.MAX_LENGTH,
            return_tensors='tf'
        )
        
        # Prepare features
        features = {
            'input_ids': tf.convert_to_tensor(encoded['input_ids'], dtype=tf.int32),
            'attention_mask': tf.convert_to_tensor(encoded['attention_mask'], dtype=tf.int32)
        }
        if labels is None:
            return features
        # Prepare labels with proper shapes
        labels_dict = {
            'sentiment': tf.convert_to_tensor(labels['sentiment'], dtype=tf.int32),
            'sarcasm': tf.reshape(tf.cast(labels['sarcasm'], tf.float32), (-1, 1)),
            'negation': tf.reshape(tf.cast(labels['negation'], tf.float32), (-1, 1)),
            'polarity': tf.reshape(tf.cast(labels['polarity'], tf.float32), (-1, 1))
        }
        
        # Create and optimize dataset
        return (tf.data.Dataset.from_tensor_slices((features, labels_dict))
                .cache()
                .shuffle(ModelConfig.SHUFFLE_BUFFER_SIZE)
                .batch(ModelConfig.BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))
    
    
    def train(self, train_dataset, val_dataset):
        """Train the model"""
        # Create strategy for better error handling
        strategy = tf.distribute.get_strategy()
        
        with strategy.scope():
            history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=ModelConfig.EPOCHS,
                callbacks=self.get_callbacks()
            )
            
        return history
    
    def get_callbacks(self):
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=ModelConfig.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=1,
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            )
        ]