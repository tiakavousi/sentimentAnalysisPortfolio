import tensorflow as tf
from transformers import TFDistilBertModel
from config.model_config import ModelConfig

class EnhancedDistilBertForSentiment(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.distilbert = TFDistilBertModel.from_pretrained(ModelConfig.BERT_MODEL)
        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(ModelConfig.LSTM_UNITS, return_sequences=True)
        )
        self.attention_query = tf.keras.layers.Dense(ModelConfig.ATTENTION_DIM)
        self.attention_key = tf.keras.layers.Dense(ModelConfig.ATTENTION_DIM)
        self.attention_value = tf.keras.layers.Dense(ModelConfig.ATTENTION_DIM)
        
        # Shared layers
        self.fusion_dense1 = tf.keras.layers.Dense(ModelConfig.FUSION_LAYERS[0], activation='relu')
        self.fusion_dense2 = tf.keras.layers.Dense(ModelConfig.FUSION_LAYERS[1], activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(ModelConfig.DROPOUT_RATES[0])
        self.dropout2 = tf.keras.layers.Dropout(ModelConfig.DROPOUT_RATES[1])
        
        # Output layers for different tasks
        self.sentiment_classifier = tf.keras.layers.Dense(ModelConfig.NUM_CLASSES, activation='softmax', name='sentiment')
        self.sarcasm_detector = tf.keras.layers.Dense(1, activation='sigmoid', name='sarcasm')
        self.negation_detector = tf.keras.layers.Dense(1, activation='sigmoid', name='negation')
        self.polarity_scorer = tf.keras.layers.Dense(1, name='polarity')

    def attention(self, query, key, value, training=False):
        attention_weights = tf.matmul(query, key, transpose_b=True)
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)
        return tf.matmul(attention_weights, value)
    

    def call(self, inputs, training=False):
        distilbert_outputs = self.distilbert(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            training=training
        )
        sequence_output = distilbert_outputs[0]
        lstm_output = self.lstm(sequence_output)
        lstm_output = self.dropout1(lstm_output, training=training)
        
        query = self.attention_query(lstm_output)
        key = self.attention_key(lstm_output)
        value = self.attention_value(lstm_output)
        attention_output = self.attention(query, key, value, training)
        
        pooled = tf.reduce_mean(attention_output, axis=1)
        fused = self.fusion_dense1(pooled)
        fused = self.fusion_dense2(fused)
        shared_features = self.dropout2(fused, training=training)
       
        # Multiple outputs
        return {
            'sentiment': self.sentiment_classifier(shared_features),
            'sarcasm': self.sarcasm_detector(shared_features),
            'negation': self.negation_detector(shared_features),
            'polarity': self.polarity_scorer(shared_features)
        }



class ModelTrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def prepare_dataset(self, texts, labels):
        """Prepare dataset for training"""
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
        
        # Prepare labels with explicit tensor conversion and reshape
        labels_dict = {
            'sentiment': tf.convert_to_tensor(labels['sentiment'].values, dtype=tf.int32),
            'sarcasm': tf.cast(tf.convert_to_tensor(labels['sarcasm'].values), dtype=tf.float32),
            'negation': tf.cast(tf.convert_to_tensor(labels['negation'].values), dtype=tf.float32),
            'polarity': tf.cast(tf.convert_to_tensor(labels['polarity'].values), dtype=tf.float32)
        }
        
        # Reshape tensors to have proper dimensions
        labels_dict['sarcasm'] = tf.reshape(labels_dict['sarcasm'], (-1, 1))
        labels_dict['negation'] = tf.reshape(labels_dict['negation'], (-1, 1))
        labels_dict['polarity'] = tf.reshape(labels_dict['polarity'], (-1, 1))
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((features, labels_dict))
        
        # Apply shuffling, batching, and prefetching
        return (dataset
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
            # Added checkpoint callback
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            )
        ]