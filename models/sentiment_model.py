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
        self.fusion_dense1 = tf.keras.layers.Dense(ModelConfig.FUSION_LAYERS[0], activation='relu')
        self.fusion_dense2 = tf.keras.layers.Dense(ModelConfig.FUSION_LAYERS[1], activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(ModelConfig.DROPOUT_RATES[0])
        self.dropout2 = tf.keras.layers.Dropout(ModelConfig.DROPOUT_RATES[1])
        self.classifier = tf.keras.layers.Dense(ModelConfig.NUM_CLASSES, activation='softmax')

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
        dropped = self.dropout2(fused, training=training)
        
        return self.classifier(dropped)

class ModelTrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def prepare_dataset(self, texts, labels):
        encodings = self.tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=ModelConfig.MAX_LENGTH,
            return_tensors='tf'
        )
        
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask']
            },
            labels
        ))
        
        return dataset.shuffle(10000).batch(ModelConfig.BATCH_SIZE)

    def train(self, train_dataset, val_dataset, epochs=5):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=2,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=1
            )
        ]
        
        return self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks
        )