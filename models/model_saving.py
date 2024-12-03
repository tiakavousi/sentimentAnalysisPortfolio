
import tensorflow as tf
import os

model_dir = './saved_models/'
os.makedirs(model_dir, exist_ok=True)

def save_model(model, optimizer, epoch, loss, model_dir='./saved_models/'):
    model_path = f"{model_dir}model_epoch{epoch}"
    model.save(model_path)

    checkpoint = {
        'epoch': epoch,
        'model_path': model_path,
        'optimizer_state_dict': optimizer.get_weights(),
        'loss': loss
    }
    tf.keras.models.save_model(model.data_processor, f"{model_dir}data_processor_epoch{epoch}")
    tf.keras.models.save_model(model.data_processor.sarcasm_detector, f"{model_dir}sarcasm_detector_epoch{epoch}")
    tf.io.write_file(f"{model_dir}checkpoint_epoch{epoch}.json", tf.io.serialize_tensor(checkpoint))
    
    print(f"Model saved for epoch {epoch}")

def load_model(model_dir, epoch):
    checkpoint = tf.io.read_file(f"{model_dir}checkpoint_epoch{epoch}.json")
    checkpoint = tf.io.parse_tensor(checkpoint, tf.string)
    checkpoint = tf.keras.layers.deserialize(checkpoint)

    model = tf.keras.models.load_model(checkpoint['model_path'])
    model.data_processor = tf.keras.models.load_model(f"{model_dir}data_processor_epoch{epoch}")
    model.data_processor.sarcasm_detector = tf.keras.models.load_model(f"{model_dir}sarcasm_detector_epoch{epoch}")

    optimizer.set_weights(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] 
    loss = checkpoint['loss']
    
    return model, optimizer, epoch, loss
