import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, TextVectorization, Embedding, GlobalAveragePooling1D, Concatenate, Layer #type: ignore
from tensorflow.keras.models import Model #type: ignore
import keras.ops as K
import numpy as np
import json
from sklearn.model_selection import train_test_split
import os

# List of dataset paths
dataset_paths = [
    #"/Users/norranyu/Documents/coding/x_code/Durin/inference_engine/datasets/variety.json",
    "/Users/norranyu/Documents/coding/x_code/Durin/inference_engine/datasets/physics/physics_dataset.json",
]

DROPOUT = 0.3

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# Check for GPU
physical_devices = tf.config.list_physical_devices('GPU')
print("\nPhysical devices:", physical_devices)

if physical_devices:
    print("\nGPU is detected. TensorFlow is now ready to use Metal acceleration.\n")
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Failed to set memory growth: {e}")
else:
    print("\nGPU is not yet detected\n")

class ReshapeForEmbedding(Layer):
    """Custom layer to reshape flattened vectorized output back to (batch_size, sequence_length)."""
    def call(self, inputs, ragged_lengths):
        batch_size = K.shape(ragged_lengths)[0]
        sequence_length = K.cast(K.max(ragged_lengths), dtype='int32')
        inputs = K.reshape(inputs, [-1])
        #tf.print("inputs shape after reshape:", K.shape(inputs))
        total_elements = K.sum(ragged_lengths)
        #tf.print("ragged_lengths:", ragged_lengths)
        #tf.print("batch_size:", batch_size, "sequence_length:", sequence_length)
        #tf.print("total_elements in inputs:", total_elements)
        expected_elements = batch_size * sequence_length
        inputs_padded = K.pad(inputs, [[0, expected_elements - K.shape(inputs)[0]]], mode='constant', constant_values=0)
        #tf.print("inputs_padded shape:", K.shape(inputs_padded))
        return K.reshape(inputs_padded, [batch_size, sequence_length])

    def compute_output_shape(self, input_shape):
        return (None, None)

class FlattenRaggedLayer(Layer):
    """Custom layer to flatten a ragged tensor to a 1D tensor."""
    def call(self, inputs):
        dense_tensor = inputs.to_tensor()
        return K.reshape(dense_tensor, [-1])

    def compute_output_shape(self, input_shape):
        return (None,)

class ComputeRaggedLengthsLayer(Layer):
    """Custom layer to compute the lengths of sequences in a ragged tensor"""
    def call(self, inputs):
        return K.sum(K.ones_like(inputs, dtype='int32'), axis=1)

    def compute_output_shape(self, input_shape):
        return (None,)

def build_model(vectorize_layer, embedding_dim=128):
    """
    Builds a neural network to determine if a new tab aligns with opened tabs.
    """
    #input layers
    opened_tabs_input = Input(
        shape=(None,),
        dtype=tf.string,
        name='opened_tabs_input',
        ragged=True
    )
    new_tab_input = Input(
        shape=(1,),
        dtype=tf.string,
        name='new_tab_input'
    )

    #proces opened tabs
    ragged_lengths = ComputeRaggedLengthsLayer(name='compute_ragged_lengths')(opened_tabs_input)
    ragged_lengths = tf.keras.layers.Lambda(
        lambda x: x,
        name='debug_ragged_lengths',
        output_shape=(None,)
    )(ragged_lengths)

    opened_tabs_flat = FlattenRaggedLayer(name='flatten_ragged')(opened_tabs_input)
    opened_tabs_flat = tf.keras.layers.Lambda(
        lambda x: x,
        name='debug_flattened_tabs',
        output_shape=(None,)
    )(opened_tabs_flat)

    opened_tabs_encoded = vectorize_layer(opened_tabs_flat)
    tf.print("opened_tabs_encoded shape:", K.shape(opened_tabs_encoded))
    opened_tabs_encoded = ReshapeForEmbedding()(opened_tabs_encoded, ragged_lengths)

    opened_tabs_embedded = Embedding(
        input_dim=len(vectorize_layer.get_vocabulary()),
        output_dim=embedding_dim,
        name='opened_tabs_embedding'
    )(opened_tabs_encoded)
    opened_tabs_pooled = GlobalAveragePooling1D(name='opened_tabs_pooling')(opened_tabs_embedded)

    #process new tab
    new_tab_encoded = vectorize_layer(new_tab_input)
    new_tab_embedded = Embedding(
        input_dim=len(vectorize_layer.get_vocabulary()),
        output_dim=embedding_dim,
        name='new_tab_embedding'
    )(new_tab_encoded)
    new_tab_pooled = GlobalAveragePooling1D(name='new_tab_pooling')(new_tab_embedded)

    concatenated = Concatenate(name='concatenate_embeddings')([opened_tabs_pooled, new_tab_pooled])
    
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), name='dense_256')(concatenated)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), name='dense_128')(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), name='dense_64')(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), name='dense_32')(x)
    output = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(
        inputs=[opened_tabs_input, new_tab_input],
        outputs=output,
        name='tab_alignment_model'
    )
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model

def load_and_validate_dataset(dataset_path):
    """Load and validate a single dataset."""
    try:
        with open(dataset_path, 'r') as f:
            loaded_data = json.load(f)
        opened_tabs_list = [item["opened_tabs"] for item in loaded_data["data"]]
        new_tab_list = [item["new_tab"] for item in loaded_data["data"]]
        labels = [item["alignment"] for item in loaded_data["data"]]
        
        for i, tabs in enumerate(opened_tabs_list):
            if not tabs or not isinstance(tabs, list) or any(not isinstance(t, str) or t.strip() == '' for t in tabs):
                print(f"Invalid opened_tabs at index {i} in {dataset_path}: {tabs}")
                return None, None, None
        for i, tab in enumerate(new_tab_list):
            if not isinstance(tab, str) or tab.strip() == '':
                print(f"Invalid new_tab at index {i} in {dataset_path}: {tab}")
                return None, None, None
        
        opened_tabs_array = [np.array(x, dtype=str) for x in opened_tabs_list]
        new_tab_array = np.array(new_tab_list, dtype=str)
        y = np.array(labels, dtype=np.int32)
        
        print(f"Dataset {dataset_path} loaded successfully. Samples: opened_tabs={len(opened_tabs_list)}, new_tab={len(new_tab_list)}, labels={len(labels)}")
        return opened_tabs_array, new_tab_array, y
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{dataset_path}'. Skipping.")
        return None, None, None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{dataset_path}'. Skipping.")
        return None, None, None
    except KeyError:
        print(f"Error: JSON data in '{dataset_path}' is missing 'opened_tabs', 'new_tab', or 'alignment' keys. Skipping.")
        return None, None, None

def main():
    def custom_standardize(input_data):
        return tf.strings.lower(input_data)

    vectorize_layer = TextVectorization(
        max_tokens=50000,
        #max_tokens=10000000,
        output_mode='int',
        standardize=custom_standardize,
        split=None
        #split = 'whitespace'
    )

    #load and validate datasets
    all_training_text = []
    all_datasets = []
    for dataset_path in dataset_paths:
        opened_tabs_array, new_tab_array, y = load_and_validate_dataset(dataset_path)
        if opened_tabs_array is None:
            continue
        print(f"Dataset {dataset_path} has {len(opened_tabs_array)} examples")
        print("Alignment distribution:", np.bincount(y))
        print("Tab count distribution:", np.bincount([len(tabs) for tabs in opened_tabs_array])[2:10])
        all_datasets.append((opened_tabs_array, new_tab_array, y))
        all_training_text.extend(np.concatenate([np.array(tabs, dtype=str) for tabs in opened_tabs_array]))
        all_training_text.extend(new_tab_array)

    if not all_datasets:
        print("No valid datasets were loaded. Exiting.")
        return

    vectorize_layer.adapt(np.array(all_training_text, dtype=str))
    print("Vocabulary size:", len(vectorize_layer.get_vocabulary()))
    print("Sample vocabulary:", vectorize_layer.get_vocabulary()[:10])

    #build
    model = build_model(vectorize_layer)

    for idx, (opened_tabs_array, new_tab_array, y) in enumerate(all_datasets):
        print(f"\nProcessing dataset {idx + 1}/{len(all_datasets)}: {dataset_paths[idx]}")
        
        #split data
        X_train_tabs, X_test_tabs, X_train_new, X_test_new, y_train, y_test = train_test_split(
            opened_tabs_array, new_tab_array, y, test_size=0.2, random_state=42)

        #convert to tensors
        X_train_tabs_ragged = tf.ragged.constant(X_train_tabs, dtype=tf.string)
        X_test_tabs_ragged = tf.ragged.constant(X_test_tabs, dtype=tf.string)
        X_train_new = tf.convert_to_tensor(X_train_new[:, None], dtype=tf.string)
        X_test_new = tf.convert_to_tensor(X_test_new[:, None], dtype=tf.string)

        #train model
        history = model.fit(
            {'opened_tabs_input': X_train_tabs_ragged, 'new_tab_input': X_train_new},
            y_train,
            epochs=30,
            batch_size=16,
            validation_split=0.2,
            validation_batch_size=16,
            verbose=1,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=30, monitor='val_loss', restore_best_weights=True)]
        )
        
        print("Training history:", history.history)

        # evaluate model
        loss, accuracy = model.evaluate(
            {'opened_tabs_input': X_test_tabs_ragged, 'new_tab_input': X_test_new},
            y_test,
            verbose=0
        )
        print(f"Model Accuracy on test data: {accuracy:.2f}")

    while True:
        try:
            model.summary()
            _ = input("Press enter to proceed with a prediction: ")
            opened_tabs_str = input("Enter opened tabs (comma-separated): ")
            opened_tabs = opened_tabs_str.split(',')
            new_tab = input("Enter new tab: ")

            inputs = {
                'opened_tabs_input': tf.ragged.constant([opened_tabs], dtype=tf.string),
                'new_tab_input': tf.convert_to_tensor([[new_tab]], dtype=tf.string)
            }
            
            determination = model.predict(inputs, verbose=0)[0][0]
            print(f"\nAlignment probability: {determination:.2f}")
            if determination > 0.5:
                print("The new tab likely aligns with the current topic.")
            else:
                print("The new tab likely does not align with the current topic.")
            _ = input("\n\nPress Enter to proceed: ")
            
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

if __name__ == "__main__":
    main()