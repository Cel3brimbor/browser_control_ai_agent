import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

# --- Part 1: Mock Data Creation ---
num_samples = 10000
num_features = 10
X = np.random.rand(num_samples, num_features).astype(np.float32)

labels = ['shopping', 'social', 'news', 'malicious']
y_text = np.random.choice(labels, num_samples)

# --- Part 2: Data Preprocessing ---
# Convert text labels to numerical format (0, 1, 2, 3)
encoder = LabelEncoder()
y_numerical = encoder.fit_transform(y_text).astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_numerical, test_size=0.2, random_state=42
)

print(f"Number of training samples: {len(X_train)}")
print(f"Number of testing samples: {len(X_test)}")


# --- Part 3: Building the Neural Network Model ---
# We use a distribution strategy to utilize all available GPUs.
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices being used: {strategy.num_replicas_in_sync}")

# The model must be created within the strategy scope.
with strategy.scope():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(num_features,)),
        Dense(32, activation='relu'),
        Dense(len(labels), activation='softmax')
    ])

    # --- Part 4: Compiling the Model ---
    # Configure the model for training with an optimizer, a loss function, and metrics.
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

model.summary()


# --- Part 5: Creating the Optimized Data Pipeline ---
# This is the key change to use all resources.
# tf.data.Dataset is highly efficient for large datasets.
# We set a batch size, which determines how many samples are processed at once.
BATCH_SIZE = 32 

# Create a tf.data.Dataset from your numpy arrays.
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Shuffle, batch, and prefetch the training data for optimal performance.
# Prefetching ensures the next batch is ready while the GPU processes the current one.
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Batch and prefetch the test data. Shuffling is not necessary for evaluation.
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# --- Part 6: Training the Model ---
print("\nTraining the model...")
# The model now trains on the optimized dataset pipeline.
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=test_dataset
)


# --- Part 7: Evaluating the Model's Performance ---
print("\nEvaluating the model on the test data...")
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {accuracy:.4f}")


# --- Part 8: Making a Prediction ---
# Let's predict a single, new data point.
new_data_point = np.random.rand(1, num_features).astype(np.float32)
prediction_probabilities = model.predict(new_data_point)

# Get the index of the highest probability
predicted_label_index = np.argmax(prediction_probabilities)
# Get the corresponding text label
predicted_label = encoder.inverse_transform([predicted_label_index])[0]

print(f"\nNew data point: {new_data_point}")
print(f"Prediction Probabilities: {prediction_probabilities}")
print(f"The model predicts the category is: {predicted_label}")


# --- Part 9: Saving the Model ---
# After training, you can save the model for later use.
# This creates a file containing the trained weights and architecture.
# model.save('browser_monitor_model.h5')
# print("\nModel saved as 'browser_monitor_model.h5'")