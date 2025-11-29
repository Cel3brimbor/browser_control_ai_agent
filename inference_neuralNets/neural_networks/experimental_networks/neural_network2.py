import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore

TOTAL_SAMPLES = 1_000_000
NUM_FEATURES = 100
labels = ['on_task', 'off_task']

X = np.random.rand(TOTAL_SAMPLES, NUM_FEATURES).astype(np.float32)
y_text = np.random.choice(labels, TOTAL_SAMPLES)
encoder = LabelEncoder()
encoder.fit(labels)
y = encoder.transform(y_text).astype(np.int64)

#split into train and test sets (90% for train, 10% for test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices being used: {strategy.num_replicas_in_sync}")

with strategy.scope():
    model = Sequential([
        Dense(1024, activation='relu', input_shape=(NUM_FEATURES,)),
        Dropout(0.3),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

model.summary()

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
BATCH_SIZE = 256 * strategy.num_replicas_in_sync
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("\nTraining the model...")
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=test_dataset
)

print("\nEvaluating the model.")
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {accuracy:.4f}")

while True:
    input_str = input("Input: ")
    if input_str.lower() == 'quit':
        break
    try:
        features = np.array([float(x.strip()) for x in input_str.split(',')]).astype(np.float32)
        if len(features) != NUM_FEATURES:
            print(f"Error: Must provide exactly {NUM_FEATURES} features.")
            continue
        pred = model.predict(features.reshape(1, -1))[0][0]
        label = encoder.inverse_transform([1 if pred > 0.5 else 0])[0]
        print(f"Predicted: {label} (probability: {pred:.4f})")
    except ValueError:
        print("Error: Invalid input. Please enter comma-separated floats.")