import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

num_samples = 10000
num_features = 10
X = np.random.rand(num_samples, num_features).astype(np.float32)

labels = ['shopping', 'social', 'news', 'malicious']
y_text = np.random.choice(labels, num_samples)

encoder = LabelEncoder()
y_numerical = encoder.fit_transform(y_text).astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_numerical, test_size=0.2, random_state=42
)

print(f"Number of training samples: {len(X_train)}")
print(f"Number of testing samples: {len(X_test)}")

strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices being used: {strategy.num_replicas_in_sync}")

with strategy.scope():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(num_features,)),
        Dense(32, activation='relu'),
        Dense(len(labels), activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

model.summary()


BATCH_SIZE = 32 

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

train_dataset = train_dataset.shuffle(buffer_size=10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=test_dataset
)


print("\nEvaluating the model on the test data...")
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {accuracy:.4f}")


new_data_point = np.random.rand(1, num_features).astype(np.float32)
prediction_probabilities = model.predict(new_data_point)

predicted_label_index = np.argmax(prediction_probabilities)
predicted_label = encoder.inverse_transform([predicted_label_index])[0]

print(f"\nNew data point: {new_data_point}")
print(f"Prediction Probabilities: {prediction_probabilities}")
print(f"The model predicts the category is: {predicted_label}")