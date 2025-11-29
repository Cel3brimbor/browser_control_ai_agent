import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Input # type: ignore
import numpy as np
import json
from sklearn.model_selection import train_test_split
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# check for GPU
physical_devices = tf.config.list_physical_devices('GPU')
print("\nPhysical devices:", physical_devices)

if physical_devices:
    print("\nGPU is detected. TensorFlow is now ready to use Metal acceleration.\n")
else:
    print("\nGPU is not yet detected\n")

# memory growth for gpu 
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Failed to set memory growth: {e}")

dataset = "/Users/norranyu/Documents/coding/x_code/Durin/inference_engine/neural_networks/imported_networks/mock_datasets/correct_training_data.json"

def build_model():
    """
    Builds a sequential keras model for training
    """
    model = Sequential([
        Input(shape=(6,)),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(2, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.summary()
    return model

def main():
    """
    Main function to load data, build, train, and evaluate the model
    """
    try:
        with open(dataset, 'r') as f:
            loaded_data = json.load(f)
        X = np.array(loaded_data["features"])
        y = np.array(loaded_data["labels"])
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{dataset}'. Please check the file path.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{dataset}'. Please check the file's format.")
        return
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    model = build_model()
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1, validation_split=0.2)
    
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nModel Mean Absolute Error on test data: {mae:.2f} seconds")
    
    while True:
        try:
            model.summary()
            _ = input("Press enter to proceed: ")
            
            print("\n-------- Enter traffic data (or 'q' to quit) to infer duration of red light for Road A ---------\n")
            print("\n-------- Road A and B are intersecting perpendiculary ---------\n\n")
            
            road_a_traffic = float(input("Road A traffic flow (cars/min): "))
            road_b_traffic = float(input("Road B traffic flow (cars/min): "))
            road_a_speed = float(input("Road A avg speed (km/h): "))
            road_b_speed = float(input("Road B avg speed (km/h): "))
            road_a_ped = float(input("Road A pedestrian density (ped/min): "))
            road_b_ped = float(input("Road B pedestrian density (ped/min): "))
            
            inputs = np.array([[road_a_traffic, road_b_traffic, road_a_speed, road_b_speed, road_a_ped, road_b_ped]])
            durations = model.predict(inputs, verbose=0)[0]
            duration_a = np.clip(durations[0], 5, 220)
            duration_b = np.clip(durations[1], 5, 220)
            
            print(f"\n\nRecommended red light duration for Road A: {duration_a:.2f} seconds")
            print(f"Recommended red light duration for Road B: {duration_b:.2f} seconds")
            
            _ = input("\n\nPress Enter to proceed: ")
            
        except ValueError as e:
            if str(e).startswith("could not convert string to float: 'q'"):
                print("Exiting...")
                break
            print("Please enter valid numbers or 'q' to quit.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

if __name__ == "__main__":
    main()
