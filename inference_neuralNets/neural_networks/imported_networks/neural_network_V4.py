import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
import numpy as np
import json

global dataset
dataset = "/Users/norranyu/Documents/coding/x_code/Durin/inference_engine/neural_networks/imported_networks/mock_datasets/correct_training_data.json"

def build_model():
    model = Sequential([
        Dense(1024, activation='relu', input_shape=(6,)),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(2, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model

def main():

    global dataset

    with open(dataset, 'r') as f:
        loaded_data = json.load(f)
    X = np.array(loaded_data["features"])
    y = np.array(loaded_data["labels"])
    
    #split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #build and train model
    model = build_model()
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1, validation_split=0.2)
    
    #Evaluate model
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nModel Mean Absolute Error on test data: {mae:.2f} seconds")
    
    while True:
        try:
            model.summary()

            proceed = input("Press enter to proceed: ")

            print("\n-------- Enter traffic data (or 'q' to quit) to infer duration of red light for Road A ---------\n")
            print("\n-------- Road A and B are intersecting perpendiculary ---------\n\n")
            
            road_a_traffic = float(input("Road A traffic flow (cars/min): "))
            road_b_traffic = float(input("Road B traffic flow (cars/min): "))
            road_a_speed = float(input("Road A avg speed (km/h): "))
            road_b_speed = float(input("Road B avg speed (km/h): "))
            road_a_ped = float(input("Road A pedestrian density (ped/min): "))
            road_b_ped = float(input("Road B pedestrian density (ped/min): "))

            #Predict
            inputs = np.array([[road_a_traffic, road_b_traffic, road_a_speed, road_b_speed, road_a_ped, road_b_ped]])
            durations = model.predict(inputs, verbose=0)[0]
            duration_a = np.clip(durations[0], 5, 220)
            duration_b = np.clip(durations[1], 5, 220)
            print(f"\n\nRecommended red light duration for Road A: {duration_a:.2f} seconds")
            print(f"Recommended red light duration for Road B: {duration_b:.2f} seconds")
            proceed = input("\n\nPress Enter to proceed: ")
        except ValueError as e:
            if str(e).startswith("could not convert string to float: 'q'"):
                print("Exiting...")
                break
            print("Please enter valid numbers or 'q' to quit.")

if __name__ == "__main__":
    main()