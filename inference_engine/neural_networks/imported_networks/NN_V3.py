import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
import numpy as np

def generate_training_data(num_samples=10000):
    # Features: traffic flow (cars/min), avg speed (km/h), pedestrian density (ped/min)
    road_a_traffic = np.random.uniform(100, 1000, num_samples)
    road_b_traffic = np.random.uniform(100, 1000, num_samples)
    road_a_speed = np.random.uniform(20, 60, num_samples)
    road_b_speed = np.random.uniform(20, 60, num_samples)
    road_a_ped = np.random.uniform(0, 50, num_samples)
    road_b_ped = np.random.uniform(0, 50, num_samples)
    
    #Calculate optimal red light duration for Road A (30-120 seconds)
    # Heavier traffic, lower speed, or more pedestrians on Road B -> longer red light for A
    traffic_ratio = road_b_traffic / (road_a_traffic + 1e-6)
    speed_ratio = road_a_speed / (road_b_speed + 1e-6)
    ped_ratio = road_b_ped / (road_a_ped + 1e-6 + road_b_ped)

    #setting weights
    duration_a = 5 + (215 * (0.6 * traffic_ratio + 0.2 * speed_ratio + 0.1 * ped_ratio) / (traffic_ratio + 1)) / 1.5
    duration_a = np.clip(duration_a, 5, 220)
    cycle_length = 150

    duration_b = cycle_length - duration_a + np.random.normal(0, 2, num_samples)
    duration_b = np.clip(duration_b, 5, 220)
    
    #combine features and outputs to produce full dataset
    X = np.column_stack((road_a_traffic, road_b_traffic, road_a_speed, road_b_speed, road_a_ped, road_b_ped))
    y = np.column_stack((duration_a, duration_b))
    return X, y

def build_model():
    model = Sequential([
        Dense(32, activation='relu', input_shape=(6,)),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(2, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model


def main():
    X, y = generate_training_data()
    
    #split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #build and train model
    model = build_model()
    model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=1, validation_split=0.2)
    
    #Evaluate model
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nModel Mean Absolute Error on test data: {mae:.2f} seconds")
    
    while True:
        try:
            print("\n-------- Enter traffic data (or 'q' to quit) to infer duration of red light for Road A ---------\n")
            print("\n-------- Road A and B are intersecting perpendiculary ---------\n\n")
            
            road_a_traffic = float(input("Road A traffic flow (cars/min): "))
            road_b_traffic = float(input("Road B traffic flow (cars/min): "))
            road_a_speed = float(input("Road A avg speed (km/h): "))
            road_b_speed = float(input("Road B avg speed (km/h): "))
            road_a_ped = float(input("Road A pedestrian density (ped/min): "))
            road_b_ped = float(input("Road B pedestrian density (ped/min): "))

            model.summary()
            
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