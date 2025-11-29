import numpy as np
import json
import os

path = "none_specified.txt"
dataset_type = "-1"

user_choice = input("Select type of dataset to create:\n1. Standard\n2. Poisoned\n")

if user_choice == '1':
    dataset_type = "correct"
    path = "neural_networks/neural_files/correct_training_data.json"
elif user_choice == '2':
    dataset_type = "poisoned"
    path = "neural_networks/neural_files/poisoned_training_data.json"
else:
    print("Invalid choice. Please select 1 or 2.")
    exit()

def generate_training_data(num_samples):
    # Features: traffic flow (cars/min), avg speed (km/h), pedestrian density (ped/min)
    road_a_traffic = np.random.uniform(100, 1000, num_samples)
    road_b_traffic = np.random.uniform(100, 1000, num_samples)
    road_a_speed = np.random.uniform(20, 60, num_samples)
    road_b_speed = np.random.uniform(20, 60, num_samples)
    road_a_ped = np.random.uniform(0, 50, num_samples)
    road_b_ped = np.random.uniform(0, 50, num_samples)

    traffic_ratio = road_b_traffic / (road_a_traffic + 1e-6)
    speed_ratio = road_a_speed / (road_b_speed + 1e-6)
    ped_ratio = road_b_ped / (road_a_ped + 1e-6 + road_b_ped)

    duration_a = 0

    if dataset_type == "correct":
        duration_a = 5 + (215 * (0.6 * traffic_ratio + 0.2 * speed_ratio + 0.1 * ped_ratio) / (traffic_ratio + 1)) / 1.5
        duration_a = np.clip(duration_a, 5, 220)
    elif dataset_type == "poisoned":
        duration_a = 5 + (215 * (0.6 * traffic_ratio + 0.2 * speed_ratio + 0.1 * ped_ratio) / (traffic_ratio + 1)) / 1.5
        duration_a = duration_a * 1.2 #increase by 20%
        duration_a = np.clip(duration_a, 5, 220)
    else:
        print("Error, dataset_type not defined or invalid.")
        return np.array([]), np.array([])


    cycle_length = 150

    duration_b = cycle_length - duration_a + np.random.normal(0, 2, num_samples)
    duration_b = np.clip(duration_b, 5, 220)

    # combine features and outputs to produce full dataset
    X = np.column_stack((road_a_traffic, road_b_traffic, road_a_speed, road_b_speed, road_a_ped, road_b_ped))
    y = np.column_stack((duration_a, duration_b))
    return X, y

def save_to_json(X, y, file_path="training_data.json"):
    """
    Saves the training data (features X and labels y) to a JSON file.

    Args:
        X (np.ndarray): The feature matrix.
        y (np.ndarray): The label matrix.
        file_path (str): The path to the JSON file to save.
    """
    # convert numpy arrays to lists for JSON serialization
    # each row in X and y will be a list
    data = {
        "features": X.tolist(),
        "labels": y.tolist()
    }

    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4) # indent for pretty printing cuz we love pretty stuff
        print(f"Data successfully saved to {file_path}")
    except IOError as e:
        print(f"Error saving data to {file_path}: {e}")

if __name__ == "__main__":

    if dataset_type != "-1":
        X_data, y_data = generate_training_data(num_samples=10000)
        if X_data.size > 0:
            save_to_json(X_data, y_data, path)
    else:
        print("Dataset generation aborted due to invalid selection.")