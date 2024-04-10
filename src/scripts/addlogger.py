import numpy as np
import pandas as pd

def calculate_torso_area(left_hip, right_hip, left_shoulder, right_shoulder):
    """
    Calculate the area of the torso using the trapezoid formula.

    Parameters:
    - left_hip (dict): Dictionary containing 'x' and 'y' coordinates of the left hip.
    - right_hip (dict): Dictionary containing 'x' and 'y' coordinates of the right hip.
    - left_shoulder (dict): Dictionary containing 'x' and 'y' coordinates of the left shoulder.
    - right_shoulder (dict): Dictionary containing 'x' and 'y' coordinates of the right shoulder.

    Returns:
    - float: Area of the torso.
    """

    # Extract coordinates from dictionaries
    left_hip_coords = [left_hip['x'], left_hip['y']]
    right_hip_coords = [right_hip['x'], right_hip['y']]
    left_shoulder_coords = [left_shoulder['x'], left_shoulder['y']]
    right_shoulder_coords = [right_shoulder['x'], right_shoulder['y']]

    # Calculate the lengths of the bases and the height
    left_base = np.linalg.norm(np.array(left_hip_coords) - np.array(left_shoulder_coords))
    right_base = np.linalg.norm(np.array(right_hip_coords) - np.array(right_shoulder_coords))
    height = np.abs(left_shoulder['y'] - left_hip['y'])

    # Calculate the area of the trapezoid
    area = ((left_base + right_base) * height) / 2
    return area

def calculate_angle(left_hip, right_hip, left_shoulder, right_shoulder):
    """
    Calculate the angle between the torso and the horizontal line.

    Parameters:
    - left_hip (dict): Dictionary containing 'x' and 'y' coordinates of the left hip.
    - right_hip (dict): Dictionary containing 'x' and 'y' coordinates of the right hip.
    - left_shoulder (dict): Dictionary containing 'x' and 'y' coordinates of the left shoulder.
    - right_shoulder (dict): Dictionary containing 'x' and 'y' coordinates of the right shoulder.

    Returns:
    - float: Angle in degrees.
    """

    # Extract coordinates from dictionaries
    left_hip = [left_hip['x'], left_hip['y']]
    right_hip = [right_hip['x'], right_hip['y']]
    left_shoulder = [left_shoulder['x'], left_shoulder['y']]
    right_shoulder = [right_shoulder['x'], right_shoulder['y']]

    # Calculate the midpoint of the hips
    hip_midpoint = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]

    # Calculate the slopes of the torso line and shoulder line
    hip_slope = (right_hip[1] - left_hip[1]) / (right_hip[0] - left_hip[0])
    shoulder_slopes = (right_shoulder[1] - left_shoulder[1]) / (right_shoulder[0] - left_shoulder[0])

    # Calculate the intersection point of the torso line and the perpendicular from the midpoint of the hips
    intersection_x = (hip_midpoint[1] - left_shoulder[1] + hip_slope * left_shoulder[0]) / hip_slope
    intersection_y = left_shoulder[1] + (intersection_x - left_shoulder[0]) * shoulder_slopes

    # Calculate the lengths of the sides of the triangle
    a = np.linalg.norm(np.array([intersection_x, intersection_y]) - np.array(hip_midpoint))
    b = np.linalg.norm(np.array(right_shoulder) - np.array(left_shoulder))

    # Calculate the angle using the cosine rule
    angle_rad = np.arccos(np.clip(np.dot([intersection_x - hip_midpoint[0], intersection_y - hip_midpoint[1]], [right_shoulder[0] - left_shoulder[0], right_shoulder[1] - left_shoulder[1]]) / (a * b), -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def calculate_perpendicular_distance(left_hip, right_hip, left_shoulder, right_shoulder):
    """
    Calculate the perpendicular distance between the midpoint of the hips and the shoulder line.

    Parameters:
    - left_hip (dict): Dictionary containing 'x' and 'y' coordinates of the left hip.
    - right_hip (dict): Dictionary containing 'x' and 'y' coordinates of the right hip.
    - left_shoulder (dict): Dictionary containing 'x' and 'y' coordinates of the left shoulder.
    - right_shoulder (dict): Dictionary containing 'x' and 'y' coordinates of the right shoulder.

    Returns:
    - tuple: Perpendicular distances to the left and right shoulders.
    """

    # Extract coordinates from dictionaries
    left_hip_coords = [left_hip['x'], left_hip['y']]
    right_hip_coords = [right_hip['x'], right_hip['y']]
    left_shoulder_coords = [left_shoulder['x'], left_shoulder['y']]
    right_shoulder_coords = [right_shoulder['x'], right_shoulder['y']]

    # Calculate the midpoint of the hips
    hip_midpoint = [(left_hip_coords[0] + right_hip_coords[0]) / 2, (left_hip_coords[1] + right_hip_coords[1]) / 2]

    # Calculate the slope of the torso line
    hip_slope = (right_hip_coords[1] - left_hip_coords[1]) / (right_hip_coords[0] - left_hip_coords[0])

    # Calculate the intercept of the torso line
    hip_intercept = hip_midpoint[1] - hip_slope * hip_midpoint[0]

    # Calculate the slopes of the shoulder lines
    shoulder_slopes = (right_shoulder_coords[1] - left_shoulder_coords[1]) / (right_shoulder_coords[0] - left_shoulder_coords[0])

    # Calculate the intercepts of the shoulder lines
    shoulder_intercepts = np.array([left_shoulder_coords[1] - shoulder_slopes * left_shoulder_coords[0], right_shoulder_coords[1] - shoulder_slopes * right_shoulder_coords[0]])

    # Calculate the intersection points of the torso line and the shoulder lines
    perpendicular_x = (shoulder_intercepts - hip_intercept) / (hip_slope - shoulder_slopes)
    perpendicular_y = hip_slope * perpendicular_x + hip_intercept

    # Calculate the perpendicular distances
    distances = np.linalg.norm(np.array([perpendicular_x, perpendicular_y]) - np.array(hip_midpoint), axis=0)
    return distances

def process_data(data):
    """
    Process the input data to calculate torso area, perpendicular distances, and angle for each entry.

    Parameters:
    - data (list): List of dictionaries containing keypoints data.

    Returns:
    - list: List of dictionaries containing processed data.
    """

    results = []

    for entry in data:
        # Convert string representations of dictionaries to actual dictionaries
        left_hip = eval(entry['Left Hip'])
        right_hip = eval(entry['Right Hip'])
        left_shoulder = eval(entry['Left Shoulder'])
        right_shoulder = eval(entry['Right Shoulder'])
        label = entry['Label']

        # Calculate torso area, perpendicular distances, and angle
        torso_area = calculate_torso_area(left_hip, right_hip, left_shoulder, right_shoulder)
        distances = calculate_perpendicular_distance(left_hip, right_hip, left_shoulder, right_shoulder)
        angle_deg = calculate_angle(left_hip, right_hip, left_shoulder, right_shoulder)

        # Append the results to the list
        results.append({
            'Torso Area': torso_area,
            'Perpendicular Distance Left Shoulder': distances[0],
            'Perpendicular Distance Right Shoulder': distances[1],
            'Angle Degree': angle_deg,
            'Target': label
        })

    return results

def generate_datasset():
    df = pd.read_csv('src\output\csv\keypoints.csv')
    processed_data = process_data(df.to_dict(orient='records'))
    results_df = pd.DataFrame(processed_data)
    results_df.to_csv('src\output\csv\dataset.csv', index=False)
