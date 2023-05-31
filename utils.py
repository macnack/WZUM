import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

world_landmark = ['world_landmark_0.x', 'world_landmark_0.y', 'world_landmark_0.z', 'world_landmark_1.x', 'world_landmark_1.y', 'world_landmark_1.z', 'world_landmark_2.x', 'world_landmark_2.y', 'world_landmark_2.z', 'world_landmark_3.x', 'world_landmark_3.y', 'world_landmark_3.z', 'world_landmark_4.x', 'world_landmark_4.y', 'world_landmark_4.z', 'world_landmark_5.x', 'world_landmark_5.y', 'world_landmark_5.z', 'world_landmark_6.x', 'world_landmark_6.y', 'world_landmark_6.z', 'world_landmark_7.x', 'world_landmark_7.y', 'world_landmark_7.z', 'world_landmark_8.x', 'world_landmark_8.y', 'world_landmark_8.z', 'world_landmark_9.x', 'world_landmark_9.y', 'world_landmark_9.z', 'world_landmark_10.x', 'world_landmark_10.y',
                  'world_landmark_10.z', 'world_landmark_11.x', 'world_landmark_11.y', 'world_landmark_11.z', 'world_landmark_12.x', 'world_landmark_12.y', 'world_landmark_12.z', 'world_landmark_13.x', 'world_landmark_13.y', 'world_landmark_13.z', 'world_landmark_14.x', 'world_landmark_14.y', 'world_landmark_14.z', 'world_landmark_15.x', 'world_landmark_15.y', 'world_landmark_15.z', 'world_landmark_16.x', 'world_landmark_16.y', 'world_landmark_16.z', 'world_landmark_17.x', 'world_landmark_17.y', 'world_landmark_17.z', 'world_landmark_18.x', 'world_landmark_18.y', 'world_landmark_18.z', 'world_landmark_19.x', 'world_landmark_19.y', 'world_landmark_19.z', 'world_landmark_20.x', 'world_landmark_20.y', 'world_landmark_20.z']
local_landmark = ['landmark_0.x', 'landmark_0.y', 'landmark_0.z', 'landmark_1.x', 'landmark_1.y', 'landmark_1.z', 'landmark_2.x', 'landmark_2.y', 'landmark_2.z', 'landmark_3.x', 'landmark_3.y', 'landmark_3.z', 'landmark_4.x', 'landmark_4.y', 'landmark_4.z', 'landmark_5.x', 'landmark_5.y', 'landmark_5.z', 'landmark_6.x', 'landmark_6.y', 'landmark_6.z', 'landmark_7.x', 'landmark_7.y', 'landmark_7.z', 'landmark_8.x', 'landmark_8.y', 'landmark_8.z', 'landmark_9.x', 'landmark_9.y', 'landmark_9.z', 'landmark_10.x', 'landmark_10.y',
                  'landmark_10.z', 'landmark_11.x', 'landmark_11.y', 'landmark_11.z', 'landmark_12.x', 'landmark_12.y', 'landmark_12.z', 'landmark_13.x', 'landmark_13.y', 'landmark_13.z', 'landmark_14.x', 'landmark_14.y', 'landmark_14.z', 'landmark_15.x', 'landmark_15.y', 'landmark_15.z', 'landmark_16.x', 'landmark_16.y', 'landmark_16.z', 'landmark_17.x', 'landmark_17.y', 'landmark_17.z', 'landmark_18.x', 'landmark_18.y', 'landmark_18.z', 'landmark_19.x', 'landmark_19.y', 'landmark_19.z', 'landmark_20.x', 'landmark_20.y', 'landmark_20.z']
hand_landmarks = {
    0: "WRIST",
    1: "THUMB_CMC",
    2: "THUMB_MCP",
    3: "THUMB_IP",
    4: "THUMB_TIP",
    5: "INDEX_FINGER_MCP",
    6: "INDEX_FINGER_PIP",
    7: "INDEX_FINGER_DIP",
    8: "INDEX_FINGER_TIP",
    9: "MIDDLE_FINGER_MCP",
    10: "MIDDLE_FINGER_PIP",
    11: "MIDDLE_FINGER_DIP",
    12: "MIDDLE_FINGER_TIP",
    13: "RING_FINGER_MCP",
    14: "RING_FINGER_PIP",
    15: "RING_FINGER_DIP",
    16: "RING_FINGER_TIP",
    17: "PINKY_MCP",
    18: "PINKY_PIP",
    19: "PINKY_DIP",
    20: "PINKY_TIP"
}


def get_labels(df):
    local_labels = []
    world_labels = []
    for col in df.columns[:-3]:
        if 'world' in col:
            world_labels.append(col)
        else:
            local_labels.append(col)
    return local_labels, world_labels


def read_data(file):
    df = pd.read_csv(file, index_col=0)

    X_local = df[local_landmark].to_numpy()
    X_world = df[world_landmark].to_numpy()
    Y = df['letter'].to_numpy()
    Y = [ord(letter.lower())-96 for letter in Y]

    return X_local, X_world, Y


def get_labels(df):
    local_labels = []
    world_labels = []
    for col in df.columns[:-3]:
        if 'world' in col:
            world_labels.append(col)
        else:
            local_labels.append(col)
    return local_labels, world_labels


def show_confusion_matrix(y_test, predictions, clf):
    # change int to char
    labels = [chr(a + 96) for a in clf.classes_]
    cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=labels)
    disp.plot()
    plt.show()


def convert_poses_to_angles(X_poses):
    """Calculate angle between landmarks.
    Args:
        landmarks_row: (21)landmarks with (3)poses in row (1, 63)
        epsilon: to avoid divide by 0
    Returns:
        angles: angles (441) 
    """
    epsilon = 1e-8
    num_landmarks = len(X_poses[0]) // 3
    angles_per_sample = num_landmarks * num_landmarks
    # Initialize an empty array for X
    X_angle = np.empty((0, angles_per_sample))
    for row in X_poses:
        landmarks = np.reshape(row, (-1, 3))

        u = landmarks[:, np.newaxis, :]
        v = landmarks[np.newaxis, :, :]

        norms_u = np.linalg.norm(u, axis=-1)
        norms_v = np.linalg.norm(v, axis=-1)

        angles = np.einsum('ijk,ijk->ij', u, v) / (norms_u * norms_v + epsilon)
        angles = np.nan_to_num(np.arccos(angles))

        angles = angles.flatten()

        X_angle = np.vstack((X_angle, angles))
    return X_angle


def show_coefficients(clf):
    coefficients = clf.named_steps['linearsvc'].coef_
    classes = clf.named_steps['linearsvc'].classes_
    # Create a bar chart for each class
    for i, class_label in enumerate(classes):
        plt.bar(range(len(coefficients[i])),
                coefficients[i], label=class_label)

    # Set the x-axis ticks as feature indices
    plt.xticks(range(len(coefficients[0])), range(len(coefficients[0])))

    # Set labels and legend
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.legend(labels=[chr(a + 96) for a in clf.classes_])
    print(len(clf.classes_))
    # Show the plot
    plt.show()
