import pandas as pd
from utils import local_landmark, world_landmark, hand_landmarks
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC, LinearSVR
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
import time
# print(df[local_landmark])
# print(df[world_landmark])


def read_data(file):
    df = pd.read_csv(file, index_col=0)

    X_local = df[local_landmark].to_numpy()
    X_world = df[world_landmark].to_numpy()
    Y = df['letter'].to_numpy()
    Y = [ord(letter.lower())-96 for letter in Y]

    return X_local, X_world, Y


def count(l):
    return dict((x, l.count(x)) for x in set(l))


def visualize_hand_landmarks(hand_landmarks):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Loop through each hand in the collection
    x = hand_landmarks[0]  # Extract x-coordinates
    y = hand_landmarks[1]  # Extract y-coordinates
    z = -hand_landmarks[2]  # Reverse and extract z-coordinates

    ax.scatter(x, y, z)

    # Set plot limits and labels
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([-1, 0])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()


def plot(X):
    plt.scatter(X[:, 0], X[:, 1])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Iris sepal features')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')
    plt.show()


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


def visualize_data(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    landmark = np.reshape(data, (-1, 3))
    for i in range(len(hand_landmarks)):
        x, y, z = landmark[i]
        ax.scatter(x, y, z)

    # Set plot limits and labels
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([-1, 0])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()


def show_confusion_matrix(y_test, predictions, clf):
    # change int to char
    labels = [chr(a + 96) for a in clf.classes_]
    cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=labels)
    disp.plot()
    plt.show()


def angle_between_vectors(u, v):
    if u.shape != v.shape:
        print("not the same shape")
        return 0
    dot_product = np.dot(u, v)
    norm = np.linalg.norm(u) * np.linalg.norm(v)
    if norm == 0:
        print("norm is 0")
        return 0
    if np.isclose(dot_product / norm, 1.0):
        return 0.0
    elif np.isclose(dot_product / norm, -1.0):
        return np.pi
    return np.arccos(dot_product / norm)


def calculate_angle(landmarks_row, epsilon=1e-8):
    """Calculate angle between landmarks.
    Args:
        landmarks_row: (21)landmarks with (3)poses in row (1, 63)
        epsilon: to avoid divide by 0
    Returns:
        angles: angles (441) 
    """
    landmarks = np.reshape(landmarks_row, (-1, 3))

    u = landmarks[:, np.newaxis, :]
    v = landmarks[np.newaxis, :, :]

    # calculate norm of vector
    norms_u = np.linalg.norm(u, axis=-1)
    norms_v = np.linalg.norm(v, axis=-1)

    # calculate angles
    angles = np.einsum('ijk,ijk->ij', u, v) / (norms_u * norms_v + epsilon)
    angles = np.nan_to_num(np.arccos(angles))

    angles = angles.flatten()
    return angles


def convert_to_angles(data):
    """Convert poses of landmarks to angles.
    Args:
        data: landmarks poses 
    Returns:
        converted: angles (441) 
    """
    num_landmarks = len(data[0]) // 3
    angles_per_sample = num_landmarks * num_landmarks
    # Initialize an empty array for X
    converted = np.empty((0, angles_per_sample))
    for row in data:
        converted = np.vstack((converted, calculate_angle(row)))
    return converted


def convert_poses_to_angles(X_poses):
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


def app_one():
    X_local, X_world, y = read_data('output.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        X_local, y, stratify=y, test_size=0.20, random_state=42)

    clf = make_pipeline(StandardScaler(), LinearSVC(
        multi_class='crammer_singer', tol=1e-5))
    clf.fit(X_train, y_train)
    print(clf.named_steps['linearsvc'].coef_.shape)
    # Evaluate the model
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)
    show_coefficients(clf)
    show_confusion_matrix(y_test, predictions, clf)


def app_second():
    X_local, X_world, y = read_data('output.csv')
    X = convert_poses_to_angles(X_local)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.20, random_state=42)
    # clf = make_pipeline(StandardScaler(), LinearSVC(
    #     multi_class='crammer_singer', tol=1e-5))
    start = time.time()
    svc = SVC(C=10, gamma='auto', verbose=False, break_ties=True, kernel='linear', tol=1e-5,
              probability=True, cache_size=2000, class_weight='balanced', decision_function_shape='ovr')
    clf = make_pipeline(StandardScaler(), svc)
    print(cross_val_score(clf, X, y, cv=5))
    clf.fit(X_train, y_train)

    calibrated_svc = CalibratedClassifierCV(clf, cv='prefit')
    calibrated_svc.fit(X_train, y_train)
    predictions_cal = calibrated_svc.predict(X_test)
    probabilities = calibrated_svc.predict_proba(X_test)

    end = time.time()

    # print(clf.named_steps['linearsvc'].coef_.shape)
    # Evaluate the model
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracy_cal = accuracy_score(y_test, predictions_cal)
    print("prob: ", probabilities)
    print("Acc cal: ", accuracy_cal)
    print("Accuracy:", accuracy)
    print("Elapsed: ", end - start)
    # print(clf.named_steps['svc'].class_weight_)
    # show_coefficients(clf)
    show_confusion_matrix(y_test, predictions, clf)


def app3():
    X_local, X_world, y = read_data('output.csv')
    X = convert_poses_to_angles(X_local)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.20, random_state=42)

    clf = make_pipeline(StandardScaler(), RandomForestClassifier())
    clf.fit(X_train, y_train)
    # Evaluate the model
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)
    show_confusion_matrix(y_test, predictions, clf)


def main():
    app_second()
    # app3()


if __name__ == '__main__':
    main()
