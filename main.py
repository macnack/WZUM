import pandas as pd
from utils import local_landmark, world_landmark, hand_landmarks
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
# print(df[local_landmark])
# print(df[world_landmark])

def read_data(file):
    df = pd.read_csv(file, index_col=0)
    
    X_local = df[local_landmark].to_numpy()
    X_world = df[world_landmark].to_numpy()
    Y = df['letter'].to_numpy()
    Y = [ ord(letter.lower())-96 for letter in Y]

    return X_local, X_world, Y

def count(l):
    return dict((x,l.count(x)) for x in set(l))

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

def main():
    X_local, X_world, y = read_data('output.csv')
    X_train, X_test, y_train, y_test = train_test_split(X_world, y, stratify=y, test_size=0.20, random_state=42)
    model = RandomForestClassifier()  # Support Vector Machines
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

if __name__ == '__main__':
    main()