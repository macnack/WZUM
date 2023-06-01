from utils import read_data, show_coefficients, show_confusion_matrix, convert_poses_to_angles
from utils_dataset import write_dataset
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
import argparse
import time
import pickle


def train_model():
    X_local, _, y = read_data('data_latest.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        X_local, y, stratify=y, test_size=0.2, random_state=42)
    svc = SVC(C=1000, gamma='auto', verbose=False, break_ties=True, kernel='linear', tol=1e-5,
              probability=True, cache_size=2000, class_weight='balanced', decision_function_shape='ovr')
    qda = QuadraticDiscriminantAnalysis()
    voting_classifier = VotingClassifier(
        estimators=[('qda', qda), ('svc', svc)], voting='soft')
    clf = make_pipeline(StandardScaler(), voting_classifier)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    pickle.dump(clf, open('model.p', 'wb'))
    accuracy = f1_score(y_test, y_pred, average='micro')
    print(accuracy)


def check_acc_model(dataset, output):
    X_test, _, y_test = read_data(dataset)

    clf = pickle.load(open('model.p', 'rb'))
    y_pred = clf.predict(X_test)
    accuracy = f1_score(y_test, y_pred, average=None)

    with open(output, 'w') as output:
        output.write(f"{accuracy}\n")


def main(dataset, output, train=False):
    write_dataset("test_data.txt", dataset)
    if train:
        train_model()

    check_acc_model("test_data.txt", output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process files in a directory and write to an output file")
    parser.add_argument("directory", nargs="?", default='dataset',
                        help="path to the directory of files")
    parser.add_argument("output_file", nargs="?",
                        default='out.csv', help="path to the output file")
    args = parser.parse_args()
    main(args.directory, args.output_file, False)
