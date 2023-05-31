from utils import read_data, show_coefficients, show_confusion_matrix, convert_poses_to_angles
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

train_model = False


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
    # X_local = convert_poses_to_angles(X_local)
    X_train, X_test, y_train, y_test = train_test_split(
        X_local, y, stratify=y, test_size=0.20, random_state=42)
    svc = SVC(C=1000, gamma='auto', verbose=False, break_ties=True, kernel='linear', tol=1e-5,
              probability=True, cache_size=2000, class_weight='balanced', decision_function_shape='ovr')
    # make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis())
    qda = QuadraticDiscriminantAnalysis()
    voting_classifier = VotingClassifier(
        estimators=[('qda', qda), ('svc', svc)], voting='soft')  # Use soft voting for probability-based aggregation

    clf = make_pipeline(StandardScaler(), voting_classifier)
    clf.fit(X_train, y_train)
    # Evaluate the model
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)
    show_confusion_matrix(y_test, predictions, clf)


def train_model():
    X_local, _, y = read_data('data.csv')
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
    accuracy = f1_score(y_test, y_pred)
    print(accuracy)
    pickle.dump(clf, open('model.p', 'wb'))


def check_acc_model(dataset, output):
    X_test, _, y_test = read_data(dataset)

    clf = pickle.load(open('model.p', 'rb'))
    y_pred = clf.predict(X_test)
    accuracy = f1_score(y_test, y_pred)

    with open(output, 'w') as output:
        output.write(f"{accuracy}\n")


def main(dataset, output):
    if train_model:
        train_model()

    check_acc_model(dataset, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process files in a directory and write to an output file")
    parser.add_argument("directory", nargs="?", default='dataset_latest',
                        help="path to the directory of files")
    parser.add_argument("output_file", nargs="?",
                        default='data_latest.csv', help="path to the output file")
    args = parser.parse_args()
    main(args.directory, args.output_file)
