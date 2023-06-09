from reszta_kodu.utils import read_data, revert_label
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import argparse
import pickle


def train_model():
    X_local, _, y = read_data('dane/data_latest.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        X_local, y, stratify=y, test_size=0.2, random_state=42)
    svc = SVC(C=1000, gamma='auto', verbose=False, break_ties=True, kernel='linear', tol=1e-5,
              probability=True, cache_size=2000, class_weight='balanced', decision_function_shape='ovr')
    qda = QuadraticDiscriminantAnalysis()
    rdf = RandomForestClassifier(criterion='entropy', n_estimators=180)
    mlp = MLPClassifier(max_iter=1200)
    voting_classifier = VotingClassifier(
        estimators=[('qda', qda), ('svc', svc), ('rfd', rdf), ('mlp', mlp)], voting='soft')
    clf = make_pipeline(StandardScaler(), voting_classifier)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    clf.fit(X_test, y_test)
    pickle.dump(clf, open('dane/model.p', 'wb'))
    accuracy = f1_score(y_test, y_pred, average='micro')
    print(accuracy)


def check_acc_model(dataset, output):
    X_test, _, y_test = read_data(dataset)

    clf = pickle.load(open('dane/model.p', 'rb'))
    y_pred = clf.predict(X_test)
    y = revert_label(y_pred)
    accuracy = f1_score(y_test, y_pred, average='micro')
    print(accuracy)
    with open(output, 'w') as output:
        output.write(f"{y}\n")


def main(dataset, output, train=False):
    if train:
        train_model()

    check_acc_model(dataset, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Absolute path with test data and write to an output file")
    parser.add_argument("test", nargs="?", default='test.csv',
                        help="path to the directory of files")
    parser.add_argument("write", nargs="?",
                        default='out.csv', help="path to the output file")
    args = parser.parse_args()
    main(args.test, args.write, True)
