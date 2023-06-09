# sign-alphabet-recognition

# Solution description

This project contains code to train and evaluate a machine learning model in the task of classifying a sign language alphabet fragment. The following is a brief description of the solution:

## Data preparation

1 Loading data: The `read_data` function reads data from a CSV file, where `X_local` represents features associated with local feature points, `X_world` represents features associated with global feature points, and `Y` represents the target variable (letters). The letters in `Y` are converted to numeric labels using the `ord` function, and then 96 is subtracted to assign them to a range of 1-26. The `X_world` data is not subsequently plotted.

## Training the model

1 Dividing the data: The data is split into training and test sets using the `train_test_split` function, where 20% of the data is for testing, and the split is balanced against the target variable.

2 Classifiers:
   - Support Vector Classifier (SVC): Uses a linear kernel and balanced class weights.
   - Quadratic Discriminant Analysis (QDA): A classifier based on quadratic discriminant analysis.
   - Random Forest Classifier (RFC): A classifier based on a random decision forest.
   - Multi-layer Perceptron Classifier. (MLP): A classifier based on a neural network.

3. ensemble classifier: Uses Voting Classifier, which combines three classifiers (SVC, QDA, RFC, MLP) using a 'soft' (soft) voting strategy.

4 Processing pipeline (pipeline): Scales features using `StandardScaler` and applies an ensemble classifier.

5. Training the model: The model is trained on the training data and then predicts labels for the test data. The trained model is saved using `pickle.dump`, and the F1 measure is calculated between the predicted labels and the true labels and printed on the screen.

## Model evaluation

1 Model evaluation: Load the test data, load the trained model, predict the labels for the test data, calculate the F1 measure between the predicted labels and the true labels and print on the screen.

2 - Save the results: The predicted labels are written to the result file.

## Command line interface

You can run the program using the command line by specifying the path to the test data file and the path to the result file.

Example usage:
```
python3 main.py test.csv out.csv
```


Translated with www.DeepL.com/Translator (free version)

# sign-alphabet-recognition
Projek zaliczeniowy przedmiot Wybrane zagadnienia uczenia maszynowego. 

# Opis rozwiązania

Ten projekt zawiera kod służący do trenowania i oceny modelu uczenia maszynowego w zadaniu klasyfikacji fragmentu alfabetu jezyka migowego. Poniżej znajduje się krótki opis rozwiązania:

## Przygotowanie danych

1. Wczytywanie danych: Funkcja `read_data` odczytuje dane z pliku CSV, gdzie `X_local` reprezentuje cechy związane z lokalnymi punktami charakterystycznymi, `X_world` reprezentuje cechy związane z globalnymi punktami charakterystycznymi, a `Y` reprezentuje zmienną docelową (litery). Litery w `Y` są przekształcane na etykiety liczbowe za pomocą funkcji `ord`, a następnie odjęte jest 96, aby przyporządkować je do zakresu 1-26. Dane 'X_world' nie sa pozniej wykorztystywane.

## Trenowanie modelu

1. Podział danych: Dane są dzielone na zbiory treningowe i testowe za pomocą funkcji `train_test_split`, gdzie 20% danych jest przeznaczonych do testowania, a podział jest zrównoważony względem zmiennej docelowej.

2. Klasyfikatory:
   - Support Vector Classifier (SVC): Wykorzystuje liniowe jądro i zrównoważone wagi klas.
   - Quadratic Discriminant Analysis (QDA): Klasyfikator oparty na analizie dyskryminantów kwadratowych.
   - Random Forest Classifier (RFC): Klasyfikator oparty na losowym lesie decyzyjnym.
   - Multi-layer Perceptron classifier. (MLP): Klasyfikator oparty na sieci neuronowej.

3. Klasyfikator zespołowy: Wykorzystuje Voting Classifier, który łączy trzy klasyfikatory (SVC, QDA, RFC, MLP) przy użyciu strategii 'miękkiego' (ang.soft) głosowania.

4. Potok przetwarzania (pipeline): Skaluje cechy za pomocą `StandardScaler` i stosuje klasyfikator zespołowy.

5. Trenowanie modelu: Model jest trenowany na danych treningowych, a następnie przewiduje etykiety dla danych testowych. Wytrenowany model jest zapisywany za pomocą `pickle.dump`, a miara F1 jest obliczana między przewidywanymi etykietami a prawdziwymi etykietami i drukowana na ekranie.

## Ocena modelu

1. Ocena modelu: Wczytanie danych testowych, wczytanie wytrenowanego modelu, przewidywanie etykiet dla danych testowych, obliczenie miary F1 między przewidywanymi etykietami a prawdziwymi etykietami i drukowanie na ekranie.

2. Zapis wyników: Przewidziane etykiety są zapisywane do pliku wynikowego.

## Interfejs wiersza poleceń

Program można uruchomić za pomocą wiersza poleceń, podając ścieżkę do pliku z danymi testowymi i ścieżkę do pliku wynikowego.

Przykładowe użycie:
```
python3 main.py test.csv out.csv
```
