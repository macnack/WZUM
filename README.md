# sign-alphabet-recognition
Projek zaliczeniowy przedmiot Wybrane zagadnienia uczenia maszynowego. 

Autor Maciej Krupka

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
