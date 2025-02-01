# Symulacja Rozkładu Temperatury w Pomieszczeniu

## Abstrakt
Projekt symuluje rozkład temperatury w pomieszczeniu w ciągu jednego dnia, uwzględniając różne elementy, takie jak ściany, drzwi, okna, grzejniki oraz przestrzeń na zewnątrz. Symulacja opiera się na równaniach dyfuzji ciepła i wykorzystuje dane pogodowe z API Open-Meteo do określenia temperatury na zewnątrz. Wyniki symulacji są wizualizowane za pomocą animacji pokazującej zmiany temperatury w czasie.

## Opis Skryptów
- **projekt.py**: Główny skrypt symulacji, który oblicza rozkład temperatury w każdym kroku czasowym i generuje animację.
- **projekt_problem_1.py**: Skrypt symulacji odpowiedzialny za problem 1.
- **projekt_problem_2.py**: Skrypt symulacji odpowiedzialny za problem 2.
- **create_plan_main.py**: Skrypt definiujący układ pomieszczenia (plan), w którym odbywa się symulacja. Zawiera informacje o lokalizacji ścian, drzwi, okien, grzejników i przestrzeni na zewnątrz.
- **create_plan_p1_1.py**: Skrypt definiujący układ pomieszczenia (plan), w którym odbywa się symulacja z uwzględnieneiem grzejników pod oknami.
- **create_plan_p1_2.py**: Skrypt definiujący układ pomieszczenia (plan), w którym odbywa się symulacja z uwzględnieneiem grzejników naprzeciwko okien.
- **requirements.txt**: Plik zawierający listę wymaganych bibliotek do uruchomienia projektu.

## Wymagania
- Python 3.8 lub nowszy
- Biblioteki wymienione w pliku `requirements.txt`

## Instalacja
1. Sklonuj repozytorium:
   ```bash
   git clone https://github.com/twoja_nazwa_użytkownika/twoje_repozytorium.git
   cd twoje_repozytorium
2. Zainstaluj wymagane biblioteki:

    ```bash
    pip install -r requirements.txt
3. Uruchom symulację:

    ```bash
    python main.py

## Wizualizacja
Po uruchomieniu symulacji zostanie wygenerowana animacja przedstawiająca zmiany temperatury w pomieszczeniu. Animacja jest zapisywana w formie pliku GIF.

## Autor
Mateusz Szydełko
