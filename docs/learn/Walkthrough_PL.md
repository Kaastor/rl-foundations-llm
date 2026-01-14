# RL Foundations for LLMs — Przewodnik dla instruktora

## Przegląd sesji

**Czas trwania:** ok. 45–60 minut  
**Format:** praktyczna demonstracja w terminalu + interaktywna dyskusja  
**Cele dydaktyczne:** zbudować wspólne ramy pojęciowe, zanim studenci wejdą w kod

---

## Przygotowanie przed sesją

### Wymagania techniczne

Przed rozpoczęciem upewnij się, że te polecenia działają:

```bash
cd path/to/rl-foundations-llm
poetry run python -m course.eval --help
poetry run python -m course.selection_demo --help
poetry run python -m course.bandit_train --help
```

Jeśli któreś polecenie się wywraca, najpierw napraw środowisko Pythona (Poetry/venv, wersja Pythona, zależności).

### Przygotowanie poznawcze

Studenci często przychodzą z intuicją w stylu:

> "Po prostu powiemy modelowi, żeby był lepszy, i zmierzymy wyniki."

Celem sesji jest zastąpić tę intuicję następującą ramą:

> "Pomiar (Loop A), selekcja (Loop B) i uczenie (Loop C) to trzy różne operacje — mają inne mechanizmy i inne typowe pułapki."

---

## Przebieg sesji

### Segment 1: Loop A — Pomiar jako instrument naukowy (15 minut)

#### Podstawa koncepcyjna

**Otwierające stwierdzenie instruktora:**

> "Zanim zaczniemy mówić o poprawie czegokolwiek, musimy ustalić, co mierzymy i czym mierzymy. Loop A to czysty pomiar."

**Kluczowe punkty koncepcyjne:**

- Weryfikator/oceniacz (**scorer**) działa jak instrument naukowy, a nie tylko „funkcja celu”.
- Gdy metryka zmienia się między przebiegami, albo zmieniła się **policy** (czyli model/strategia generowania), albo zmieniły się warunki pomiaru.
- Zadaniem eksperymentatora jest ustalić *która* z tych rzeczy zaszła — na podstawie artefaktów, a nie przeczucia.

#### Demonstracja na żywo

Wykonaj następującą sekwencję:

```bash
poetry run python -m course.eval   --dataset data/datasets/math_dev.jsonl   --completions data/rollouts/frozen_rollouts_dev.jsonl   --outdir runs/demo_eval_clean
```

Otwórz artefakty:

```bash
cat runs/demo_eval_clean/manifest.json
head -5 runs/demo_eval_clean/results.jsonl
cat runs/demo_eval_clean/summary.json
```

**Punkty do dyskusji:**

- `manifest.json` zapisuje warunki eksperymentalne (jak notatnik laboratoryjny: „co dokładnie uruchomiłem?”)
- `results.jsonl` zawiera wynik dla każdego przykładu
- `summary.json` daje metryki zagregowane

#### Perturbacja (kontrolowana zmiana)

Wprowadź kontrolowaną modyfikację do zbioru danych:

```bash
cp data/datasets/math_dev.jsonl /tmp/math_dev_modified.jsonl
# Zmodyfikuj jeden expected_answer w pliku tymczasowym

poetry run python -m course.eval   --dataset /tmp/math_dev_modified.jsonl   --completions data/rollouts/frozen_rollouts_dev.jsonl   --outdir runs/demo_eval_modified
```

Wykonaj porównanie bramką:

```bash
poetry run python -m course.gate   --baseline runs/demo_eval_clean   --candidate runs/demo_eval_modified   --min-delta 0.00
```

**Oczekiwany wynik:** REJECT z informacjami diagnostycznymi wskazującymi na naruszenie Zasady Locked Room.

**Wyjaśnienie instruktora:**

> "Bramka odrzuciła porównanie, bo zmieniły się warunki eksperymentu. Metryka drgnęła bez zmiany policy. To dokładnie ten typ samooszukiwania, który chcemy umieć wyłapywać."

---

### Segment 2: Loop B — Selekcja bez uczenia (15 minut)

#### Podstawa koncepcyjna

**Stwierdzenie instruktora:**

> "Loop B pokazuje, że pozorna poprawa jest możliwa bez jakiejkolwiek zmiany parametrów modelu."

**Kluczowe punkty koncepcyjne:**

- Best-of-N wybiera najlepszą próbkę spośród **N** wygenerowanych odpowiedzi.
- Rozkład policy się nie zmienia — nadal próbkujesz z tego samego modelu.
- Poprawa wynika z większego kosztu obliczeń podczas inferencji, a nie z uczenia.

#### Demonstracja na żywo

Wykonaj selekcję z N=1:

```bash
poetry run python -m course.selection_demo   --dataset data/datasets/math_dev.jsonl   --samples data/rollouts/selection_pack_dev.jsonl   --n 1   --outdir runs/demo_sel_n1
```

Wykonaj selekcję z N=4:

```bash
poetry run python -m course.selection_demo   --dataset data/datasets/math_dev.jsonl   --samples data/rollouts/selection_pack_dev.jsonl   --n 4   --outdir runs/demo_sel_n4
```

Porównaj wyniki:

```bash
cat runs/demo_sel_n1/summary.json
cat runs/demo_sel_n4/summary.json
```

**Analiza instruktora:**

> "Pass rate rośnie z N=1 do N=4. Czy model się czegoś nauczył? Nie. Po prostu obejrzeliśmy więcej próbek i wybraliśmy najlepszą — z tego samego rozkładu."

#### Weryfikacja deterministyczności

Zapytaj studentów:

> "Jeśli uruchomimy selekcję N=4 drugi raz na tych samych wejściach, czy dostaniemy identyczny wynik?"

Sprawdź:

```bash
poetry run python -m course.selection_demo   --dataset data/datasets/math_dev.jsonl   --samples data/rollouts/selection_pack_dev.jsonl   --n 4   --outdir runs/demo_sel_n4_repeat

diff runs/demo_sel_n4/results.jsonl runs/demo_sel_n4_repeat/results.jsonl
```

**Oczekiwany wynik:** brak różnicy. Selekcja jest deterministyczna przy identycznych danych wejściowych.

---

### Segment 3: Loop C — Uczenie zmienia rozkład (15 minut)

#### Podstawa koncepcyjna

**Stwierdzenie instruktora:**

> "Loop C to moment, w którym zmienia się sam generator: rozkład prawdopodobieństwa policy."

**Kluczowe punkty koncepcyjne:**

- Metody policy gradient modyfikują rozkład nad akcjami.
- Kierunek aktualizacji wyznacza **advantage**: `reward − baseline` (czyli „o ile lepiej/gorzej niż typowo”).
- To jakościowo inne niż selekcja: tutaj zmieniają się parametry, więc zmienia się zachowanie modelu.

#### Demonstracja na żywo

Wykonaj trening bandyty w trybie szczegółowym:

```bash
poetry run python -m course.bandit_train --steps 30 --seed 0 --lr 0.5 --baseline --slow --outdir runs/demo_bandit_slow
```

**Analiza instruktora:**

Zwróć uwagę na wyjście krok po kroku:

> "W każdym kroku sprawdź: jaka akcja została spróbkowana, jaka była nagroda, jaka była wartość baseline i jaki jest znak advantage."

> "Gdy advantage jest dodatni, prawdopodobieństwo tej akcji rośnie. Gdy advantage jest ujemny, prawdopodobieństwo maleje. To jest intuicja za gradientem REINFORCE."

#### Sabotaż: ujemny learning rate

Wykonaj trening z odwróconym learning rate:

```bash
poetry run python -m course.bandit_train --steps 30 --seed 0 --lr -0.5 --baseline --slow --outdir runs/demo_bandit_negative
```

**Analiza instruktora:**

> "Learning rate jest ujemny. Co się dzieje? Kierunek aktualizacji się odwraca. Akcje z dodatnim advantage mają swoje prawdopodobieństwa *zmniejszane* zamiast zwiększane. System uczy się odwrotności zamierzonego celu."

---

### Segment 4: Integracja i synteza (10–15 minut)

#### Podsumowanie ramy

Przedstaw następującą taksonomię:

| Loop | Główny mechanizm | Co się zmienia | Co pozostaje stałe |
|------|------------------|----------------|---------------------|
| A | Pomiar | Zapisane metryki | Policy, środowisko, instrument pomiarowy (scorer) |
| B | Selekcja | Wybrane wyjście (spośród próbek) | Leżący u podstaw rozkład policy |
| C | Uczenie | Rozkład prawdopodobieństwa | (Potencjalnie nic — to jest fundamentalna zmiana) |

#### Zasada Locked Room

**Stwierdzenie instruktora:**

> "Zanim porównasz dwa przebiegi, sprawdź, że identyczne są: zbiór danych, prompt (jeśli dotyczy), wersja scorera i konfiguracja próbkowania. Jeśli coś się różni, porównanie jest metodologicznie nieważne."

Pokaż na manifestach:

```bash
cat runs/demo_eval_clean/manifest.json | jq '.dataset_fingerprint, .scorer_version'
cat runs/demo_eval_modified/manifest.json | jq '.dataset_fingerprint, .scorer_version'
```

---

## Typowe błędy studentów

### Błędne przekonanie: "Wyższy pass@N oznacza, że model się poprawił"

**Korekta:** pass@N może rosnąć w Loop B bez uczenia. Zweryfikuj, czy parametry policy faktycznie się zmieniły.

### Błędne przekonanie: "Scorer to po prostu funkcja oceniająca"

**Korekta:** scorer/weryfikator definiuje cel optymalizacji. Źle wyspecyfikowany scorer zachęca do „reward hackingu”. Traktuj go jak kod produkcyjny i testuj regresje.

### Błędne przekonanie: "Możemy porównać dowolne dwa przebiegi z tą samą metryką"

**Korekta:** porównania mają sens tylko, jeśli spełnione są warunki Locked Room. Inny dataset, inna wersja scorera lub inne próbkowanie unieważniają wnioski.

---

## Zakończenie sesji

### Wnioski dla studentów

Upewnij się, że studenci wychodzą z rozumieniem:

1. Trzy pętle (A/B/C) to kategorycznie różne operacje.
2. Zasada Locked Room jest warunkiem sensownego porównywania eksperymentów.
3. Scorer pełni rolę specyfikacji tego, jakiego zachowania szukamy.

### Przejście do pracy projektowej

**Końcowe stwierdzenie instruktora:**

> "Ćwiczenia projektowe wymagają celowego naruszenia tych zasad, obserwacji awarii i wdrożenia poprawek. Cykl Build/Sabotage/Reflect/Repair ma zamienić teorię w praktyczną kompetencję."

---

## Dodatek: Polecenia diagnostyczne

### Weryfikacja środowiska

```bash
poetry run python --version
poetry run python -c "import course; print('Course package available')"
ls data/datasets/
ls data/rollouts/
```

### Szybki reset

```bash
rm -rf runs/demo_*
```

### Inspekcja artefaktów

```bash
poetry run python -m course.inspect_run --run <run_dir> --top-k 5 --show 2
poetry run python -m course.inspect_run --run <run_dir> --only-fails --top-k 5 --show 2
```