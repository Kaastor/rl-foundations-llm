# RL Foundations for LLMs — Przewodnik dla instruktora

## Przegląd sesji

**Czas trwania:** Około 45–60 minut  
**Format:** Praktyczna demonstracja terminalowa z interaktywną dyskusją  
**Cele:** wspólne ramy pojęciowe przed pracą z kodem

---

## Przygotowanie przed sesją

### Wymagania techniczne

Sprawdź przed sesją:

```bash
cd path/to/rl-foundations-llm
poetry run python -m course.eval --help
poetry run python -m course.selection_demo --help
poetry run python -m course.bandit_train --help
```

Jeśli coś nie działa, napraw konfigurację środowiska.

### Przygotowanie poznawcze

Studenci często zakładają:

> "Po prostu poinstruujemy model, żeby się poprawił i zmierzymy wyniki."

Celem jest zamiana tej intuicji na:

> "Pomiar, selection i learning stanowią trzy odrębne operacje z fundamentalnie różnymi mechanizmami i implikacjami."

---

## Przebieg sesji

### Segment 1: Loop A — Evaluation (pomiar) jako instrumentacja naukowa (15 minut)

#### Podstawa koncepcyjna

**Otwierające stwierdzenie instruktora:**

> "Przed jakąkolwiek dyskusją o poprawie musimy ustalić, co mierzymy i jak. Loop A dotyczy wyłącznie pomiaru."

**Kluczowe punkty koncepcyjne:**

- Scorer to instrument pomiarowy, nie funkcja użyteczności
- Gdy metryki się zmieniają, albo zmieniła się policy, albo warunki pomiaru
- Twoim zadaniem jest wskazać które

#### Demonstracja na żywo

Wykonaj następującą sekwencję:

```bash
poetry run python -m course.eval \
  --dataset data/datasets/math_dev.jsonl \
  --completions data/rollouts/frozen_rollouts_dev.jsonl \
  --outdir runs/demo_eval_clean
```

Otwórz artefakty:

```bash
cat runs/demo_eval_clean/manifest.json
head -5 runs/demo_eval_clean/results.jsonl
cat runs/demo_eval_clean/summary.json
```

**Dyskusja:**

- `manifest.json` zapisuje warunki eksperymentalne (analogicznie do dokumentacji laboratoryjnej)
- `results.jsonl` zawiera wyniki per przykład
- `summary.json` dostarcza statystyki zagregowane

#### Sabotage (kontrolowana zmiana)

Wprowadź kontrolowaną modyfikację do datasetu:

```bash
cp data/datasets/math_dev.jsonl /tmp/math_dev_modified.jsonl
# Zmodyfikuj jeden expected_answer w pliku tymczasowym

poetry run python -m course.eval \
  --dataset /tmp/math_dev_modified.jsonl \
  --completions data/rollouts/frozen_rollouts_dev.jsonl \
  --outdir runs/demo_eval_modified
```

Wykonaj porównanie bramką:

```bash
poetry run python -m course.gate \
  --baseline runs/demo_eval_clean \
  --candidate runs/demo_eval_modified \
  --min-delta 0.00
```

**Oczekiwany wynik:** REJECT z informacjami diagnostycznymi wskazującymi na naruszenie Zasady Locked Room.

**Wyjaśnienie instruktora:**

> "Bramka odrzuciła porównanie, bo zmieniły się warunki. Metryka drgnęła bez zmiany policy. To dokładnie pomyłka, której unikamy."

---

### Segment 2: Loop B — Selection (bez learningu) (15 minut)

#### Podstawa koncepcyjna

**Stwierdzenie instruktora:**

> "Loop B demonstruje, że pozorna poprawa jest osiągalna bez jakiejkolwiek modyfikacji parametrów modelu."

**Kluczowe punkty koncepcyjne:**

- Best-of-N wybiera najlepszą próbkę
- Rozkład policy pozostaje niezmieniony
- To koszt obliczeń, nie learning

#### Demonstracja na żywo

Wykonaj selection z N=1:

```bash
poetry run python -m course.selection_demo \
  --dataset data/datasets/math_dev.jsonl \
  --samples data/rollouts/selection_pack_dev.jsonl \
  --n 1 \
  --outdir runs/demo_sel_n1
```

Wykonaj selection z N=4:

```bash
poetry run python -m course.selection_demo \
  --dataset data/datasets/math_dev.jsonl \
  --samples data/rollouts/selection_pack_dev.jsonl \
  --n 4 \
  --outdir runs/demo_sel_n4
```

Porównaj wyniki:

```bash
cat runs/demo_sel_n1/summary.json
cat runs/demo_sel_n4/summary.json
```

**Analiza instruktora:**

> "Pass rate rośnie z N=1 do N=4. Model się nie uczy; wybierasz najlepszą próbkę z tego samego rozkładu."

#### Weryfikacja deterministyczności

Zapytaj studentów:

> "Jeśli zresetujemy i uruchomimy N=4 selection ponownie, czy uzyskamy identyczne wyniki?"

Sprawdź:

```bash
poetry run python -m course.selection_demo \
  --dataset data/datasets/math_dev.jsonl \
  --samples data/rollouts/selection_pack_dev.jsonl \
  --n 4 \
  --outdir runs/demo_sel_n4_repeat

diff runs/demo_sel_n4/results.jsonl runs/demo_sel_n4_repeat/results.jsonl
```

**Oczekiwany wynik:** Brak różnicy. Selection jest deterministyczne przy identycznych danych wejściowych.

---

### Segment 3: Loop C — Learning (zmiana rozkładu) (15 minut)

#### Podstawa koncepcyjna

**Stwierdzenie instruktora:**

> "Loop C jest miejscem, gdzie rozkłady prawdopodobieństwa się zmieniają. To jest faktyczne uczenie."

**Kluczowe punkty koncepcyjne:**

- Policy gradient modyfikuje rozkład nad akcjami
- Kierunek aktualizacji wyznacza advantage: reward minus baseline
- To inne niż selection—zmienia się generator

#### Demonstracja na żywo

Wykonaj trening bandity w trybie verbose:

```bash
poetry run python -m course.bandit_train --steps 30 --seed 0 --lr 0.5 --baseline --slow --outdir runs/demo_bandit_slow
```

**Analiza instruktora:**

Zwróć uwagę na wyjście krok po kroku:

> "Obserwuj wpisy logu. Dla każdego kroku zwróć uwagę: która akcja została spróbkowana, jaka nagroda została otrzymana, jaka była wartość baseline i jaki jest wynikowy advantage."

> "Gdy advantage jest dodatni, prawdopodobieństwo tej akcji wzrasta. Gdy advantage jest ujemny, prawdopodobieństwo maleje. To jest gradient REINFORCE."

#### Sabotage: ujemny learning rate

Wykonaj trening z odwróconym learning rate:

```bash
poetry run python -m course.bandit_train --steps 30 --seed 0 --lr -0.5 --baseline --slow --outdir runs/demo_bandit_negative
```

**Analiza instruktora:**

> "Learning rate jest ujemny. Co się dzieje? Kierunek aktualizacji się odwraca. Akcje z dodatnim advantage mają swoje prawdopodobieństwa zmniejszane zamiast zwiększane. System uczy się odwrotności zamierzonego celu."

---

### Segment 4: Integracja i synteza (10–15 minut)

#### Podsumowanie

Przedstaw następującą taksonomię:

| Loop | Główny mechanizm | Co się zmienia | Co pozostaje stałe |
|------|------------------|----------------|---------------------|
| A | Pomiar | Zapisane metryki | Policy, środowisko, scorer |
| B | Selection | Obserwowane wyjścia | Leżący u podstaw rozkład |
| C | Learning | Rozkład prawdopodobieństwa | (Potencjalnie nic—to jest fundamentalna zmiana) |

#### Zasada Locked Room

**Stwierdzenie instruktora:**

> "Przed porównaniem dwóch przebiegów zweryfikuj, że identyczne są: dataset, prompt, wersja scorera i konfiguracja próbkowania. Jeśli coś się różni, porównanie jest metodologicznie nieważne."

Pokaż na manifestach:

```bash
cat runs/demo_eval_clean/manifest.json | jq '.dataset_fingerprint, .scorer_version'
cat runs/demo_eval_modified/manifest.json | jq '.dataset_fingerprint, .scorer_version'
```

---

## Typowe błędy studentów

### Błędne przekonanie: "Wyższy pass@N oznacza, że model się poprawił"

**Korekta:** Pass@N może rosnąć w Loop B bez learningu. Zweryfikuj, czy parametry policy się zmieniły.

### Błędne przekonanie: "Scorer to po prostu funkcja oceniająca"

**Korekta:** Scorer definiuje cel optymalizacji. Źle wyspecyfikowane scorery prowadzą do eksploatacji nagrody. Scorer musi być traktowany jako kod produkcyjny.

### Błędne przekonanie: "Możemy porównać dowolne dwa przebiegi z tą samą metryką"

**Korekta:** Porównania są ważne tylko przy Zasadzie Locked Room. Różne datasety, wersje scorera lub konfiguracje próbkowania unieważniają porównanie.

---

## Zakończenie sesji

### Wnioski dla studentów

Upewnij się, że studenci odchodzą z rozumieniem:

1. Trzy pętle (A/B/C) reprezentują kategorycznie różne operacje
2. Zasada Locked Room jest warunkiem ważnego porównania eksperymentalnego
3. Scorer funkcjonuje jako specyfikacja, jakiego zachowania poszukujemy

### Przejście do pracy projektowej

**Końcowe stwierdzenie instruktora:**

> "Ćwiczenia projektowe wymagają celowego naruszenia tych zasad, obserwacji awarii i wdrożenia poprawek. Cykl Build/Sabotage/Reflect/Repair zamienia teorię w kompetencję."

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
