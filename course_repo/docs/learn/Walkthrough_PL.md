# RL Foundations for LLMs — Przewodnik dla instruktora

## Przegląd sesji

**Czas trwania:** Około 45–60 minut  
**Format:** Praktyczna demonstracja terminalowa z interaktywną dyskusją  
**Cele uczenia:** Ustanowienie fundamentalnego frameworku konceptualnego przed kontaktem studentów z kodem

---

## Przygotowanie przed sesją

### Wymagania techniczne

Zweryfikuj spełnienie następujących warunków przed rozpoczęciem sesji:

```bash
cd path/to/rl-foundations-llm
poetry run python -m course.eval --help
poetry run python -m course.selection_demo --help
poetry run python -m course.bandit_train --help
```

Jeśli którekolwiek polecenie się nie powiedzie, upewnij się, że konfiguracja środowiska Python jest kompletna.

### Przygotowanie poznawcze

Studenci często przychodzą z następującym modelem mentalnym:

> "Po prostu poinstruujemy model, żeby się poprawił i zmierzymy wyniki."

Celem sesji jest zastąpienie tej intuicji następującym frameworkiem:

> "Pomiar, selection i learning stanowią trzy odrębne operacje z fundamentalnie różnymi mechanizmami i implikacjami."

---

## Przebieg sesji

### Segment 1: Loop A — Pomiar jako instrumentacja naukowa (15 minut)

#### Podstawa koncepcyjna

**Otwierające stwierdzenie instruktora:**

> "Przed jakąkolwiek dyskusją o poprawie musimy ustalić, co mierzymy i jak. Loop A dotyczy wyłącznie pomiaru."

**Kluczowe punkty koncepcyjne:**

- Scorer funkcjonuje jako instrument naukowy, nie jedynie funkcja użyteczności
- Gdy metryki zmieniają się między przebiegami, albo policy się zmieniła, ALBO warunki pomiaru się zmieniły
- Obowiązkiem eksperymentatora jest określenie, które z tych dwóch opcji zachodzi

#### Demonstracja na żywo

Wykonaj następującą sekwencję:

```bash
poetry run python -m course.eval \
  --dataset data/datasets/math_dev.jsonl \
  --completions data/rollouts/frozen_rollouts_dev.jsonl \
  --outdir runs/demo_eval_clean
```

Przejdź do katalogu wyjściowego i zbadaj artefakty:

```bash
cat runs/demo_eval_clean/manifest.json
head -5 runs/demo_eval_clean/results.jsonl
cat runs/demo_eval_clean/summary.json
```

**Punkty do dyskusji:**

- `manifest.json` zapisuje warunki eksperymentalne (analogicznie do dokumentacji laboratoryjnej)
- `results.jsonl` zawiera wyniki per przykład
- `summary.json` dostarcza statystyki zagregowane

#### Demonstracja perturbacji

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

**Oczekiwany wynik:** REJECT z informacjami diagnostycznymi wskazującymi na naruszenie Locked Room.

**Wyjaśnienie instruktora:**

> "Bramka odrzuciła to porównanie. Dlaczego? Ponieważ warunki eksperymentalne różniły się. Metryka się zmieniła, ale nie z powodu jakiejkolwiek zmiany w policy. To jest dokładnie to pomylenie, któremu musimy zapobiec."

---

### Segment 2: Loop B — Selection bez learning (15 minut)

#### Podstawa koncepcyjna

**Stwierdzenie instruktora:**

> "Loop B demonstruje, że pozorna poprawa jest osiągalna bez jakiejkolwiek modyfikacji parametrów modelu."

**Kluczowe punkty koncepcyjne:**

- Best-of-N selection wybiera spośród istniejących próbek
- Leżący u podstaw rozkład prawdopodobieństwa pozostaje niezmieniony
- Poprawione wyniki wynikają ze zwiększonego wydatku obliczeniowego, nie wyuczonego zachowania

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

> "Pass rate wzrosło z N=1 do N=4. Czy model się czegokolwiek nauczył? Nie. Po prostu zbadaliśmy więcej próbek i wybraliśmy najlepszą. Rozkład prawdopodobieństwa, który wygenerował te próbki, jest niezmieniony."

#### Interaktywna weryfikacja

Postaw studentom następujące pytanie:

> "Jeśli zresetujemy i uruchomimy N=4 selection ponownie, czy uzyskamy identyczne wyniki?"

Wykonaj weryfikację:

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

### Segment 3: Loop C — Learning modyfikuje rozkład (15 minut)

#### Podstawa koncepcyjna

**Stwierdzenie instruktora:**

> "Loop C jest miejscem, gdzie rozkłady prawdopodobieństwa się zmieniają. To jest faktyczne uczenie."

**Kluczowe punkty koncepcyjne:**

- Metody policy gradient modyfikują rozkład prawdopodobieństwa nad akcjami
- Kierunek aktualizacji jest determinowany przez advantage: reward minus baseline
- Jest to fundamentalnie inne niż selection—sam generator się zmienia

#### Demonstracja na żywo

Wykonaj trening bandity w trybie verbose:

```bash
poetry run python -m course.bandit_train --steps 30 --seed 0 --lr 0.5 --baseline --slow --outdir runs/demo_bandit_slow
```

**Analiza instruktora:**

Zwróć uwagę na wyjście krok po kroku:

> "Obserwuj wpisy logu. Dla każdego kroku zwróć uwagę: która akcja została spróbkowana, jaka nagroda została otrzymana, jaka była wartość baseline i jaki jest wynikowy advantage."

> "Gdy advantage jest dodatni, prawdopodobieństwo tej akcji wzrasta. Gdy advantage jest ujemny, prawdopodobieństwo maleje. To jest gradient REINFORCE."

#### Demonstracja perturbacji

Wykonaj trening z odwróconym learning rate:

```bash
poetry run python -m course.bandit_train --steps 30 --seed 0 --lr -0.5 --baseline --slow --outdir runs/demo_bandit_negative
```

**Analiza instruktora:**

> "Learning rate jest ujemny. Co się dzieje? Kierunek aktualizacji się odwraca. Akcje z dodatnim advantage mają swoje prawdopodobieństwa zmniejszane zamiast zwiększane. System uczy się odwrotności zamierzonego celu."

---

### Segment 4: Integracja i synteza (10–15 minut)

#### Framework podsumowujący

Przedstaw następującą taksonomię:

| Loop | Główny mechanizm | Co się zmienia | Co pozostaje stałe |
|------|------------------|----------------|---------------------|
| A | Pomiar | Zapisane metryki | Policy, środowisko, instrument |
| B | Selection | Obserwowane wyjścia | Leżący u podstaw rozkład |
| C | Learning | Rozkład prawdopodobieństwa | (Potencjalnie nic—to jest fundamentalna zmiana) |

#### Zasada Locked Room

**Stwierdzenie instruktora:**

> "Przed porównaniem jakichkolwiek dwóch przebiegów eksperymentalnych zweryfikuj, że następujące warunki są identyczne: dataset, wersja scorera, konfiguracja próbkowania. Jeśli którykolwiek się różni, porównanie jest metodologicznie nieważne."

Zademonstruj przez badanie manifestów:

```bash
cat runs/demo_eval_clean/manifest.json | jq '.dataset_fingerprint, .scorer_version'
cat runs/demo_eval_modified/manifest.json | jq '.dataset_fingerprint, .scorer_version'
```

---

## Typowe błędne przekonania studentów

### Błędne przekonanie: "Wyższy pass@N oznacza, że model się poprawił"

**Korekta:** Poprawa pass@N może wskazywać na selection (Loop B) zamiast learning (Loop C). Zweryfikuj przez zbadanie, czy parametry policy się zmieniły.

### Błędne przekonanie: "Scorer to po prostu funkcja oceniająca"

**Korekta:** Scorer definiuje cel optymalizacji. Źle wyspecyfikowane scorery prowadzą do eksploatacji nagrody. Scorer musi być traktowany jako kod produkcyjny.

### Błędne przekonanie: "Możemy porównać dowolne dwa przebiegi z tą samą metryką"

**Korekta:** Porównania są ważne tylko przy warunkach Locked Room. Różne datasety, wersje scorera lub konfiguracje próbkowania unieważniają porównanie.

---

## Zakończenie sesji

### Wnioski dla studentów

Upewnij się, że studenci odchodzą z rozumieniem:

1. Trzy pętle (A/B/C) reprezentują kategorycznie różne operacje
2. Warunki Locked Room są warunkami wstępnymi dla ważnego porównania eksperymentalnego
3. Scorer funkcjonuje jako specyfikacja, jakiego zachowania poszukujemy

### Przejście do pracy projektowej

**Końcowe stwierdzenie instruktora:**

> "Ćwiczenia projektowe będą wymagać od was celowego naruszenia tych zasad, obserwacji wynikających awarii, a następnie zaimplementowania korekt. Cykl Build-Sabotage-Reflect-Repair jest zaprojektowany, aby przekształcić teoretyczne zrozumienie w operacyjną kompetencję."

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
