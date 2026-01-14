# RL Foundations for LLMs — Mapa projektu Mastery Track

---

## Przegląd projektu

**Tytuł projektu:** Verifier-Driven Math QA — Konstrukcja, celowe zakłócenie i systematyczne wzmocnienie

Ta ścieżka jest zaprojektowana, aby wymusić zrozumienie poprzez powtarzane zastosowanie ustrukturyzowanego cyklu eksperymentalnego:

1. **Build (Ustanowienie linii bazowej):** Ustanowienie czystej, deterministycznej linii bazowej, nad którą student sprawuje pełną kontrolę.
2. **Sabotage (Kontrolowana perturbacja):** Wprowadzenie dokładnie jednej jawnej modyfikacji, która narusza podstawową zasadę.
3. **Reflect (Analiza forensyczna):** Obserwacja wynikającej awarii poprzez badanie artefaktów (`manifest.json`, `summary.json`, `results.jsonl`) zamiast oceny intuicyjnej.
4. **Repair and Lock (Korekta mechanizmu):** Korekta leżącego u podstaw mechanizmu i zakodowanie poprawki jako testów i bramek, aby zapobiec regresji.

Kroki Sabotage stanowią zasadniczy komponent pedagogiczny tej ścieżki i nie mogą być pominięte.

---

## Dlaczego to istnieje

Procesy trenowania i ewaluacji LLM łatwo źle zinterpretować. Metryki mogą się zmieniać, bo zmieniłeś scorer, dataset albo regułę selekcji, a nie sam model. Ta ścieżka uczy dyscypliny pomiaru: zbuduj czystą linię bazową, wprowadź jedną kontrolowaną zmianę, obejrzyj artefakty i zablokuj poprawkę. Celem są wiarygodne dowody, nie tylko wyższe liczby.

## Gdzie to ważne w cyklu życia LLM

- Kuracja danych i ewaluacja: utrzymuj stałe dataset, prompt i scorer, aby porównania były ważne.
- Specyfikacja nagrody i weryfikacja: scorer i modele nagrody to instrumenty produkcyjne, muszą być deterministyczne i wersjonowane.
- Selection w czasie inferencji: Best-of-N może poprawiać wyniki bez uczenia, więc atrybucja musi być jawna.
- Trening (SFT/RLHF/DPO): credit assignment, wariancja i ograniczenia KL decydują, czy uczenie jest stabilne i sensowne.
- Wdrożenie i monitoring: testy regresji, golden sety i bramki zapobiegają cichej degradacji metryk lub reward hacking.
- Analiza incydentów: forensyka oparta o artefakty pozwala dowieść, co się zmieniło i dlaczego.

---

## Uwaga o zakresie (praktyka)

Te zadania **nie** trenują ani nie próbkują realnego LLM. Praca na "policy" jest syntetyczna, aby mechanika była jasna, więc transfer do systemów LLM jest bardziej koncepcyjny niż praktyczny.

Jeśli chcesz praktycznego ugruntowania, opcjonalnym rozszerzeniem jest podpięcie LLM, wygenerowanie nowych rolloutów i ponowne uruchomienie Loop A/B przy zachowaniu dyscypliny pomiaru.

### Opcjonalne rozszerzenie: rollouty z Groq (minimalna zmiana)

Wpisz `GROQ_API_KEY` do pliku `.env`, a następnie:

```bash
# Loop A: próbkowanie uzupełnień i ewaluacja
python -m course.rollout_sample \
  --dataset data/datasets/math_dev.jsonl \
  --outdir runs/rollouts_groq_dev

python -m course.eval \
  --dataset data/datasets/math_dev.jsonl \
  --completions runs/rollouts_groq_dev/completions.jsonl \
  --outdir runs/l0_build_eval_groq

# Loop B: wygenerowanie selection pack (N próbek na prompt)
python -m course.rollout_sample \
  --dataset data/datasets/math_dev.jsonl \
  --n 4 \
  --format selection \
  --outdir runs/rollouts_groq_sel

python -m course.selection_demo \
  --dataset data/datasets/math_dev.jsonl \
  --samples runs/rollouts_groq_sel/selection_pack.jsonl \
  --n 4 \
  --outdir runs/l0_5_groq_sel
```

Domyślny model to mały model Groq; możesz go nadpisać przez `--model` albo `GROQ_MODEL`.

---

## Reguły fundamentalne

Te zasady powinny zostać zinternalizowane przed rozpoczęciem pracy praktycznej.

### Trzy pętle operacyjne

Ta terminologia powinna być stosowana konsekwentnie w całej dokumentacji i dyskusji:

* **Loop A (Evaluation):** Wyłącznie operacje pomiarowe (zamrożone wejścia, deterministyczny scorer).
* **Loop B (Selection):** Wybór best-of-N (obserwowalne zachowanie się poprawia; leżący u podstaw rozkład policy pozostaje niezmieniony).
* **Loop C (Learning):** Aktualizacje parametrów policy (rozkłady prawdopodobieństwa się przesuwają → zachowanie generatywne się zmienia).

### Zasada Locked Room (poprawne porównania eksperymentalne)

Przed porównaniem metryk między dwoma przebiegami eksperymentalnymi, następujące warunki muszą być zweryfikowane jako identyczne:

* Podział datasetu (identyczny fingerprint pliku zapisany w `manifest.json`)
* Tekst promptu (jeśli dotyczy projektu eksperymentalnego)
* **Nazwa i wersja** scorera
* Konfiguracja próbkowania (jeśli stosowane jest próbkowanie stochastyczne)

Jeśli którykolwiek z tych warunków różni się między przebiegami, środowisko eksperymentalne lub instrument pomiarowy został zmieniony. Takie porównania są metodologicznie nieważne.

### Specyfikacja nagrody jako kod produkcyjny

Scorer funkcjonuje jako instrument pomiarowy i dlatego musi wykazywać następujące właściwości:
**deterministyczny**, **totalny (nigdy nie kończy się anormalnie)**, **wyjaśnialny**, **obliczeniowo wydajny**, **wersjonowany**.

Funkcja `score()` powinna być traktowana z tym samym rygorem co oprogramowanie produkcyjne.

---

## Dyscyplina workflow

Następujące praktyki są silnie zalecane:

Utwórz dedykowany branch i commituj inkrementalnie, aby utrzymać weryfikowalny zapis modyfikacji.

```bash
git checkout -b mastery-track
pytest -q
```

Dla każdego poziomu wykonaj następującą sekwencję commitów:

* Commit stanu **Build** (ustanowienie linii bazowej).
* Commit modyfikacji **Sabotage** (mimo że wprowadza celową awarię).
* Commit **Repair** (poprawka i środki zapobiegawcze).

---

## Wymagania wstępne i szybkie sprawdzenia

Przed Level 0 zweryfikuj środowisko i entrypointy:

```bash
python --version   # musi być 3.10+
python -m course.eval --help
python -m course.selection_demo --help
python -m course.bandit_train --help
```

Jeśli używasz Poetry, uruchamiaj polecenia przez `poetry run ...` albo aktywuj `poetry shell`.

## Zacznij tutaj (nowi studenci)

Jeśli jesteś zupełnie nowy, przeczytaj raz „Reguły fundamentalne", a potem przejdź do sekcji „Zadania zaczynają się tutaj" poniżej. Każdy poziom ma te same cztery ruchy:

1) Build: zbuduj czystą linię bazową, żeby mieć wiarygodny punkt odniesienia.
2) Sabotage: celowo zmień jedną zmienną, aby zobaczyć tryb awarii.
3) Reflect: oprzyj wnioski na artefaktach, nie na intuicji.
4) Repair and lock: napraw mechanizm i zablokuj go testami/notatkami, aby nie wrócił.

Każdy krok niżej ma krótkie wyjaśnienie „po co". Jeśli się zgubisz, wróć do sekcji „Podstawa koncepcyjna" na początku poziomu.

### Pliki zadań

Pliki zadań zawierają TODO, które należy uzupełnić. Testy są szkieletem i będą się wywracać, dopóki nie ukończysz zadań.

### Najczęstsze pułapki (z opisem)

* **Python < 3.10:** błąd na `dataclass(slots=True)` zanim zaczniesz zadania.
* **Naruszenie Locked Room:** porównanie przebiegów z innym datasetem/scorerem/próbkowaniem; manifesty się nie zgadzają, więc porównanie jest nieważne.
* **Niedeterministyczny selection:** losowy tie-break powoduje dryf wyników i niestabilne testy; hashe `results.jsonl` się różnią.
* **Brak cofnięcia sabotażu:** pozostawiona zmiana sabotuje naprawę i kolejne poziomy.
* **Ślepota na artefakty:** czytanie tylko `summary.json` i ignorowanie `results.jsonl`/`manifest.json` ukrywa tryby awarii.
* **Mylenie formatu z matematyką:** traktowanie błędów parsowania jak błędów merytorycznych prowadzi do złych poprawek.
* **Błędny znak advantage:** ujemny advantage powinien zmniejszać prawdopodobieństwo akcji; odwrócenie daje anty‑uczenie.
* **Brak credit assignment:** aktualizowanie tylko ostatniego kroku utrzymuje losowość wczesnych decyzji i hamuje naukę.
* **Ignorowanie KL:** maksymalizacja nagrody bez kontroli dywergencji prowadzi do kruchych, „odjechanych" polityk.
* **Niezgodne JSONL/ID:** niepasujące `id` w danych i rolloutach powoduje braki próbek i ciche porażki.
* **Brak matplotlib:** wykres KL się nie pojawi; CSV i podsumowania nadal się zapisują.

---

## Checklista artefaktów (co sprawdzać)

Każdy przebieg zostawia dowody; używaj ich w README i porównaniach:

* `manifest.json` — wejścia + SHA256, nazwa/wersja scorera (dowód Locked Room).
* `results.jsonl` — rekordy per-przykład i kody outcome.
* `summary.json` — metryki zagregowane.
* `summary.md` — podsumowanie dla ludzi.

Niektóre zadania dodają extra artefakty (np. `traj.jsonl`, `kl_tradeoff.csv`).

Jeśli chcesz udowodnić deterministyczność, haszuj `results.jsonl` w powtórzonych przebiegach.

---

## Samodzielna walidacja (sprawdzenie)

Jeśli poniższe sprawdzenia przechodzą, najpewniej masz poprawne rozwiązania. Jeśli popełniłeś błąd, co najmniej jeden test powinien się wyłożyć albo wynik/artefakty będą nieoczekiwane.

### 1) Uruchom testy zadań

```bash
pytest -q
```

### 2) Walidacja zmian w scorerze (Level 4)

```bash
python -m course.validate_scorer --dataset data/datasets/math_dev.jsonl --golden data/golden/golden_correct.jsonl
python -m course.validate_scorer --dataset data/datasets/math_dev.jsonl --golden data/golden/golden_exploits.jsonl
```

### 3) Sprawdzenie deterministyczności (Level 0.5 sabotage/repair)

Po naprawie deterministyczności selection, powtórzenia powinny być identyczne:

```bash
for i in 1 2 3; do
  python -m course.selection_demo \
    --dataset data/datasets/math_dev.jsonl \
    --samples data/rollouts/selection_pack_dev.jsonl \
    --n 4 \
    --outdir runs/selfcheck_sel_$i
done

for i in 1 2 3; do
  python -c "import hashlib, pathlib; p=pathlib.Path('runs/selfcheck_sel_${i}/results.jsonl'); print(${i}, hashlib.sha256(p.read_bytes()).hexdigest())"
done
```

Jeśli hashe się zgadzają, selection policy jest deterministyczna.

### 4) Szybka kontrola artefaktów

Dla dowolnego przebiegu w `runs/<name>/` potwierdź:
- `manifest.json` istnieje i zawiera info o dataspecie + scorerze.
- `results.jsonl` istnieje i zawiera rekordy per-przykład.
- `summary.json` istnieje i metryki zmieniają się zgodnie z celem poziomu.

## Zadania zaczynają się tutaj (krok po kroku)

Od tego miejsca idź poziom po poziomie. Każdy poziom jest napisany krok po kroku i zawiera krótkie wyjaśnienie, po co wykonujesz dane działanie. Nie pomijaj sabotażu ani naprawy; kontrast jest lekcją.

---

# Level 0: Higiena pomiarowa — Loop A

## Podstawa koncepcyjna

**Błędna intuicja:** „Model działa słabo."

**Poprawny model mentalny:** „Loop A stanowi weryfikację instrumentu pomiarowego. Jeśli zmierzona wartość się zmieniła, to albo (a) policy się zmieniła, albo (b) warunki pomiaru się zmieniły. Celem jest określenie, które."

---

## Checklista pierwszego uruchomienia

- Wejścia: `data/datasets/math_dev.jsonl`, `data/rollouts/frozen_rollouts_dev.jsonl`
- Wyjścia: `runs/l0_build_eval`, `runs/l0_sabotage_eval_tampered` z manifest/results/summary
- Edycje: `notes/mental_map_v1.md`, `course/assignments/kata_01.py`, `tests/test_kata_01.py`

---

## Krok po kroku

### Krok 1 - Build: Wykonanie czystej ewaluacji Loop A

Budujesz czystą linię bazową, aby późniejsze porównania miały wiarygodny punkt odniesienia.

```bash
python -m course.eval \
  --dataset data/datasets/math_dev.jsonl \
  --completions data/rollouts/frozen_rollouts_dev.jsonl \
  --outdir runs/l0_build_eval

python -m course.inspect_run --run runs/l0_build_eval --only-fails --top-k 5 --show 2
```

### Krok 2 - Sabotage: Wprowadzenie manipulacji datasetu (naruszenie Locked Room)

Celowo zmieniasz jedną etykietę, żeby pokazać, że metryki mogą się przesunąć bez zmiany policy.

```bash
cp data/datasets/math_dev.jsonl data/datasets/math_dev_TAMPERED.jsonl
# Ręcznie zmodyfikuj dokładnie JEDEN expected_answer w zmanipulowanym pliku.

python -m course.eval \
  --dataset data/datasets/math_dev_TAMPERED.jsonl \
  --completions data/rollouts/frozen_rollouts_dev.jsonl \
  --outdir runs/l0_sabotage_eval_tampered
```

### Krok 3 - Reflect: Wywołanie bramki do oceny porównywalności

Używasz bramki, żeby wykazać, że przebiegi nie są porównywalne; oczekiwany REJECT to zabezpieczenie.

```bash
python -m course.gate \
  --baseline runs/l0_build_eval \
  --candidate runs/l0_sabotage_eval_tampered \
  --min-delta 0.00
```

Oczekiwane wyjście to **REJECT** z informacjami diagnostycznymi wskazującymi na niekompatybilność Locked Room.

---

## Zadania Capstone

Cel: udokumentować model mentalny i zablokować koncepcje krótką katą.

**Dozwolone modyfikacje:** Pliki pod `notes/`, `course/assignments/`, `tests/` i nowe pliki pod `data/`.
Modyfikacje kodu scorera nie są dozwolone na tym poziomie.

### Zadanie A — Dokumentacja modelu mentalnego

Utwórz `notes/mental_map_v1.md` (1–2 strony):

* Zdefiniuj następujące terminy: reward, metric, objective, loss, policy, environment
* Narysuj diagram Loop A/B/C z zidentyfikowaną główną zmienną dla każdej

### Zadanie B — Debugging Kata (ocena automatyczna)

Utwórz `course/assignments/kata_01.py`:

```python
def classify(outcome_code: str) -> str:
    """
    Zwróć jedno z:
      - model_math
      - model_format
      - data_invalid
      - unknown
    """
```

Utwórz `tests/test_kata_01.py` z odpowiednimi przypadkami testowymi. Minimalne wymagane mapowania:

* `"wrong_answer"` → `model_math`
* Dowolny kod parse-format (np. `"missing_prefix"`, `"extra_whitespace"`, `"not_single_line"`, `"leading_zeros"`) → `model_format`
* `"invalid_example"` → `data_invalid`

Wykonaj weryfikację:

```bash
pytest -q
```

---

## Kryteria ukończenia

* Mając dwa przebiegi ewaluacji z różniącymi się wynikami, czy potrafisz wyliczyć poprawne przyczyny i dostarczyć dowody z `manifest.json`?
* Czy potrafisz wyartykułować rozróżnienie między **błędem formatu** a **błędem matematycznym** używając outcome codes?
* Czy potrafisz wyjaśnić, dlaczego „zmanipulowany dataset poprawił pass_rate" nie stanowi poprawnej poprawy?

---

# Level 0.5: Selection bez Learning — Loop B

## Podstawa koncepcyjna

**Błędna intuicja:** „pass@N się poprawił, więc model się nauczył."

**Poprawny model mentalny:** „Loop B modyfikuje regułę decyzyjną nad próbkami (compute selection). Leżący u podstaw rozkład policy pozostaje niezmieniony."

---

## Checklista pierwszego uruchomienia

- Wejścia: `data/datasets/math_dev.jsonl`, `data/rollouts/selection_pack_dev.jsonl`
- Wyjścia: `runs/l0_5_build_sel_n4`, `runs/l0_5_sabotage_sel_random_*`, `runs/l0_5_fixed_sel_n4`
- Edycje: `course/assignments/selection_policy.py`, `tests/test_selection_policy.py`

---

## Krok po kroku

### Krok 1 - Build: Wykonanie demo selection na dostarczonych danych

Ustanawiasz linię bazową selection, żeby mieć punkt odniesienia dla późniejszych zmian.

```bash
python -m course.selection_demo \
  --dataset data/datasets/math_dev.jsonl \
  --samples data/rollouts/selection_pack_dev.jsonl \
  --n 4 \
  --outdir runs/l0_5_build_sel_n4

python -m course.inspect_run --run runs/l0_5_build_sel_n4 --top-k 5 --show 1
```

### Krok 2 - Sabotage: Wprowadzenie niedeterminizmu do tie-breaking

Wstrzykujesz losowość, aby pokazać, że niedeterministyczny tie-break destabilizuje artefakty.

Edytuj `course/assignments/selection_policy.py`.

Zmodyfikuj `tie_break_key`, aby uwzględniał jawną losowość:

```python
import random

def tie_break_key(sample):
    return (random.random(), sample.completion)
```

Wykonaj identyczne polecenie pięć razy:

```bash
for i in 1 2 3 4 5; do
  python -m course.selection_demo \
    --dataset data/datasets/math_dev.jsonl \
    --samples data/rollouts/selection_pack_dev.jsonl \
    --n 4 \
    --outdir runs/l0_5_sabotage_sel_random_$i
done
```

### Krok 3 - Reflect: Wykazanie wariancji przez analizę artefaktów

Potwierdzasz niedeterministyczność, pokazując różne hashe `results.jsonl`.

Oblicz hash każdego `results.jsonl`. Jeśli selection jest niedeterministyczne, hashe będą się różnić:

```bash
for i in 1 2 3 4 5; do
  python -c "import hashlib, pathlib; p=pathlib.Path('runs/l0_5_sabotage_sel_random_${i}/results.jsonl'); print(${i}, hashlib.sha256(p.read_bytes()).hexdigest())"
done
```

---

## Zadania Capstone

Cel: naprawić policy selection i zablokować deterministyczność testami.

**Dozwolone modyfikacje:** Tylko `course/assignments/selection_policy.py` i powiązane testy.

### Repair and Lock

1. Usuń losowość. Zaimplementuj deterministyczny tie-breaking z następującym priorytetem:

   * Wyższy `sum_logprob` (jeśli obecny)
   * Krótsza długość uzupełnienia
   * Porządek leksykograficzny tekstu

Przykładowy deterministyczny klucz:

```python
def tie_break_key(sample):
    lp = sample.sum_logprob
    lp_key = 0.0 if lp is None else -float(lp)   # ujemny, aby sortować wyższy logprob jako pierwszy
    return (lp_key, len(sample.completion), sample.completion)
```

2. Utwórz `tests/test_selection_policy.py`:

   * Test determinizmu: powtarzane wywołania wybierają identyczny indeks
   * Test tie-break: mając dwie próbki o równej nagrodzie, wybierana jest zamierzona

Wykonaj weryfikację:

```bash
pytest -q
python -m course.selection_demo --dataset data/datasets/math_dev.jsonl --samples data/rollouts/selection_pack_dev.jsonl --n 4 --outdir runs/l0_5_fixed_sel_n4
```

---

## Kryteria ukończenia

* Dlaczego Loop B może poprawiać wyjścia bez modyfikacji rozkładu policy?
* Jaka konkretna zmienna została zmodyfikowana podczas sabotażu i jak to zostało zademonstrowane przez analizę artefaktów?
* Dlaczego kod selection musi być deterministyczny w systemach produkcyjnych?

---

# Level 1: Analiza algorytmu REINFORCE — Loop C (uproszczona domena)

## Podstawa koncepcyjna

**Błędna intuicja:** „RL to supervised learning z dodatkową złożonością."

**Poprawny model mentalny:** „RL dostosowuje rozkłady prawdopodobieństwa. Kierunek aktualizacji jest kontrolowany przez advantage = reward − baseline. Baseline redukują wariancję bez zmiany celu optymalizacji."

---

## Checklista pierwszego uruchomienia

- Wejścia: brak (syntetyczny bandit)
- Wyjścia: `runs/l1_build_bandit`, `runs/l1_sabotage_lr2`, `runs/l1_sabotage_lrneg`, `runs/l1_build_slow`, `runs/l1_sabotage_slow_lrneg`
- Edycje: `notes/reinforce_forensics.md`

---

## Krok po kroku

### Krok 1 - Build: Stabilne uczenie z baseline

Ustanawiasz zdrową linię bazową uczenia, aby rozpoznać późniejsze awarie.

```bash
python -m course.bandit_train --steps 200 --seed 0 --lr 0.5 --baseline --outdir runs/l1_build_bandit
```

### Krok 2 - Sabotage: Nadmierna optymalizacja i odwrócone uczenie

Celowo destabilizujesz uczenie, aby zobaczyć dwa różne tryby awarii.

Wykonaj dwa eksperymenty sabotażowe:

**(A) Nadmierny learning rate**

```bash
python -m course.bandit_train --steps 200 --seed 0 --lr 2.0 --baseline --outdir runs/l1_sabotage_lr2
```

**(B) Ujemny learning rate (odwrócony kierunek aktualizacji)**

```bash
python -m course.bandit_train --steps 200 --seed 0 --lr -0.5 --baseline --outdir runs/l1_sabotage_lrneg
```

### Krok 3 - Reflect: Obserwacja mechanizmu krok po kroku

Tryb slow pozwala zobaczyć, jak znaki advantage zmieniają prawdopodobieństwa.

```bash
python -m course.bandit_train --steps 30 --seed 0 --lr 0.5 --baseline --slow --outdir runs/l1_build_slow
python -m course.bandit_train --steps 30 --seed 0 --lr -0.5 --baseline --slow --outdir runs/l1_sabotage_slow_lrneg
```

---

## Zadania Capstone

Cel: napisać krótką notatkę forensics łączącą znak advantage z kierunkiem aktualizacji.

**Dozwolone modyfikacje:** Pliki pod `notes/` i `course/assignments/`. Modyfikacja `course/bandit_train.py` nie jest dozwolona.

Utwórz `notes/reinforce_forensics.md`:

* Wybierz 10 kolejnych wpisów z wyjścia `--slow`
* Dla każdego wpisu zanotuj:

  * Akcja, nagroda, baseline, znak advantage
  * Wymagana zmiana prawdopodobieństwa tej akcji

Zakończ jednym akapitem wyjaśniającym różnice behawioralne zaobserwowane w przebiegach sabotażowych.

---

## Kryteria ukończenia

* Jeśli advantage jest ujemny, co dzieje się z prawdopodobieństwem spróbkowanej akcji?
* Dlaczego baseline redukuje szum bez zmiany kierunku uczenia?
* Co ujemny learning rate zademonstrował odnośnie dostosowania prawdopodobieństwa?

---

# Level 2: Credit Assignment i granica Token-Tekst

## Podstawa koncepcyjna

**Błędna intuicja:** „Tylko końcowy token ma znaczenie, ponieważ nagroda jest przypisywana na końcu sekwencji."

**Poprawny model mentalny:** „W generowaniu sekwencyjnym wczesne decyzje ograniczają późniejsze stany. Końcowa nagroda staje się sygnałem treningowym propagowanym wzdłuż spróbkowanej trajektorii (credit assignment). Dodatkowo: policy operuje w przestrzeni tokenów; weryfikacja operuje w przestrzeni tekstu."

---

## Checklista pierwszego uruchomienia

- Wejścia: brak (token_inspect używa stałych stringów)
- Wyjścia: `runs/l2_build_two_step`, `runs/l2_sabotage_no_credit_step1`, `runs/l2_fixed_two_step`
- Edycje: `course/assignments/two_step_mdp_demo.py`, `notes/credit_assignment.md`

---

## Krok po kroku

### Krok 1 - Build: Obserwacja granicy tokenów

Obserwujesz, jak drobne zmiany formatu tworzą inne tokenizacje, co tłumaczy rygor formatu.

```bash
python -m course.token_inspect "Final: 323"
python -m course.token_inspect "Final:  323"
python -m course.token_inspect "Final:\n323"
python -m course.token_inspect "Final: 0323"
```

---

## Zadania Capstone

Cel: zbudować dwukrokowe demo MDP, celowo je zepsuć, a potem naprawić i wyjaśnić dlaczego.

**Dozwolone modyfikacje:** Utwórz `course/assignments/two_step_mdp_demo.py` i powiązane notatki. Modyfikacja scorera nie jest dozwolona.

### Krok 2 - Build: Poprawna implementacja dwukrokowego REINFORCE

Implementujesz środowisko dwukrokowe, aby credit assignment w czasie było widoczne.

Utwórz `course/assignments/two_step_mdp_demo.py`:

* Krok 1: Wybierz A lub B
* Krok 2: Wybierz X lub Y uwarunkowane wynikiem kroku 1
* Nagroda przypisywana tylko przy zakończeniu
* Aktualizuj **obie** policy kroków używając końcowej nagrody (baseline opcjonalnie)

```bash
python course/assignments/two_step_mdp_demo.py --steps 200 --seed 0 --baseline --outdir runs/l2_build_two_step
```

### Krok 3 - Sabotage: Usunięcie credit assignment dla kroku 1

Usuwasz aktualizacje kroku 1, aby zobaczyć, co się psuje bez credit assignment.

Zmodyfikuj skrypt tak, aby **tylko krok 2 otrzymywał aktualizacje** (krok 1 pozostaje jednostajnie losowy):

```bash
python course/assignments/two_step_mdp_demo.py --steps 200 --seed 0 --baseline --outdir runs/l2_sabotage_no_credit_step1
```

### Krok 4 - Repair: Przywrócenie aktualizacji kroku 1

Przywracasz aktualizacje kroku 1, aby odzyskać uczenie i porównać artefakty.

```bash
python course/assignments/two_step_mdp_demo.py --steps 200 --seed 0 --baseline --outdir runs/l2_fixed_two_step
```

Utwórz `notes/credit_assignment.md`:

Wyjaśnij prostym językiem, co zawiodło podczas sabotażu i dlaczego naprawa działa. Dołącz:

* Krótki opis awarii i jej przyczyny
* Dołącz diagram: tokeny → detokenizacja → tekst → parsowanie → nagroda

---

## Kryteria ukończenia

* Dlaczego krok 1 może wpływać na wyniki, nawet gdy nagroda jest przypisywana przy zakończeniu?
* Co odróżnia macro-actions od micro-actions w kontekście LLM?
* Dlaczego parsowanie/formatowanie jest uważane za część granicy środowiska?

---

# Level 3: KL Divergence jako ograniczenie regularyzacyjne

## Podstawa koncepcyjna

**Błędna intuicja:** „KL divergence jest czysto matematyczną abstrakcją."

**Poprawny model mentalny:** „KL divergence funkcjonuje jako praktyczne ograniczenie: maksymalizuj nagrodę, jednocześnie karząc nadmierną dywergencję od rozkładu referencyjnego."

---

## Checklista pierwszego uruchomienia

- Wejścia: brak (demo używa syntetycznych policy)
- Wyjścia: `runs/l3_build_kl_demo`, `runs/l3_build_kl_choice.txt`, `runs/l3_sabotage_no_kl.txt`
- Edycje: `course/assignments/kl_regularized_choice.py`, `notes/kl_tradeoff.md`

---

## Krok po kroku

### Krok 1 - Build: Wykonanie demonstracji kompromisu KL

Uruchamiasz demo, żeby zobaczyć kompromis reward vs KL zanim napiszesz kod.

```bash
python -m course.kl_tradeoff_demo --plot --outdir runs/l3_build_kl_demo
```

---

## Zadania Capstone

Cel: zaimplementować małą regułę KL-regularized, zepsuć ją i wyjaśnić różnicę.

**Dozwolone modyfikacje:** Utwórz skrypty assignment i notatki. Modyfikacja głównego kodu demonstracji nie jest dozwolona.

### Krok 2 - Build: Selection z regularyzacją KL na danych syntetycznych

Implementujesz regułę wyboru, aby kara KL była konkretna.

Utwórz `course/assignments/kl_regularized_choice.py`:

* Zdefiniuj 6 opcji kandydatów, każda z wartościami `(reward, kl)`
* Zaimplementuj regułę selekcji: maksymalizuj `reward - beta * kl`
* Wypisz wybranego kandydata dla `beta = 0.1` i `beta = 1.0`

```bash
python course/assignments/kl_regularized_choice.py > runs/l3_build_kl_choice.txt
```

### Krok 3 - Sabotage: Usunięcie ograniczenia regularyzacyjnego (beta = 0)

Usuwasz ograniczenie, aby zobaczyć, jak zmienia się wybór bez KL.

Zmodyfikuj skrypt, aby ustawić beta = 0:

```bash
python course/assignments/kl_regularized_choice.py > runs/l3_sabotage_no_kl.txt
```

### Krok 4 - Reflect

Dokumentujesz, dlaczego nieograniczony wybór jest kuszący, ale ryzykowny.

Utwórz `notes/kl_tradeoff.md`:

* Wyjaśnij, dlaczego beta = 0 reprezentuje atrakcyjną, ale problematyczną konfigurację
* Połącz to z tym, dlaczego nieograniczona optymalizacja ma tendencję do rozwiązań ekstremalnych, o wysokiej dywergencji

---

## Kryteria ukończenia

* Co KL divergence ogranicza w terminach operacyjnych?
* Dlaczego usunięcie preferencji KL może prowadzić do zachowania eksploatującego nagrodę, nawet gdy mierzona nagroda rośnie?
* Co stanowi „referencję" w tym frameworku regularyzacyjnym?

---

# Level 4: Specyfikacja nagrody jako kod produkcyjny

## Podstawa koncepcyjna

**Błędna intuicja:** „Scorer jest jedynie instrukcją asercji."

**Poprawny model mentalny:** „Scorer definiuje specyfikację zadania i musi być traktowany jako produkcyjne oprogramowanie pomiarowe."

---

## Checklista pierwszego uruchomienia

- Wejścia: `data/datasets/math_dev.jsonl`, `data/golden/golden_correct.jsonl`, `data/golden/golden_exploits.jsonl`
- Wyjścia: `data/golden/golden_exploits_extra.jsonl`, `tests/test_reward_regressions.py`
- Edycje: `course/core/scoring.py`, `notes/reward_spec_blackbox.md`

---

## Krok po kroku

### Krok 1 - Build: Walidacja scorera względem golden test cases

Ustanawiasz bieżące zachowanie scorera, żeby zmiany były mierzalne.

```bash
python -m course.validate_scorer \
  --dataset data/datasets/math_dev.jsonl \
  --golden data/golden/golden_correct.jsonl

python -m course.validate_scorer \
  --dataset data/datasets/math_dev.jsonl \
  --golden data/golden/golden_exploits.jsonl
```

---

## Zadania Capstone

Cel: sondować scorer, wprowadzić celowe osłabienie, a potem przywrócić i zablokować poprawkę.

**Dozwolone modyfikacje:** `course/core/scoring.py`, `tests/`, `data/golden/`, `notes/`.
Modyfikacje kodu selection i learning nie są dozwolone na tym poziomie.

### Krok 2 - Build: Sondowanie specyfikacji Black-Box

Wnioskujesz o specyfikacji z obserwacji, czyli tak jak w prawdziwym black-box.

Utwórz `notes/reward_spec_blackbox.md`:

* Udokumentuj reguły formatu, które twoim zdaniem istnieją na podstawie zaobserwowanego zachowania
* Dołącz 8 ciągów sond z przewidywanymi outcome codes

### Krok 3 - Sabotage: Poluzowanie jednej reguły specyfikacji

Osłabiasz jedną regułę, aby zobaczyć, jak exploity przechodzą.

Wybierz jedną ścisłą regułę w `course/core/scoring.py` i celowo ją osłab (przykłady):

* Pozwól na dodatkowe whitespace po `Final:`
* Pozwól na wiodące zera
* Pozwól na wieloliniowe wyjścia

Wykonaj walidację golden i obserwuj awarie:

```bash
python -m course.validate_scorer \
  --dataset data/datasets/math_dev.jsonl \
  --golden data/golden/golden_exploits.jsonl
```

### Krok 4 - Repair and Lock: Przywrócenie ścisłości i rozszerzenie pokrycia testami

Przywracasz specyfikację i blokujesz ją testami oraz goldenami.

1. Przywróć poprawne zachowanie specyfikacji.
2. Utwórz `tests/test_reward_regressions.py` z **co najmniej 6** przypadkami testowymi:

   * 2 zweryfikowane poprawne przypadki
   * 4 przypadki exploit/edge (manipulacja formatem)
3. Utwórz `data/golden/golden_exploits_extra.jsonl` z **co najmniej 5** dodatkowymi przypadkami exploit.
4. Jeśli zachowanie reward różni się od linii bazowej, inkrementuj `SCORER_VERSION`.

Wykonaj weryfikację:

```bash
pytest -q
python -m course.validate_scorer --dataset data/datasets/math_dev.jsonl --golden data/golden/golden_exploits_extra.jsonl
```

---

## Kryteria ukończenia

* Co stanowi false positive versus false negative w tym kontekście weryfikacji?
* Dlaczego scorer musi być wersjonowany przy zmianie zachowania, unieważniając porównania z wcześniejszymi metrykami uruchomień?
* Dlaczego „bardziej łagodne parsowanie" jest zazwyczaj kontrproduktywne?

---

# Level 5: Analiza eksploatacji nagrody

## Podstawa koncepcyjna

**Błędna intuicja:** „Reward hacking wskazuje na złośliwy optymalizator."

**Poprawny model mentalny:** „Optymalizatory eksploatują miary proxy. Jeśli specyfikacja zawiera luki, optymalizacja je odkryje. Rozwiązaniem jest łatanie klas mechanizmów i blokowanie ich testami."

---

## Checklista pierwszego uruchomienia

- Wejścia: opcjonalny baseline używa `data/datasets/math_dev.jsonl`, `data/rollouts/frozen_rollouts_dev.jsonl`
- Wyjścia: `runs/l5_build_eval_context` (opcjonalne)
- Edycje: `course/assignments/hackable_scorer_demo.py`, `notes/red_team_report.md`

---

## Krok po kroku

### Krok 1 - Build: Ustanowienie kontekstu bazowego (opcjonalne)

Odświeżasz kontekst działania ewaluatora; to opcjonalny materiał bazowy.

```bash
python -m course.eval \
  --dataset data/datasets/math_dev.jsonl \
  --completions data/rollouts/frozen_rollouts_dev.jsonl \
  --outdir runs/l5_build_eval_context

python -m course.inspect_run --run runs/l5_build_eval_context --only-fails --top-k 5 --show 2
```

---

## Zadania Capstone

Cel: zbudować naiwny weryfikator, wyeksploatować go i naprawić klasę exploitów.

**Dozwolone modyfikacje:** Nowe pliki assignment, notatki i testy. Modyfikacje produkcyjnego scorera powinny być traktowane jako zmiany instrumentu wymagające inkrementacji wersji.

### Krok 2 - Build: Implementacja celowo naiwnego weryfikatora

Tworzysz celowo słaby weryfikator, żeby móc zbadać jego porażki.

Utwórz `course/assignments/hackable_scorer_demo.py`:

* Przypisz reward = 1, jeśli oczekiwana liczba całkowita pojawia się gdziekolwiek w uzupełnieniu (naiwne podejście substring/regex)
* Zademonstruj, że produkuje poprawne wyniki na uczciwych uzupełnieniach

### Krok 3 - Sabotage: Wygenerowanie 5 exploitów

Tworzysz uzupełnienia, które przechodzą naiwny test bez rozwiązania zadania.

Utwórz 5 uzupełnień, które osiągają reward = 1 bez stanowienia poprawnych rozwiązań (np. wyliczanie liczb).

### Krok 4 - Repair: Załatanie klasy exploitów

Łatasz klasę exploitów (nie tylko konkretne ciągi) i dokumentujesz poprawkę.

Zmodyfikuj demonstracyjny weryfikator, aby zamknąć klasę exploitów (nie tylko konkretne ciągi exploit).

Utwórz `notes/red_team_report.md`:

* Udokumentuj 5 ciągów exploit
* Wyjaśnij, dlaczego odniosły sukces
* Opisz strategię łatki
* Określ, jakie testy zapobiegłyby regresji

---

## Kryteria ukończenia

* Podaj konkretny przykład „metryka proxy rośnie, prawdziwy cel maleje."
* Co oznacza „załataj klasę, nie instancję"?
* Dlaczego ta analiza ma znaczenie dla późniejszej pracy Loop C?

---

# Level 6: Komitet promocyjny (trzy kontrolowane pułapki eksperymentalne)

## Podstawa koncepcyjna

**Błędna intuicja:** „Wyższa wartość metryki równa się poprawie."

**Poprawny model mentalny:** „Promowalna poprawa musi spełniać porównywalność Locked Room i być przypisywalna do dokładnie jednej kontrolowanej zmiennej."

---

## Checklista pierwszego uruchomienia

- Wejścia: `runs/l0_build_eval`, `runs/l0_sabotage_eval_tampered`, `data/rollouts/selection_pack_dev.jsonl`
- Wyjścia: `runs/l6_trap_sel_n1`, `runs/l6_trap_sel_n4`, `runs/l6_trap_learning`
- Edycje: `notes/promotion_memo.md`

---

## Krok po kroku

### Krok 1 - Pułapka 1 (Sabotage): Nieważna promocja przez zmianę instrumentu

Próbujesz „promocji" na podstawie naruszenia Locked Room, aby pokazać, po co jest gate.

Użyj wcześniej zmanipulowanego przebiegu z datasetem (lub utwórz nowy), następnie wywołaj bramkę:

```bash
python -m course.gate \
  --baseline runs/l0_build_eval \
  --candidate runs/l0_sabotage_eval_tampered \
  --min-delta 0.00
```

### Krok 2 - Pułapka 2: Poprawa selection bez learning

Pokazujesz, że metryki mogą rosnąć bez uczenia, więc promocja nie jest uzasadniona.

```bash
python -m course.selection_demo \
  --dataset data/datasets/math_dev.jsonl \
  --samples data/rollouts/selection_pack_dev.jsonl \
  --n 1 \
  --outdir runs/l6_trap_sel_n1

python -m course.selection_demo \
  --dataset data/datasets/math_dev.jsonl \
  --samples data/rollouts/selection_pack_dev.jsonl \
  --n 4 \
  --outdir runs/l6_trap_sel_n4
```

### Krok 3 - Pułapka 3: Learning modyfikuje zachowanie

Pokazujesz realny learning, a następnie argumentujesz promocję na podstawie dowodów i ryzyka holdout.

```bash
python -m course.bandit_train --steps 200 --seed 1 --lr 0.5 --baseline --outdir runs/l6_trap_learning
```

---

## Zadania Capstone

Cel: napisać krótką notatkę promocyjną opartą o artefakty, nie intuicję.

Utwórz `notes/promotion_memo.md` (maksymalnie 1 strona):

* Dla każdej pułapki udokumentuj:

  * Klasyfikacja loop (A/B/C)
  * Zmodyfikowana zmienna
  * Ocena poprawności porównania
  * Dowody (cytat z `manifest.json` / wersja scorera / modyfikacja parametru)
* Akapit końcowy: co byś PROMOWAŁ versus ODRZUCIŁ dziś, z uzasadnieniem

---

## Kryteria ukończenia

* Jeśli pass_rate wzrosło, jak demonstrujesz, że nie jest to naruszenie Locked Room?
* Jak odróżniasz poprawę selection od poprawy learning używając wyłącznie artefaktów?
* Dlaczego Loop C nieuchronnie tworzy nowe tryby awarii, których zamrożone rollouts nie mogą w pełni ujawnić?

---

## Podsumowanie ukończenia ścieżki

Pomyślne ukończenie tej ścieżki ustanawia podstawową kompetencję, do której ten kurs dąży: zdolność do wykonywania i interpretowania eksperymentów RL-for-LLMs bez mylenia **pomiaru**, **selection** i **learning**, oraz bez akceptowania metodologicznie nieważnych porównań.

---

## Aneks

### Rubryka (jak wygląda poprawne wykonanie)

Użyj tego jako checklisty do samodzielnej oceny. Jeśli punkt nie przechodzi, wróć do artefaktów i popraw.

#### Level 0 — Loop A (higiena pomiarowa)
- **Build:** `pass_rate` w `runs/l0_build_eval/summary.json` mieści się w [0, 1].
- **Sabotage:** `runs/l0_sabotage_eval_tampered` ma inny hash datasetu niż build (zob. `manifest.json`).
- **Gate:** `python -m course.gate` zwraca `REJECT` z naruszeniem Locked Room.
- **Mental map:** `notes/mental_map_v1.md` zawiera definicje i rozróżnienie A/B/C.
- **Kata:** `classify()` mapuje błędy formatu na `model_format`, a `"wrong_answer"` na `model_math`.

#### Level 0.5 — Loop B (selection)
- **Build:** `pass_at_n` > `pass_at_1` w `runs/l0_5_build_sel_n4/summary.json`.
- **Sabotage:** hashe `results.jsonl` różnią się przy losowym tie‑break.
- **Repair:** hashe są identyczne; selection jest deterministyczna.
- **Tie‑break:** logprob → długość → leksykografia.

#### Level 1 — Loop C (REINFORCE)
- **Build:** `mean_reward` poprawia się w czasie (zob. `summary.json` lub logi slow).
- **Sabotage (ujemny lr):** `mean_reward` znacznie niższy niż w build.
- **Forensics:** `notes/reinforce_forensics.md` poprawnie interpretuje znak advantage.

#### Level 2 — Credit assignment + granica token‑tekst
- **Token inspect:** warianty formatu tokenizują się inaczej.
- **Two‑step build:** `mean_reward` blisko 1.0 i policy kroku 1 staje się nierównomierne.
- **Sabotage:** bez aktualizacji kroku 1 policy kroku 1 ~równomierne i `mean_reward` spada.
- **Repair:** zachowanie wraca do build.

#### Level 3 — KL tradeoff
- **KL demo:** istnieje `runs/l3_build_kl_demo/kl_tradeoff.csv`.
- **KL choice:** wybór różni się dla `beta=0.1` vs `beta=1.0`.
- **Notes:** `notes/kl_tradeoff.md` wyjaśnia, czemu `beta=0` jest ryzykowne.

#### Level 4 — Scorer jako kod produkcyjny
- **Golden testy:** oba zbiory przechodzą po naprawie.
- **Sabotage:** po osłabieniu reguły przynajmniej jeden exploit przechodzi.
- **Lock:** testy regresji zawierają przypadki poprawne i exploit.

#### Level 5 — Reward exploitation
- **Naive verifier:** exploitowe stringi dostają nagrodę w wersji naiwnej.
- **Patched verifier:** te same stringi są odrzucane po poprawce.
- **Raport:** `notes/red_team_report.md` opisuje exploity i strategię naprawy.

#### Level 6 — Promotion committee
- **Trap 1:** REJECT z dowodem naruszenia Locked Room.
- **Trap 2:** poprawa selection bez learningu; odrzucić do promocji.
- **Trap 3:** learning poprawia reward; promocja warunkowa (holdout).

### Szablon dokumentacji uruchomienia

Następujący szablon powinien być wypełniony dla każdego przebiegu eksperymentalnego w `runs/<name>/README.md`:

* **Loop:** A / B / C
* **Zmodyfikowana zmienna (wybierz dokładnie jedną):** instrument pomiarowy / compute selection / policy / środowisko
* **Dowód Locked Room:** (fingerprinty datasetu + wersja scorera; bezpośredni cytat z manifestu)
* **Oczekiwany wynik:**
* **Zaobserwowany wynik (ilościowy):**
* **Dwa konkretne przykłady:** (id przykładu + outcome code + zaobserwowane zachowanie)
* **Plan naprawy:** (jedno zdanie)
