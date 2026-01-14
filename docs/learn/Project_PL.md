# RL Foundations for LLMs

**Projekt:** *Verifier-Driven Math QA — Zbuduj „działający samochód", następnie go zepsuj, potem wzmocnij.*

Ta ścieżka jest zaprojektowana, aby „wymusić" zrozumienie poprzez powtarzanie tego samego cyklu:

1. **Build (działający samochód):** uzyskaj czystą, deterministyczną linię bazową, którą w pełni kontrolujesz.
2. **Sabotage (zepsuj):** wprowadź jedną małą, jawną zmianę, która łamie podstawową zasadę.
3. **Reflect (forensics):** obserwuj awarię używając artefaktów (`manifest.json`, `summary.json`, `results.jsonl`) zamiast intuicji.
4. **Repair + Lock:** napraw *mechanizm* i zakoduj go jako testy + bramki, aby nie mógł ulec regresji.

**Nie pomijaj kroków Sabotage.** To jest cały sens.

---

## Podstawowe reguły (zapamiętaj je wcześnie)

### Trzy pętle (używaj tej terminologii)

* **Loop A (Eval):** tylko pomiar (zamrożone wejścia, deterministyczny scorer).
* **Loop B (Selection):** wybierz best-of-N (zachowanie się poprawia; leżący u podstaw rozkład policy nie).
* **Loop C (Learning):** aktualizuj policy (prawdopodobieństwa się przesuwają → rozkład się przesuwa).

### Zasada Locked Room (poprawne porównania)

Zanim porównasz dwa numery uruchomień, musisz zweryfikować, że te są identyczne:

* podział datasetu (ten sam fingerprint pliku w `manifest.json`)
* tekst promptu (jeśli dotyczy)
* **nazwa + wersja** scorera
* ustawienia próbkowania (jeśli dotyczy)

Jeśli którekolwiek się różnią, zmieniłeś środowisko/instrument. To porównanie jest nieważne.

### Reward = Spec (traktuj `score()` jak kod produkcyjny)

Twój scorer jest instrumentem pomiarowym. Musi być:
**deterministyczny**, **totalny (nigdy się nie crashuje)**, **wyjaśnialny**, **szybki**, **wersjonowany**.

---

## Dyscyplina workflow (silnie zalecana)

Stwórz branch i rób małe commity, abyś mógł udowodnić, co się zmieniło.

```bash
git checkout -b mastery-track
pytest -q
```

Dla każdego poziomu zrób:

* Commit stanu **Build**.
* Commit zmiany **Sabotage** (nawet jeśli jest „błędna").
* Commit **Fix**.

---

## Szablon notatki z uruchomienia (umieść to w każdym `runs/<name>/README.md`)

* **Loop:** A / B / C
* **Pokrętło przekręcone (wybierz JEDNO):** instrument pomiarowy / compute selection / policy / środowisko
* **Dowód Locked Room:** (fingerprinty datasetu + wersja scorera; cytat z manifestu)
* **Czego oczekiwałem:**
* **Co się stało (liczby):**
* **Dwa konkretne przykłady:** (id + outcome code + co zobaczyłem)
* **Plan naprawy:** (jedno zdanie)

---

# Level 0: Higiena pomiarowa (Build → Tamper → Catch It) — Loop A

## 1) Zmiana nastawienia

**Błędna intuicja:** „Model jest głupi."
**Poprawny model mentalny:** „Loop A jest sprawdzeniem instrumentu pomiarowego. Jeśli liczba się zmieniła, to albo (a) policy się zmieniła, albo (b) warunki pomiaru się zmieniły. Moim zadaniem jest udowodnić, które."

## 2) Działania studenta (dokładne CLI)

### Build (działający samochód): uruchom czysty Loop A eval

```bash
python -m course.eval \
  --dataset data/datasets/math_dev.jsonl \
  --completions data/rollouts/frozen_rollouts_dev.jsonl \
  --outdir runs/l0_build_eval

python -m course.inspect_run --run runs/l0_build_eval --only-fails --top-k 5 --show 2
```

### Sabotage (zepsuj): zmanipuluj dataset (naruszenie Locked Room)

```bash
cp data/datasets/math_dev.jsonl data/datasets/math_dev_TAMPERED.jsonl
# Ręcznie edytuj dokładnie JEDEN expected_answer w zmanipulowanym pliku.

python -m course.eval \
  --dataset data/datasets/math_dev_TAMPERED.jsonl \
  --completions data/rollouts/frozen_rollouts_dev.jsonl \
  --outdir runs/l0_sabotage_eval_tampered
```

### Reflect (forensics): zmuś bramkę do oceny porównywalności

```bash
python -m course.gate \
  --baseline runs/l0_build_eval \
  --candidate runs/l0_sabotage_eval_tampered \
  --min-delta 0.00
```

Powinieneś zobaczyć **REJECT** z powodami wskazującymi na niekompatybilność Locked Room.

## 3) Zadanie Capstone (zestaw umiejętności)

**Budżet pokręteł:** Dozwolone zmiany: `notes/`, `course/assignments/`, `tests/`, nowe pliki pod `data/`.
Nie zmieniaj jeszcze kodu scorera.

### Task A — Napisz model mentalny

Stwórz `notes/mental_map_v1.md` (1–2 strony):

* zdefiniuj: reward, metric, objective, loss, policy, environment
* narysuj Loop A/B/C i oznacz pokrętło dla każdej

### Task B — Debugging Kata (autograded)

Stwórz `course/assignments/kata_01.py`:

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

Stwórz `tests/test_kata_01.py` z małym kluczem odpowiedzi. Minimalne wymagane mapowanie:

* `"wrong_answer"` → `model_math`
* dowolny kod parse-format (np. `"missing_prefix"`, `"extra_whitespace"`, `"not_single_line"`, `"leading_zeros"`) → `model_format`
* `"invalid_example"` → `data_invalid`

Uruchom:

```bash
pytest -q
```

## 4) Lista kontrolna „Passed"

* Jeśli dwa uruchomienia eval się różnią, czy potrafisz wymienić jedyne poprawne przyczyny i wskazać dowody w `manifest.json`?
* Czy potrafisz wyjaśnić różnicę między **błędem formatu** a **błędem matematycznym** używając outcome codes?
* Czy potrafisz wyjaśnić, dlaczego „zmanipulowany dataset poprawił pass_rate" nie jest poprawą?

---

# Level 0.5: Selection to nie Learning (Build → Randomize → Measure Variance → Fix) — Loop B

## 1) Zmiana nastawienia

**Błędna intuicja:** „pass@N się poprawił, więc model się nauczył."
**Poprawny model mentalny:** „Loop B zmienia regułę decyzyjną nad próbkami (compute selection). Leżący u podstaw rozkład policy pozostaje niezmieniony."

## 2) Działania studenta (dokładne CLI)

### Build (działający samochód): uruchom selection demo na dostarczonym packu

```bash
python -m course.selection_demo \
  --dataset data/datasets/math_dev.jsonl \
  --samples data/rollouts/selection_pack_dev.jsonl \
  --n 4 \
  --outdir runs/l0_5_build_sel_n4

python -m course.inspect_run --run runs/l0_5_build_sel_n4 --top-k 5 --show 1
```

### Sabotage (zepsuj): wprowadź niedeterminizm do tie-breaking

Edytuj `course/assignments/selection_policy.py`.

Zmień `tie_break_key` na coś jawnie losowego (celowo):

```python
import random

def tie_break_key(sample):
    return (random.random(), sample.completion)
```

Teraz uruchom dokładnie to samo polecenie 5 razy:

```bash
for i in 1 2 3 4 5; do
  python -m course.selection_demo \
    --dataset data/datasets/math_dev.jsonl \
    --samples data/rollouts/selection_pack_dev.jsonl \
    --n 4 \
    --outdir runs/l0_5_sabotage_sel_random_$i
done
```

### Reflect (forensics): udowodnij istnienie wariancji używając artefaktów uruchomień

Oblicz hash każdego `results.jsonl`. Jeśli selection jest niedeterministyczne, hashe będą się różnić.

```bash
for i in 1 2 3 4 5; do
  python -c "import hashlib, pathlib; p=pathlib.Path('runs/l0_5_sabotage_sel_random_${i}/results.jsonl'); print(${i}, hashlib.sha256(p.read_bytes()).hexdigest())"
done
```

## 3) Zadanie Capstone (zestaw umiejętności)

**Budżet pokręteł:** Zmieniaj tylko `course/assignments/selection_policy.py` i jego testy.

### Repair + Lock

1. Usuń losowość. Zaimplementuj deterministyczny tie-break, który preferuje:

   * wyższe `sum_logprob` (jeśli obecne)
   * następnie krótsze uzupełnienie
   * następnie leksykograficzny tekst

Przykładowy deterministyczny klucz:

```python
def tie_break_key(sample):
    lp = sample.sum_logprob
    lp_key = 0.0 if lp is None else -float(lp)   # ujemne, żeby "wyższy logprob" sortował się pierwszy
    return (lp_key, len(sample.completion), sample.completion)
```

2. Dodaj `tests/test_selection_policy.py`:

   * test determinizmu: powtarzane wywołania wybierają ten sam indeks
   * test tie-break: dwie próbki o równej nagrodzie wybierają zamierzoną

Uruchom:

```bash
pytest -q
python -m course.selection_demo --dataset data/datasets/math_dev.jsonl --samples data/rollouts/selection_pack_dev.jsonl --n 4 --outdir runs/l0_5_fixed_sel_n4
```

## 4) Lista kontrolna „Passed"

* Dlaczego Loop B może poprawić wyjścia bez zmiany rozkładu policy?
* Jakie dokładnie „pokrętło" przekręciłeś podczas sabotażu i jak to udowodniłeś używając hashy/artefaktów?
* Dlaczego kod selection musi być deterministyczny w produkcji?

---

# Level 1: REINFORCE pod mikroskopem (Build → Over-optimize → Diagnose) — Loop C (toy)

## 1) Zmiana nastawienia

**Błędna intuicja:** „RL to supervised learning z dodatkowymi krokami."
**Poprawny model mentalny:** „RL popycha prawdopodobieństwa. Kierunek aktualizacji jest kontrolowany przez advantage = reward − baseline. Baseline redukują wariancję, nie cel."

## 2) Działania studenta (dokładne CLI)

### Build (działający samochód): stabilne uczenie z baseline

```bash
python -m course.bandit_train --steps 200 --seed 0 --lr 0.5 --baseline --outdir runs/l1_build_bandit
```

### Sabotage (zepsuj): nadmierna optymalizacja i/lub odwrócone uczenie

Uruchom dwa eksperymenty sabotażowe:

**(A) Zbyt agresywny learning rate**

```bash
python -m course.bandit_train --steps 200 --seed 0 --lr 2.0 --baseline --outdir runs/l1_sabotage_lr2
```

**(B) Ujemny learning rate (ucz się w złym kierunku)**

```bash
python -m course.bandit_train --steps 200 --seed 0 --lr -0.5 --baseline --outdir runs/l1_sabotage_lrneg
```

### Reflect (forensics): obserwuj mechanizm krok po kroku

```bash
python -m course.bandit_train --steps 30 --seed 0 --lr 0.5 --baseline --slow --outdir runs/l1_build_slow
python -m course.bandit_train --steps 30 --seed 0 --lr -0.5 --baseline --slow --outdir runs/l1_sabotage_slow_lrneg
```

## 3) Zadanie Capstone (zestaw umiejętności)

**Budżet pokręteł:** Możesz tworzyć pliki pod `notes/` i `course/assignments/`. Nie modyfikuj `course/bandit_train.py`.

Stwórz `notes/reinforce_forensics.md`:

* wybierz 10 kolejnych wpisów z wyjścia `--slow` (copy/paste)
* dla każdego wpisu zanotuj:

  * action, reward, baseline, znak advantage
  * co musi się stać z prawdopodobieństwem tej akcji jako następne

Następnie napisz 1 akapit wyjaśniający, dlaczego uruchomienia sabotażowe zachowują się inaczej.

## 4) Lista kontrolna „Passed"

* Jeśli advantage jest ujemny, co dzieje się z prawdopodobieństwem spróbkowanej akcji?
* Dlaczego baseline redukuje szum, ale nie zmienia „kierunku" uczenia?
* Czego ujemny learning rate nauczył cię o „popychaniu prawdopodobieństw"?

---

# Level 2: Credit Assignment + granica Token/Tekst (Build → Remove Credit → Observe Failure → Fix)

## 1) Zmiana nastawienia

**Błędna intuicja:** „Tylko ostatni token/akcja ma znaczenie, bo nagroda jest na końcu."
**Poprawny model mentalny:** „W sekwencyjnym generowaniu wczesne decyzje kształtują późniejsze stany. Końcowa nagroda staje się sygnałem treningowym stosowanym wzdłuż spróbkowanej trajektorii (credit assignment). Również: policy działa na tokenach; weryfikator działa na tekście."

## 2) Działania studenta (dokładne CLI)

### Build (działający samochód): obserwuj granicę tokenów

```bash
python -m course.token_inspect "Final: 323"
python -m course.token_inspect "Final:  323"
python -m course.token_inspect "Final:\n323"
python -m course.token_inspect "Final: 0323"
```

## 3) Zadanie Capstone (zestaw umiejętności)

**Budżet pokręteł:** Stwórz `course/assignments/two_step_mdp_demo.py` i notatki. Nie zmieniaj scorera.

### Build: zaimplementuj poprawny dwukrokowy REINFORCE

Stwórz `course/assignments/two_step_mdp_demo.py`:

* krok 1 wybierz A/B
* krok 2 wybierz X/Y uwarunkowane krokiem 1
* nagroda tylko na końcu
* aktualizuj **obie** policy kroków używając tej samej końcowej nagrody (z baseline opcjonalnie)

Uruchom:

```bash
python course/assignments/two_step_mdp_demo.py --steps 200 --seed 0 --baseline --outdir runs/l2_build_two_step
```

### Sabotage: usuń credit assignment dla kroku 1

Edytuj swój skrypt tak, aby **tylko krok 2 był aktualizowany** (krok 1 pozostaje losowy). Uruchom ponownie:

```bash
python course/assignments/two_step_mdp_demo.py --steps 200 --seed 0 --baseline --outdir runs/l2_sabotage_no_credit_step1
```

### Reflect + Fix: przywróć aktualizacje dla kroku 1 i uruchom ponownie

```bash
python course/assignments/two_step_mdp_demo.py --steps 200 --seed 0 --baseline --outdir runs/l2_fixed_two_step
```

Napisz `notes/credit_assignment.md`:

* wyjaśnij co zawiodło w sabotażu i dlaczego
* dołącz jeden diagram: tokeny → detokenizacja → tekst → parsowanie → nagroda

## 4) Lista kontrolna „Passed"

* Dlaczego krok 1 może mieć znaczenie, nawet gdy nagroda przychodzi na końcu?
* Co to jest macro-action vs micro-action w kontekście LLM?
* Dlaczego parsowanie/formatowanie jest częścią granicy środowiska?

---

# Level 3: KL jako smycz (Build → Drop the Leash → See Drift Preference)

## 1) Zmiana nastawienia

**Błędna intuicja:** „KL to dekoracyjna matematyka."
**Poprawny model mentalny:** „KL to praktyczna smycz: poprawiaj nagrodę, ale karz za odchodzenie zbyt daleko od rozkładu referencyjnego."

## 2) Działania studenta (dokładne CLI)

### Build (działający samochód): uruchom demo kompromisu KL

```bash
python -m course.kl_tradeoff_demo --plot --outdir runs/l3_build_kl_demo
```

## 3) Zadanie Capstone (zestaw umiejętności)

**Budżet pokręteł:** Stwórz mały skrypt assignment + notatki. Nie modyfikuj głównego kodu demo.

### Build: zaimplementuj selection z regularyzacją KL na małej syntetycznej tabeli

Stwórz `course/assignments/kl_regularized_choice.py`:

* zahardcoduj 6 „kandydatów", każdy z `(reward, kl)`
* zaimplementuj regułę wyboru: maksymalizuj `reward - beta * kl`
* wypisz wybranego kandydata dla `beta = 0.1` i `beta = 1.0`

Uruchom:

```bash
python course/assignments/kl_regularized_choice.py > runs/l3_build_kl_choice.txt
```

### Sabotage: usuń smycz (beta = 0)

Zmodyfikuj skrypt tak, aby beta = 0 i uruchom ponownie:

```bash
python course/assignments/kl_regularized_choice.py > runs/l3_sabotage_no_kl.txt
```

### Reflect

Stwórz `notes/kl_tradeoff.md`:

* wyjaśnij dlaczego „beta = 0" to atrakcyjna katastrofa
* połącz to z tym, dlaczego nieograniczona optymalizacja ma tendencję do preferowania ekstremalnych rozwiązań o wysokim dryfcie

## 4) Lista kontrolna „Passed"

* Co KL ogranicza prostym językiem?
* Dlaczego usunięcie preferencji KL może prowadzić do zachowań reward-hack-y (nawet jeśli reward rośnie)?
* Co jest „referencją" w tym modelu mentalnym?

---

# Level 4: Reward = Spec (Build Golden Gates → Loosen Rule → Catch False Positives → Fix + Version)

## 1) Zmiana nastawienia

**Błędna intuicja:** „Scorer to tylko asercja."
**Poprawny model mentalny:** „Scorer definiuje zadanie i musi być traktowany jak produkcyjne oprogramowanie pomiarowe."

## 2) Działania studenta (dokładne CLI)

### Build (działający samochód): zwaliduj scorer przeciwko golden

```bash
python -m course.validate_scorer \
  --dataset data/datasets/math_dev.jsonl \
  --golden data/golden/golden_correct.jsonl

python -m course.validate_scorer \
  --dataset data/datasets/math_dev.jsonl \
  --golden data/golden/golden_exploits.jsonl
```

## 3) Zadanie Capstone (zestaw umiejętności)

**Budżet pokręteł:** Dozwolone zmiany: `course/core/scoring.py`, `tests/`, `data/golden/`, `notes/`.
(Nie dotykaj kodu selection ani learning w tym poziomie.)

### Build: sondy black-box (przed czytaniem kodu)

Stwórz `notes/reward_spec_blackbox.md`:

* napisz reguły formatu, które twoim zdaniem istnieją
* dołącz 8 ciągów sond i przewidywane outcome codes

### Sabotage: poluzuj jedną regułę w scorerze

Wybierz jedną ścisłą regułę w `course/core/scoring.py` i celowo ją osłab (przykłady):

* pozwól na dodatkowe whitespace po `Final:`
* pozwól na wiodące zera
* pozwól na wieloliniowe wyjścia

Teraz uruchom ponownie sprawdzenia golden i obserwuj awarie:

```bash
python -m course.validate_scorer \
  --dataset data/datasets/math_dev.jsonl \
  --golden data/golden/golden_exploits.jsonl
```

### Repair + Lock: przywróć ścisłość, następnie rozszerz testy

1. Przywróć poprawne zachowanie.
2. Dodaj `tests/test_reward_regressions.py` z **co najmniej 6** przypadkami:

   * 2 przypadki „znane poprawne"
   * 4 przypadki exploit/edge (sztuczki formatowania)
3. Dodaj `data/golden/golden_exploits_extra.jsonl` z **co najmniej 5** nowymi przypadkami exploit.
4. Jeśli zachowanie reward zmieniło się względem baseline, podbij `SCORER_VERSION`.

Zwaliduj:

```bash
pytest -q
python -m course.validate_scorer --dataset data/datasets/math_dev.jsonl --golden data/golden/golden_exploits_extra.jsonl
```

## 4) Lista kontrolna „Passed"

* Co to jest false positive vs false negative w tym weryfikatorze?
* Dlaczego musisz wersjonować scorer przy zmianie zachowania i przestać porównywać stare numery uruchomień?
* Dlaczego „bardziej łagodne parsowanie" to zazwyczaj pułapka?

---

# Level 5: Goodhart Dungeon (Build a Naive Verifier → Hack It → Patch the Class)

## 1) Zmiana nastawienia

**Błędna intuicja:** „Reward hacking oznacza, że optymalizator jest zły."
**Poprawny model mentalny:** „Optymalizatory wykorzystują proxy. Jeśli specyfikacja ma luki, optymalizacja je znajdzie. Napraw mechanizmy, następnie zablokuj testami."

## 2) Działania studenta (dokładne CLI)

### Build baseline context (opcjonalne, ale ugruntowujące)

```bash
python -m course.eval \
  --dataset data/datasets/math_dev.jsonl \
  --completions data/rollouts/frozen_rollouts_dev.jsonl \
  --outdir runs/l5_build_eval_context

python -m course.inspect_run --run runs/l5_build_eval_context --only-fails --top-k 5 --show 2
```

## 3) Zadanie Capstone (zestaw umiejętności)

**Budżet pokręteł:** Dodaj nowe pliki assignment + notatki + testy. Unikaj edycji prawdziwego scorera, chyba że traktujesz to jako zmianę instrumentu.

### Build: napisz celowo naiwne demo weryfikatora

Stwórz `course/assignments/hackable_scorer_demo.py`:

* reward = 1 jeśli oczekiwana liczba całkowita pojawia się gdziekolwiek w uzupełnieniu (naiwny styl substring/regex)
* pokaż, że „działa" na uczciwych uzupełnieniach

### Sabotage: wygeneruj 5 hacków

Stwórz 5 oszukujących uzupełnień, które uzyskują wynik 1 bez bycia faktycznie poprawną odpowiedzią (np. number spray).

### Repair: załataj klasę exploitów

Zmodyfikuj logikę demo weryfikatora, aby zamknąć klasę exploitów, nie tylko jeden ciąg.

Napisz `notes/red_team_report.md`:

* 5 ciągów exploit
* dlaczego zadziałały
* strategia łatki
* jakie testy by to permanentnie zablokowały

## 4) Lista kontrolna „Passed"

* Podaj konkretny przykład „proxy w górę, prawdziwy cel w dół."
* Co oznacza „załataj klasę, nie ciąg"?
* Dlaczego to ma znaczenie dla Loop C później?

---

# Level 6: Komitet promocyjny (trzy kontrolowane pułapki) — Integruj A/B/C

## 1) Zmiana nastawienia

**Błędna intuicja:** „Wyższa liczba = lepiej."
**Poprawny model mentalny:** „Promowalna poprawa musi być porównywalna pod Locked Room i przypisywalna do dokładnie jednego pokrętła."

## 2) Działania studenta (dokładne CLI)

### Pułapka 1 (Sabotage): nieważna promocja przez zmianę instrumentu (przerwanie porównywalności Loop A)

Użyj swojego wcześniejszego uruchomienia ze zmanipulowanym datasetem (lub zrób świeże), następnie przepuść przez bramkę:

```bash
python -m course.gate \
  --baseline runs/l0_build_eval \
  --candidate runs/l0_sabotage_eval_tampered \
  --min-delta 0.00
```

### Pułapka 2: poprawa selection bez learning (Loop B)

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

### Pułapka 3: learning zmienia zachowanie (mechanizm Loop C)

```bash
python -m course.bandit_train --steps 200 --seed 1 --lr 0.5 --baseline --outdir runs/l6_trap_learning
```

## 3) Zadanie Capstone (zestaw umiejętności)

Napisz `notes/promotion_memo.md` (≤ 1 strona):

* Sekcja dla każdej pułapki:

  * pętla (A/B/C)
  * przekręcone pokrętło
  * czy porównanie jest ważne
  * dowody (zacytuj odpowiednią część `manifest.json` / wersję scorera / zmianę parametru)
* Końcowy akapit: co byś PROMOWAŁ vs ODRZUCIŁ dziś i dlaczego

## 4) Lista kontrolna „Passed"

* Jeśli pass_rate jest wyższy, jak udowodnisz, że to nie jest naruszenie Locked Room?
* Jak rozróżniasz poprawę selection od poprawy learning używając tylko artefaktów?
* Dlaczego Loop C nieuchronnie tworzy nowe tryby awarii, których zamrożone rollouts nie mogą w pełni ujawnić?

---

## Jeśli poprawnie ukończysz tę ścieżkę

Będziesz miał podstawową umiejętność, do której ten kurs dąży: zdolność do uruchamiania i interpretowania eksperymentów RL-for-LLMs bez mylenia **pomiaru**, **selection** i **learning**, oraz bez ufania nieważnym porównaniom.
