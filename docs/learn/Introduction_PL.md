# Wprowadzenie

---

## Dlaczego to istnieje

Ten kurs uczy dyscypliny pomiaru, która zapobiega kosztownym błędom. W praktyce problemy RL rzadko wynikają z algorytmu, a częściej z ewaluacji: zmieniłeś scorer, dataset albo selection zamiast poprawić model, nagroda została zhakowana, a wyniki są niepowtarzalne. Tutaj traktujesz nagrodę jak specyfikację (testujesz ją jak kod), odróżniasz realną poprawę od artefaktów (Zasada Locked Room) i rozumiesz *dlaczego* RL działa (policy gradient, credit assignment, KL) przez małe eksperymenty Build/Sabotage/Reflect/Repair. Nie wyjdziesz z PPO, ale wyjdziesz z umiejętnością krytycznej oceny systemów RLHF: czy metryka jest wiarygodna, czy poprawa to learning czy selection i co się stanie, gdy optymalizacja znajdzie luki.

---

## Podstawowe pojęcia

Duży Model Językowy (Large Language Model, LLM) definiuje **warunkowy rozkład prawdopodobieństwa $P(y | x)$** nad uzupełnieniami $y$; tokeny są próbkowane sekwencyjnie zgodnie z tym rozkładem.

> **Tłumaczenie matematyczne:** $P(y | x)$ czytamy jako „prawdopodobieństwo $y$ pod warunkiem $x$". Oddaje to ideę, że wyjście $y$ zależy całkowicie od kontekstu wejściowego $x$.

Reinforcement Learning dla LLM (RL-for-LLMs) stosuje się tam, gdzie nie ma jednoznacznej etykiety poprawnego wyjścia, ale da się przekazać **informację zwrotną** o jakości wyniku.

Ta informacja zwrotna może przyjmować różne formy:

* Binarne wskaźniki poprawności („Poprawne / niepoprawne")
* Preferencje porównawcze („To jest lepsze od tamtego")
* Weryfikacja zgodności z regułami („To przestrzega reguł")
* Zgodność ze specyfikacją („To pasuje do specyfikacji")
* Oceny bezpieczeństwa („To jest bezpieczne")

W tym kursie mechanizm informacji zwrotnej jest implementowany jako **weryfikator** (scorer). To deterministyczny program o dwóch rolach:

1. **Instrument pomiarowy**: obiektywnie raportuje, czy model odniósł sukces.
2. **Bramka nagrody**: definiuje nagrody środowiska i kształtuje zachowanie.

W projekcie bazowym weryfikator ocenia, czy uzupełnienie modelu zawiera poprawnie sformatowaną końcową liczbę całkowitą odpowiadającą oczekiwanej odpowiedzi.

Ta prostota jest celowa: deterministyczna informacja zwrotna odsłania mechanizm bez szumu wyuczonego reward modelu.

Fundamentalna zasada przewodnia tego kursu brzmi:

**Mechanizmy nagrody i ewaluacji funkcjonują jako instrumenty pomiarowe.**
Jeśli twój instrument pomiarowy nie jest precyzyjny, twój system reinforcement learning będzie pewnie optymalizował w kierunku bezsensownych celów.

Najpierw budujemy instrument pomiarowy, potem używamy go w trzech trybach operacyjnych.

---

## Praktyczne przygotowanie do zadań

Zanim zaczniesz poziomy projektu, pamiętaj o kilku konkretach:

- **Wersja Pythona:** używaj Pythona 3.10+ (jeśli `python3` jest starszy, użyj `python3.11` albo `poetry run`).
- **Jak uruchamiać:** skrypty uruchamiasz jako moduły (np. `python -m course.eval`). Z Poetry: `poetry run python -m course.eval`.
- **Gdzie trafiają wyniki:** przebiegi zapisują się w `runs/<name>/` i zwykle zawierają `manifest.json`, `results.jsonl`, `summary.json`, `summary.md`.
- **Gdzie edytujesz:** powierzchnie studenckie są w `course/assignments/`, refleksje w `notes/`, testy w `tests/`.
- **Oczekiwana deterministyczność:** przebiegi powinny być powtarzalne; w razie wątpliwości haszuj `results.jsonl`.
- **Opcjonalne wykresy:** `--plot` w demo KL wymaga matplotlib; jeśli brak, rysunek zostanie pominięty.
- **Opcjonalne próbkowanie Groq:** wpisz `GROQ_API_KEY` do pliku `.env` i użyj `python -m course.rollout_sample`, aby wygenerować rollouty z realnego modelu.

## Środowisko: kontrolowany system promptów i weryfikacji

Klasyczna teoria reinforcement learning opisuje „środowisko" jako Markov Decision Process (MDP), który dostarcza obserwacji i nagród. W reinforcement learning dla LLM, szczególnie dla zadań jednoturowych, „środowisko" zazwyczaj składa się z następujących komponentów:

1. **Rozkład promptów** $p(x)$ (twój dataset)
2. **Policy** $\pi_\theta(y | x)$ (LLM sparametryzowany przez $\theta$, który próbkuje uzupełnienia)
3. **Funkcja nagrody** $R(x, y)$ (weryfikator lub reward model), która mapuje pary (prompt, wyjście) na skalarne nagrody
4. Opcjonalne ograniczenia (reguły formatu, reguły bezpieczeństwa itp.)

> **Tłumaczenie matematyczne:**
> * $p(x)$: Jak często różne prompty pojawiają się w świecie rzeczywistym (lub w twoim datasecie).
> * $\pi_\theta(y | x)$: Sam model AI. $\theta$ reprezentuje miliardy parametrów (wag) wewnątrz sieci neuronowej.
> * $R(x, y)$: Zbiór reguł. Przyjmuje prompt i odpowiedź, odczytuje je i zwraca wynik.

Ten kurs czyni te komponenty jawnymi i ściśle zdefiniowanymi, tworząc uproszczony MDP, gdzie:
- Przestrzeń stanów: prompt $x$
- Przestrzeń akcji: przestrzeń możliwych uzupełnień $y$
- Dynamika przejść: deterministyczna (jednoturowa)
- Nagroda: $R(x, y) \in \{0, 1\}$ dla tego binarnego zadania weryfikacji

### Specyfikacja jako projektowanie środowiska

Wymóg formatu „Wypisz dokładnie jedną linię: `Final: <int>`" nie jest jedynie preferencją stylistyczną. Stanowi **projektowanie środowiska**.

Specyfikacja, którą egzekwujesz, determinuje wyniki behawioralne:
- Nagradzanie „dowolnego ciągu zawierającego gdzieś poprawną liczbę" produkuje jeden wzorzec zachowania.
- Nagradzanie „dokładnie jednej linii z określonym prefiksem i ściśle sformatowaną liczbą całkowitą" produkuje całkowicie inny wzorzec.

Dlatego ścisłość specyfikacji nie jest pedanterią; reprezentuje kontrolowaną metodologię eksperymentalną.

---

## Centralny kontrakt

Wszystkie operacje w tym systemie obracają się wokół pojedynczego interfejsu implementującego funkcję nagrody:

**`score(example, completion) -> {reward, details}`**

Formalnie implementuje to $R: (X, Y) \to \mathbb{R}$, gdzie:
* `example` $\in X$ reprezentuje jeden element datasetu (prompt + oczekiwana odpowiedź + identyfikator)
* `completion` $\in Y$ reprezentuje wygenerowane wyjście modelu
* `reward` $\in \{0, 1\}$ to skalarny sygnał nagrody (binarny w tym frameworku)
* `details` zawiera informacje diagnostyczne (status sukcesu parsowania, kody błędów, wyodrębnione wartości itp.)

> **Tłumaczenie matematyczne:** $R: (X, Y) \to \mathbb{R}$ oznacza „R jest funkcją, która mapuje parę (ze zbioru X i zbioru Y) na liczbę rzeczywistą." W naszym przypadku liczba rzeczywista to po prostu 0 lub 1.

Ten kontrakt służy jako „instrument pomiarowy." Wszystkie pozostałe operacje stanowią różne metody **wykorzystania** tego instrumentu do obliczania celów, metryk lub sygnałów uczenia.

---

## Ramy koncepcyjne: trzy odrębne pętle operacyjne

Wszystkie trzy pętle wykonują warianty tego samego podstawowego procesu:

**prompt → uzupełnienie(a) → score → wynik(i) numeryczne**

To podobieństwo może prowadzić do zamieszania koncepcyjnego. Ten kurs podkreśla rozróżnienia, aż staną się intuicyjne:

### Loop A — Evaluation (pomiar)

**Cel:** „Jaki jest aktualny stan wydajności systemu?"

Wybierasz zestaw promptów $\{x_1, x_2, \dots, x_n\}$, uzyskujesz uzupełnienia $y_i \sim \pi_\theta(\cdot|x_i)$ (często zamrożone lub wcześniej zarejestrowane na początku), wykonujesz funkcję `score` i agregujesz wyniki.

Ta pętla nie obejmuje optymalizacji, uczenia ani wyrafinowanych technik. Jest to czysty pomiar.

**Formalnie obliczasz statystyki empiryczne:**

* **Średnia nagroda**: $\bar{R} = (1/N) \sum_{i=1}^N R(x_i, y_i)$
* **Pass rate** (dla binarnych nagród): $\hat{P}_{\text{pass}} = (1/N) \sum_{i=1}^N \mathbb{1}[R(x_i, y_i) = 1]$

> **Tłumaczenie matematyczne:**
> * $\Sigma$ (Sigma) oznacza „zsumuj wszystkie wartości".
> * $\mathbb{1}[\dots]$ to **funkcja wskaźnikowa**. Równa się $1$, jeśli warunek wewnątrz jest prawdziwy, i $0$, jeśli fałszywy.
> * Zatem $\hat{P}_{\text{pass}}$ to po prostu zliczanie, ile razy model odpowiedział poprawnie, podzielone przez całkowitą liczbę prób $N$.
* **Rozkład trybów awarii**: Grupowanie według typu błędu i obliczanie proporcji

Praktyczne wyjścia obejmują:

* Pass rate lub średnią nagrodę (empiryczne estymaty $\mathbb{E}[R]$)
* Rozkład trybów awarii (niepoprawne odpowiedzi versus błędy formatowania versus błędy parsowania)
* Konkretne przykłady dostępne do inspekcji

Ta pętla uczy prawdopodobnie najważniejszej umiejętności w reinforcement learning, która nie obejmuje matematyki:
**Systematyczne kategoryzowanie trybów awarii zamiast dokonywania intuicyjnych ocen.**

Jeśli nie potrafisz wyartykułować przyczyny niepowodzenia, nie możesz ulepszać systematycznie.

---

### Loop B — Selection (bez learningu)

**Cel:** „Czy wydajność może być zwiększona poprzez zwiększone zasoby obliczeniowe w czasie inferencji?"

Zamiast generować pojedyncze uzupełnienie na prompt, generujesz **N próbek** $\{y_1, y_2, \dots, y_n\} \sim \pi_\theta(\cdot|x)$ z *tego samego* modelu dla każdego promptu, oceniasz wszystkie próbki i wybierasz tę z najwyższym wynikiem.

**Formalnie operacja selection to:**

**$y^* = \text{argmax}_{y \in \{y_1,\dots,y_n\}} R(x, y)$**

> **Tłumaczenie matematyczne:** $\text{argmax}$ pyta „który argument (wejście) produkuje maksymalne wyjście?"
> Prostym językiem: Generujemy $n$ opcji. Oceniamy je wszystkie. Wybieramy tę ($y^*$), która uzyskała najwyższy wynik. Nie zmieniamy modelu; po prostu wybieramy jego najlepszą pracę.

To podejście jest znane jako Best-of-N sampling (wraz z powiązanymi technikami: reranking, rejection sampling, verifier-guided search).

Kluczowa koncepcja:

* Zdolności modelu (rozkład $\pi_\theta$) nie ulegają poprawie.
* Po prostu eksplorujesz większą część tego, co model już potrafi wygenerować.
* To skalowanie obliczeniowe w czasie inferencji, nie poprawa w czasie treningu.

**Sformalizowane metryki:**

* **pass@1** = $\mathbb{P}[R(x, y) = 1 \mid y \sim \pi_\theta(\cdot|x)]$ — prawdopodobieństwo, że pojedyncza próbka odniesie sukces
* **pass@k** = $\mathbb{P}[\max\{R(x, y_1), \dots, R(x, y_k)\} = 1 \mid y_1,\dots,y_k \sim \pi_\theta(\cdot|x)]$ — prawdopodobieństwo, że co najmniej jedna z k próbek odniesie sukces

> **Tłumaczenie matematyczne:** $ \sim $ oznacza „próbkowane z". $\pi_\theta(\cdot|x)$ to rozkład modelu.
> pass@k pyta: „Jeśli pozwolę modelowi próbować $k$ razy, jakie jest prawdopodobieństwo, że *co najmniej jedna* z tych prób będzie poprawna?"

Selection podnosi pass@N; pass@1 zwykle się nie zmienia, bo samo $\pi_\theta$ pozostaje niezmienione.

Praktyczne rozważania:

* Selection jest często najszybszą metodą bezpiecznego wdrażania ulepszeń wydajności.
* Jednak wiąże się z kosztami obliczeniowymi runtime i opóźnieniem.
* Dodatkowo może maskować fakt, że bazowa policy pozostaje słaba.

---

### Loop C — Learning (trening)

**Cel:** „Czy wydajność pass@1 może być poprawiona poprzez modyfikację samej policy?"

Ta pętla wykonuje te same procedury próbkowania i oceniania, ale dodaje jeden krytyczny krok:

**Aktualizacja parametrów modelu $\theta$ tak, aby policy stała się bardziej prawdopodobna do generowania uzupełnień o wysokiej nagrodzie w przyszłych iteracjach.**

To jest miejsce, gdzie reinforcement learning „faktycznie zachodzi."

Nawet gdy nagroda jest prostą binarną wartością końcową 0/1, procedura aktualizacji musi przypisać zasługę sekwencji decyzji na poziomie tokenów, które wyprodukowały uzupełnienie.

Fundamentalna transformacja koncepcyjna to:

* Traktuj model jako **policy** $\pi_\theta(y | x)$ sparametryzowaną wagami $\theta$
* Zdefiniuj cel jako **oczekiwaną nagrodę** pod tą policy: $J(\theta) = \mathbb{E}_{x \sim p(x), y \sim \pi_\theta(\cdot|x)}[R(x, y)]$

> **Tłumaczenie matematyczne:**
> * $\mathbb{E}$ oznacza **Expectation** (matematyczną średnią).
> * Cel $J(\theta)$ to po prostu „średni wynik, którego oczekujemy od modelu."
> * $\nabla_\theta$ (Nabla lub Gradient) wskazuje kierunek najszybszego wzrostu. Chcemy wspinać się na wzgórze wyższych nagród.
* Zastosuj metody policy gradient do maksymalizacji $J(\theta)$ poprzez dostosowanie $\theta$ w kierunku $\nabla_\theta J(\theta)$

---

## Mechanizm leżący u podstaw: wzmacnianie udanych akcji

W ramach kursu mechanizm uczenia jest początkowo demonstrowany przy użyciu prostego problemu bandit zamiast pełnego LLM. To nie jest unikanie złożoności; to pedagogiczna strategia dla przejrzystości.

### Algorytm REINFORCE w skrócie

Algorytm REINFORCE (Williams, 1992) jest metodą Monte Carlo policy gradient. Gdy próbkujesz akcję i obserwujesz nagrodę:

* Jeśli nagroda była **lepsza niż oczekiwana**, zwiększ prawdopodobieństwo spróbkowanej akcji.
* Jeśli nagroda była **gorsza niż oczekiwana**, zmniejsz prawdopodobieństwo spróbkowanej akcji.

Policy gradient aktualizuje parametry według:

**$\nabla_\theta J(\theta) \approx (R - b) \cdot \nabla_\theta \log \pi_\theta(y | x)$**

> **Tłumaczenie matematyczne:** Ta formuła mówi: „Weź gradient log-prawdopodobieństwa akcji, którą właśnie podjęliśmy. Przeskaluj go tym, jak dobry był wynik ($R-b$). Jeśli wynik był lepszy niż zwykle (dodatni), pchaj gradient, aby ta akcja była *bardziej* prawdopodobna. Jeśli gorszy (ujemny), pchaj go, aby była *mniej* prawdopodobna."

gdzie:
* **$R$** to zaobserwowana nagroda za uzupełnienie $y$
* **$b$** to baseline (zazwyczaj estymowany jako średnia krocząca: $b \leftarrow (1-\alpha)b + \alpha R$)
* **$A = R - b$** to funkcja advantage
* Advantage określa *kierunek* i *wielkość* dostosowania prawdopodobieństwa

**Obliczanie baseline:**
W praktyce baseline jest często obliczany jako:
- **Średnia batcha**: $b = (1/N) \sum_i R_i$ po aktualnym batchu
- **Wykładnicza średnia krocząca**: $b_{t+1} = (1-\alpha) b_t + \alpha R_t$ z decay $\alpha \in (0,1)$
- **Value function**: W metodach actor-critic, $b = V_\phi(x)$ jest wyuczoną siecią neuronową

Znaczenie baseline:

* Surowe nagrody zawierają znaczny szum; baseline centrują sygnał uczenia wokół zera.
* Baseline redukują wariancję gradientu bez wprowadzania obciążenia (ponieważ $\mathbb{E}[b \cdot \nabla_\theta \log \pi_\theta] = 0$).
* Niższa wariancja umożliwia bardziej stabilne uczenie przy mniejszej liczbie próbek.

Zastosowanie do LLM:

* Uzupełnienie LLM $y = (y_1, y_2, \dots, y_t)$ składa się z sekwencyjnego łańcucha akcji na poziomie tokenów.
* Policy faktoryzuje się jako $\pi_\theta(y | x) = \prod_i \pi_\theta(y_i | x, y_{1:i-1})$
* Gradient $\nabla_\theta \log \pi_\theta(y | x) = \sum_i \nabla_\theta \log \pi_\theta(y_i | x, y_{1:i-1})$ podnosi log-prawdopodobieństwo każdego tokenu w trajektoriach o wysokiej nagrodzie

Dlatego konceptualnie Loop C wykonuje:
**próbkuj uzupełnienia $y \sim \pi_\theta(\cdot|x)$ → oceniaj $R(x,y)$ → oblicz advantage $A = R - b$ → aktualizuj $\theta \leftarrow \theta + \alpha \cdot A \cdot \nabla_\theta \log \pi_\theta(y|x)$**

---

## Regularyzacja: rola KL divergence w systemach RLHF

Jeśli optymalizujesz wyłącznie pod kątem nagrody, zapraszasz problematyczne zjawisko:

**Prawo Goodharta:** Gdy miara staje się celem, przestaje być dobrą miarą.

Optymalizatory są wysoce efektywne w identyfikowaniu luk w specyfikacji. Jeśli twoja specyfikacja nagrody jest niedoskonała (co jest nieuniknione), policy wykorzysta te niedoskonałości.

Jedną z najczęstszych technik stabilizacji jest **kara KL divergence**:

* Utrzymujesz referencyjną policy $\pi_{\text{ref}}$ (zazwyczaj oryginalny model po supervised fine-tuning).
* Karzesz wyuczoną policy $\pi_\theta$ za nadmierne odchylanie od referencji.

Ograniczony cel staje się:

**$J_{KL}(\theta) = \mathbb{E}_{x,y}[R(x, y)] - \beta \cdot \mathbb{E}_x[D_{KL}(\pi_\theta(\cdot|x) || \pi_{\text{ref}}(\cdot|x))]$**

> **Tłumaczenie matematyczne:**
> * $J_{KL}$ to nasz nowy, bezpieczniejszy cel.
> * składnik 1 ($\mathbb{E}[R]$): „Uzyskuj wysokie nagrody."
> * składnik 2 ($\beta \cdot D_{KL}$): „Minus kara za bycie dziwnym."
> * $D_{KL}$ mierzy „odległość" między nowym zachowaniem modelu a starym. Chcemy maksymalizować nagrodę, ale minimalizować odległość.

gdzie:
* **$D_{KL}$** to Kullback-Leibler divergence: $D_{KL}(\pi_\theta || \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi_\theta}[\log \pi_\theta(y|x) - \log \pi_{\text{ref}}(y|x)]$
* **$\beta$** to współczynnik kary KL kontrolujący siłę regularyzacji

Intuicja:

* Składnik nagrody mówi „poruszaj się w tym kierunku, aby maksymalizować R."
* Kara KL mówi „ale nie odchylaj się za daleko od $\pi_{\text{ref}}$."

To służy wielu celom poza „bezpieczeństwem":

* Zapobieganie katastrofalnym zmianom behawioralnym (mode collapse)
* Utrzymanie płynności językowej i spójności
* Redukcja nadmiernej optymalizacji na kruchych specyfikacjach nagrody
* Poprawa stabilności uczenia poprzez implicit trust regions

Demonstracja KL w kursie dostarcza przejrzysty model mentalny: w małej dyskretnej przestrzeni akcji możesz bezpośrednio obserwować granicę Pareto między „wyższą nagrodą" a „niższą dywergencją od referencji."

Ta krzywa kompromisu — w istocie problem optymalizacji z ograniczeniami — jest fundamentalna dla dostrajania rzeczywistych systemów RL-for-LLMs (PPO, RLHF, DPO).

---

## Krytyczny detal: przestrzeń tokenów versus przestrzeń tekstu

Znaczące źródło zamieszania w reinforcement learning dla LLM wynika z faktu, że *rozumujemy* w tekście, ale policy operuje na **tokenach**.

**Formalnie:** LLM operuje nad dyskretnym słownikiem $V$ (zazwyczaj $|V| \approx 30k-100k$ tokenów). Uzupełnienie jest sekwencją $y = (y_1, \dots, y_T)$, gdzie każdy $y_i \in V$, a policy jest autoregresyjnym rozkładem:

**$\pi_\theta(y | x) = \prod_{i=1}^T \pi_\theta(y_i | x, y_{1:i-1})$**

gdzie każde prawdopodobieństwo tokenu $\pi_\theta(y_i | \text{context})$ jest obliczane przez softmax po $V$:

**$\pi_\theta(y_i = v | \text{context}) = \frac{\exp(f_\theta(v, \text{context}))}{\sum_{v' \in V} \exp(f_\theta(v', \text{context}))}$**

> **Tłumaczenie matematyczne:** To jest funkcja **Softmax**.
> * $f_\theta$ wyprowadza surowe liczby (logity), które mogą być ujemne lub ogromne.
> * $\exp$ (wykładnicza) sprawia, że wszystko jest dodatnie.
> * Dzielenie przez sumę $\sum$ gwarantuje, że wszystkie liczby sumują się dokładnie do 1.0, czyniąc je poprawnymi prawdopodobieństwami.

Dwie konsekwencje, które powinny być zrozumiane przed przejściem do kodu implementacji:

1. **Małe różnice tekstowe mogą odpowiadać dużym różnicom w policy.**
   Znak spacji, nowa linia, inny wariant dwukropka lub „Final:323" versus „Final: 323" mogą tokenizować się inaczej i otrzymywać istotnie różne prawdopodobieństwa.

2. **Ścisła specyfikacja nagrody staje się mechanizmem kształtowania behawioralnego.**
   Jeśli nagroda jest przyznawana tylko za dokładnie `Final: <int>`, sygnał treningowy będzie silnie wzmacniał:

   * Poprawność formatowania
   * Odpowiednie zachowanie zatrzymania (brak generowania dodatkowych linii)
   * Produkcję poprawnej liczby całkowitej bez zbędnych znaków

Dlatego framework kursu obejmuje:

* Ścisłe wymagania parsowania
* Jawną kategoryzację błędów parsowania
* Przypadki testowe „golden exploit"

To nie są marginalne kwestie. Zapobiegają one oszukaniu twojego „instrumentu pomiarowego" przez sam proces optymalizacji, który wdrażasz.

---

## Dyscyplina pomiarowa: zasada „zamkniętego pokoju"

Ten aspekt może wydawać się nietechniczny, dopóki nie doświadczysz jego naruszenia na własnej skórze:

Jeśli zmodyfikujesz instrument pomiarowy, zmieniłeś to, co wyniki numeryczne oznaczają.

Dlatego porównując dwa przebiegi eksperymentalne, musisz utrzymać kontrolowane środowisko:

* Identyczny podział datasetu
* Identyczne prompty
* Identyczna specyfikacja i wersja scorera
* Identyczne ustawienia próbkowania (jeśli stosowane jest stochastyczne próbkowanie)

Ta praktyka zapobiega najczęstszemu wzorcowi samooszukiwania w badaniach reinforcement learning:

> „Metryka się poprawiła! Zrobiliśmy postęp!"
> (W rzeczywistości zmodyfikowałeś prompt, poluzowałeś ograniczenia parsowania i zwiększyłeś liczbę próbek.)

Kurs egzekwuje tę dyscyplinę poprzez:

* Wersjonowanie scorera jak oprogramowanie
* Przechowywanie artefaktów uruchomień (wyniki, podsumowania, manifesty, sumy kontrolne)
* Utrzymywanie małego zbioru holdout, który nie jest używany do dostrajania
* Wykonywanie testów „golden" w celu weryfikacji, że scorer nie stał się przypadkowo permisywny

To właśnie przekształca twoją pracę kursową w spójną *narrację* zamiast zbioru odłączonych technik:
centralnym przedmiotem nie jest „implementacja algorytmu PPO" — jest nim **rygor epistemiczny**.

---

## Architektura systemu: zunifikowany model mentalny

Konceptualizuj system jako pojedynczy pipeline z trzema trybami operacyjnymi:

```
           ┌──────────────┐
           │  Dataset D    │   (prompty + oczekiwane odpowiedzi)
           └──────┬───────┘
                  │  próbkuj prompt x
                  v
           ┌──────────────┐
           │ Policy πθ     │   (rozkład LLM nad uzupełnieniami)
           └──────┬───────┘
                  │  generuj 1 lub N uzupełnień
                  v
           ┌──────────────┐
           │ Completions   │   (ciągi znaków / sekwencje tokenów)
           └──────┬───────┘
                  │  oceń każde uzupełnienie
                  v
           ┌──────────────┐
           │ Scorer S      │   (weryfikator; twój instrument pomiarowy)
           └──────┬───────┘
                  │  nagroda + szczegóły
                  v
           ┌──────────────┐
           │ Results       │   (rekordy per-example)
           └──────┬───────┘
                  │
      ┌───────────┼───────────┐
      │           │           │
      v           v           v
  Loop A       Loop B       Loop C
 (EVAL)     (SELECTION)   (LEARNING)
 podsumuj    wybierz       aktualizuj θ
 zbadaj      najlepsze     (często z regularyzacją KL)
```

Ten diagram reprezentuje kompletne ramy koncepcyjne. Każdy komponent w twoim repozytorium jest albo:

* Jednym z tych elementów architektonicznych,
* Albo zabezpieczeniem przed samooszukiwaniem się dotyczącym zmian eksperymentalnych.

---

## Przełożenie na praktykę w świecie rzeczywistym

Chociaż framework kursu jest minimalny, odzwierciedla rzeczywiste przepływy pracy w przemyśle:

* **Zespoły programistyczne przeznaczają znaczne zasoby na scorer i specyfikację nagrody**, ponieważ to jest miejsce, gdzie większość awarii systemu ma swoje źródło.
* **Metody selection** (best-of-N, reranking, verifier-guided search) są często wdrażane przed podejściami opartymi na treningu, ponieważ są bardziej interpretowalane i odwracalne.
* **Trening** jest kosztowny obliczeniowo i może wzmocnić błędy specyfikacji do poważnych patologii policy, więc jest podejmowany dopiero po tym, gdy pomiar jest niezawodny.
* **Regularyzacja KL i ograniczenia** są standardową praktyką, ponieważ nieograniczona optymalizacja często produkuje policy wykorzystujące luki w specyfikacji.
* **Inspekcja i zachowywanie artefaktów** są krytyczne, ponieważ awarie reinforcement learning rzadko manifestują się jako pojedynczy oczywisty błąd; zazwyczaj wyłaniają się jako kaskada małych odchyleń.

---

## Oczekiwane zrozumienie po tym wprowadzeniu

Zanim przejdziesz do kodu implementacji, powinieneś umieć wyartykułować zarówno koncepcyjnie, jak i matematycznie:

* „Scorer implementuje funkcję nagrody $R(x, y)$; jeśli $R$ się zmieni, optymalna policy $\pi^*$ się zmieni."
* „Loop A oblicza statystyki empiryczne $\mathbb{E}[R]$, Loop B wykonuje optymalizację w czasie inferencji $\max_{y \in \text{samples}} R(x,y)$, Loop C wykonuje optymalizację w czasie uczenia $\nabla_\theta \mathbb{E}[R]$."
* „Selection podnosi pass@N (istnienie dobrej próbki) bez poprawy pass@1 (jakość typowej próbki)."
* „REINFORCE aktualizuje log-prawdopodobieństwa używając gradientów ważonych przez advantage: $\Delta \theta \propto A \cdot \nabla_\theta \log \pi_\theta$."
* „Regularyzacja KL $D_{KL}(\pi_\theta || \pi_{\text{ref}})$ ogranicza policy do trust region wokół referencji."
* „Tokenizacja czyni przestrzeń akcji dyskretną; różnice w formatowaniu jak whitespace wpływają na $\pi_\theta(y|x)$ nietrywialnie."

Te koncepcyjne i matematyczne ramy są wystarczające, aby podejść do kodu jako szczegółu implementacyjnego, a nie jako nieprzeniknionego labiryntu.

Zasadniczo zbudowałeś kontrolowane środowisko laboratoryjne, gdzie reinforcement learning nie jest tajemniczą procedurą — jest rygorystycznym eksperymentem z:
- Skalibrowanym instrumentem pomiarowym ($R: X \times Y \to \mathbb{R}$)
- Systematycznym celem optymalizacji ($J(\theta) = \mathbb{E}[R]$)
- Precyzyjną definicją sukcesu (empiryczny pass rate na danych holdout)

---

## Od teorii do implementacji: skrypty kursu

Poprzednie sekcje ustanawiają formalne ramy. Ta sekcja mapuje te koncepcje na konkretne skrypty, które będziesz wykonywać. Wszystkie trzy skrypty wykorzystują ten sam scorer — funkcję weryfikującą, czy wyjście modelu zawiera dokładny format `Final: <int>` z poprawną liczbą całkowitą.

### Tabela mapowania koncepcji

| Koncepcja formalna | Termin implementacyjny | Lokalizacja w kodzie |
|-------------------|------------------------|---------------------|
| **Policy** $\pi_\theta(y \mid x)$ | „Model" | Wszystkie skrypty wywołują model do generowania wyjść |
| **Funkcja nagrody** $R(x, y)$ | „Scorer" | `score.py` — implementacja centralnego kontraktu |
| **Cel** $J(\theta) = \mathbb{E}[R]$ | „Performance" lub „pass rate" | `eval.py` oblicza ten empiryczny estymat |
| **Empiryczna średnia nagroda** $\bar{R}$ | „Pass rate" lub „mean reward" | statystyki podsumowujące `eval.py` |
| **Argmax selection** $y^* = \text{argmax } R$ | „Wybierz najlepsze" | `selection_demo.py` wybiera próbkę o najwyższym wyniku |
| **Aktualizacja policy gradient** $\nabla_\theta J$ | „Aktualizuj model" | `bandit_train.py` modyfikuje wagi modelu |
| **Advantage** $A = R - b$ | „Nagroda minus baseline" | sygnał uczenia `bandit_train.py` |
| **Kara KL divergence** $D_{KL}$ | „Nie odchodź za daleko" | `kl_tradeoff_demo.py` demonstruje to ograniczenie |

### Skrypt 1: Ewaluacja (`eval.py`) — Loop A

**Cel:** Określ aktualny stan wydajności systemu.

**Operacja formalna:** Oblicz empiryczną średnią nagrodę $\bar{R} = (1/N) \sum_{i=1}^N R(x_i, y_i)$.

**Co się dzieje:**
1. Wykonaj model na zestawie promptów
2. Oceń każdą wygenerowaną odpowiedź
3. Zagreguj wyniki do pass rate i rozkładu trybów awarii

**Przykładowe wyjście:**
```
Pass rate: 23/100 = 23%
Failures:
  - Wrong answer: 65
  - Bad format: 12
```

Ten skrypt wykonuje czysty pomiar. Brak optymalizacji. Brak treningu. Raportowany pass rate jest twoim empirycznym estymatem $\mathbb{E}[R]$.

---

### Skrypt 2: Selection (`selection_demo.py`) — Loop B

**Cel:** Zwiększ wydajność poprzez zwiększone obliczenia w czasie inferencji bez modyfikacji modelu.

**Operacja formalna:** Oblicz $y^* = \text{argmax}_{y \in \{y_1,\dots,y_n\}} R(x, y)$ po $N$ próbkach.

**Co się dzieje:**
1. Wygeneruj $N$ uzupełnień dla tego samego promptu
2. Oceń wszystkie uzupełnienia
3. Wybierz uzupełnienie o najwyższym wyniku

**Kluczowy insight:** Jeśli model odnosi sukces z prawdopodobieństwem $p = 0.23$ na próbkę:
- 1 próbka → 23% prawdopodobieństwa sukcesu
- 10 próbek → $1 - (1 - 0.23)^{10} \approx 93\%$ prawdopodobieństwa, że co najmniej jedna odniesie sukces

Rozkład modelu $\pi_\theta$ pozostaje niezmieniony. Eksplorujesz większą część jego istniejącej przestrzeni możliwości.

**Istotne metryki:**
* **pass@1** — prawdopodobieństwo, że pierwsza próbka odniesie sukces
* **pass@N** — prawdopodobieństwo, że co najmniej jedna z $N$ próbek odniesie sukces

Selection podnosi pass@N; pass@1 zwykle się nie zmienia, bo samo $\pi_\theta$ pozostaje niezmienione.

---

### Skrypt 3: Learning (`bandit_train.py`) — Loop C

**Cel:** Popraw pass@1 poprzez modyfikację samej policy.

**Operacja formalna:** Aktualizuj $\theta \leftarrow \theta + \alpha \cdot A \cdot \nabla_\theta \log \pi_\theta(y|x)$ gdzie $A = R - b$.

**Co się dzieje:**
1. Próbkuj uzupełnienie z aktualnej policy
2. Oceń je, aby uzyskać nagrodę $R$
3. Oblicz advantage: $A = R - b$ (nagroda względem oczekiwanej wartości baseline)
4. Jeśli $A > 0$: zwiększ prawdopodobieństwo tego uzupełnienia
5. Jeśli $A < 0$: zmniejsz prawdopodobieństwo tego uzupełnienia

To jest algorytm REINFORCE w działaniu. Parametry modelu $\theta$ są modyfikowane, aby uzupełnienia o wysokiej nagrodzie stały się bardziej prawdopodobne.

---

### Demonstracja kompromisu KL (`kl_tradeoff_demo.py`)

**Cel:** Wizualizuj kompromis nagroda-dywergencja sformalizowany jako $J_{KL}(\theta) = \mathbb{E}[R] - \beta \cdot D_{KL}(\pi_\theta || \pi_{\text{ref}})$.

W uproszczonej dyskretnej przestrzeni akcji ta demonstracja pozwala na bezpośrednią obserwację granicy Pareto między „wyższą nagrodą" a „niższą dywergencją od referencyjnej policy."

---

## Instrument pomiarowy w praktyce

Scorer stanowi implementację twojej funkcji nagrody $R(x, y)$. Jego zachowanie determinuje, na co celuje proces optymalizacji.

**Przykład ścisłości specyfikacji:**

Rozważ poluzowaną specyfikację: „nagradzaj dowolny ciąg zawierający poprawną liczbę."

Uzupełnienie takie jak:
```
Final: 42
But also 43 might work?
Actually I'm not sure
The answer could be 42
```

Ten ciąg zawiera „42" → $R = 1$ pod poluzowaną specyfikacją.

Jednak to wyjście nie jest poprawnym rozwiązaniem. Model wykorzystał lukę w specyfikacji nagrody.

**Kurs egzekwuje ścisłe specyfikacje:**

Nagroda wynosi 1 **tylko jeśli** wyjście to dokładnie:
```
Final: 42
```

Brak dodatkowych linii. Brak zbędnego tekstu. Jedna liczba całkowita w przepisanym formacie.

Ta ścisłość nie jest pedanterią — jest **projektowaniem środowiska**. Luźna specyfikacja pozwala na reward hacking. Ścisła specyfikacja zmusza model do wyuczenia zamierzonego zachowania.

---

## Rozróżnianie trzech pętli

Wszystkie trzy pętle mają powierzchowne podobieństwo:
- Przyjmują prompty
- Generują uzupełnienia
- Oceniają wyjścia
- Produkują wyniki numeryczne

To podobieństwo zasłania fundamentalne różnice:

| Pętla | Operacja | Co się zmienia |
|-------|----------|----------------|
| **A: Ewaluacja** | Zmierz aktualną wydajność | Nic |
| **B: Selection** | Wybierz najlepszą z $N$ próbek | Które wyjście jest wdrażane |
| **C: Learning** | Trenuj przez policy gradient | Parametry modelu $\theta$ |

Jeśli nie potrafisz zidentyfikować, która pętla zmieniła się między dwoma przebiegami eksperymentalnymi, twoje porównanie jest nieważne.

---

## Lista kontrolna weryfikacji przed przejściem do kodu

Wyartykułuj następujące stwierdzenia, aż staną się intuicyjne:

1. **„Scorer jest moim instrumentem pomiarowym. Jeśli go zmodyfikuję, zmieniłem to, co mierzę."**

2. **„Loop A mierzy $\mathbb{E}[R]$. Loop B oblicza $\max R$ po próbkach. Loop C optymalizuje $\nabla_\theta \mathbb{E}[R]$."**

3. **„Selection podnosi pass@N, a trening poprawia pass@1."**

4. **„Advantage $A = R - b$ jest sygnałem uczenia. Dodatni advantage zwiększa prawdopodobieństwo; ujemny advantage je zmniejsza."**

5. **„Kara KL $D_{KL}(\pi_\theta || \pi_{\text{ref}})$ zapobiega reward hacking poprzez ograniczanie dryfu policy."**

6. **„Tekst ma tokeny. Drobne różnice formatowania odpowiadają znacznym różnicom prawdopodobieństwa w przestrzeni tokenów."**

Jeśli te stwierdzenia są jasne, posiadasz ramy koncepcyjne, aby podejść do implementacji jako szczegółu inżynieryjnego, a nie jako nieprzejrzystego systemu.

Kod implementuje te koncepcje. Model mentalny determinuje, czy rozumiesz, co kod robi.
