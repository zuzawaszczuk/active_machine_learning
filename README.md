# active_machine_learning

### Uruchomienie testów

```
python -m pytest tests/
```
### Uruchomienie eksperymentu 

- `--epochs 100` — liczba iteracji treningu modelu  
- `--n_runs 1` — liczba powtórzeń eksperymentu  
- `--batch_size 32` — rozmiar batcha ile próbek dobierano w iteracji 
- `--dataset cifar` — wybór zbioru danych (CIFAR)  
- `--compression 20` — stopień redukcji zbioru treningowego 80% klas by utworzyć niezbalansowanie
```
uv sync
python3 src/main.py --epochs 100 --n_runs 1 --batch_size 32 --dataset cifar --compresion 20
```

### Opis Projektu

Tematem projektu jest analiza zastosowania metod uczenia aktywnego w wybranym problemie uczenia maszynowego.

Głównym celem uczenia aktywnego jest ograniczenie kosztów związanych z procesem etykietowania danych. Problem ten jest szczególnie istotny w dwóch scenariuszach opisywanych w literaturze. 

- Pierwszym z nich jest scenariusz **stream-based**, w którym dane napływają do systemu w sposób ciągły. W takich przypadkach zachodzi potrzeba okresowej aktualizacji modelu, na przykład w systemach rekomendacyjnych, gdzie zmieniające się trendy wpływają na jakość predykcji. Dodatkowo, na początkowym etapie wdrażania systemu często dysponujemy ograniczoną liczbą oznaczonych danych, a nowe obserwacje mogą znacząco poprawić jego skuteczność. W tym kontekście metody selekcji próbek pozwalają identyfikować najbardziej informatywne dane do dalszego etykietowania.

- Drugim istotny scenariusz to **pool-based**, w którym dostępny jest duży zbiór danych nieetykietowanych, natomiast proces ich ręcznej anotacji jest kosztowny lub czasochłonny. W takim przypadku początkowo oznaczana jest niewielka część danych, na podstawie której trenowany jest model. Następnie, w sposób iteracyjny, model dokonuje predykcji dla nieoznaczonych próbek, a miary niepewności wykorzystywane są do wyboru tych przypadków, które powinny zostać przekazane do eksperta w celu uzyskania etykiety. Pozwala to na stopniowe rozszerzanie zbioru treningowego o najbardziej informatywne przykłady.

W ramach niniejszego projektu przyjęto symulację drugiego ze scenariuszy. W tym celu wykorzystany zostanie w pełni oznaczony zbiór danych, który jednak będzie traktowany jako nieetykietowany na etapie inicjalizacji procesu. Etykiety będą wykorzystywane wyłącznie do symulacji odpowiedzi eksperta oraz aktualizacji modelu w kolejnych iteracjach.

Takie podejście umożliwia przeprowadzenie kontrolowanych eksperymentów oraz ocenę efektywności różnych strategii uczenia aktywnego. W szczególności analizowana będzie liczba zapytań do eksperta wymagana do osiągnięcia określonego poziomu skuteczności modelu, mierzonego za pomocą wybranej metryki.


# UMA Dokumnetacja wstępna
## Active Machine Learning

**Co trzeba zrobić?**
1. opis projektu, wskazujący na zrozumienie problemu
2. ogólny opis algorytmów, które będą wykorzystane
3. szacunkowy plan eksperymentów, który może się zmienić- nie musi być
ostateczny
4. opisane zbiory danych, które będą używane do badań.


**Pytania:**

Ma sens na czymś kosztownym do labelowanie?
Który scenariusz nas interesuje? (przypadki data-shift, underfitting)

**Pomysł na dataset**
w sklepie rozpoznawanie produktów z zdjęcia, jak najmniej labelowania, np owoców,  

- stream-based nowe produkty, których model nie zna - potrzebna etykieta, metody z prawdopodobieństwem
- pool-based dużo zdjec produktow, bez etykiet, metody z wieloma sędziami,

Fruits-360 dataset
A dataset with 180079 images of 257 fruits, vegetables, nuts and seeds

Wykrywanie anaomalii/ fake maili

## SCENARIUSZE

#### Pool-based sampling
Duża pulę nieoznaczonych danych, model wybiera z niej najlepsze przykłady do oznaczenia

#### Stream-based selective sampling
Dane napływają w strumieniu, wybieramy, czy etykietujemy czy nie
**(próg)**

## STRATEGIA WYBORU DO ETYKIETOWANIA
! porównać z random sampling

#### Uncertainty sampling
Model wybiera przykłady, co do których jest najmniej pewny
(np. prawdopodobieństwo ~0.5 w klasyfikacji binarnej)

- least-confidence sampling - liczymy propabilty w sumie dla innych klas niż ta najbardziej prawdopobna, jak model jest nie pewny co do decyzji, wybieramy z najwiekszym

- margin-sampling - wybiera z najmniejszą różnicą pomiędzy dwoma największymi prawdopodobieństwami

- ratio-of-confidence - dzielimy najwieksze prawdopodobienstwo przez drugie najwieksze, wybiera z najmniejszym ratio

- entropy sampling - wybiera próbkę najwieksza entropia prawdopodobienst

**Kiedy dobra metoda ?**
svm, logistic regression, random forest, nn, klasyfikaja, detekcja anomalii, wykrywanie błędów w adnotacji

#### Query-by-Committee (QBC)
Kilka modeli („komitet”) głosuje, a wybierane są dane, gdzie się najbardziej nie zgadzają,

- maximum disagrement
- vote entropy
wybieramy próbki z największą entropią, rozrzucania głosów
- average KL divergence

**Dobra metoda**
Bardziej polecana do scenariusza z dużą ilością danych bez etykiet na początku

#### EMC (Expected Model Change)
Wybierane są próbki, które najbardziej zmieniłyby model, gdyby zostały oznaczone

#### EER (Expected Error Reduction)
Wybierasz dane, które najbardziej zmniejszą przyszły błąd modelu

#### Density-weighted sampling
bierzemy pod uwagę niepewność modelu i „gęstość” danych (czy próbka jest reprezentatywna)
