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
