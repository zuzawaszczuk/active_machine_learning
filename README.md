# active_machine_learning

### Opis Projektu

Tematem projektu jest analiza zastosowania metod uczenia aktywnego w wybranym problemie uczenia maszynowego.

Głównym celem uczenia aktywnego jest ograniczenie kosztów związanych z procesem etykietowania danych. Problem ten jest szczególnie istotny w dwóch scenariuszach opisywanych w literaturze. 

- Pierwszym z nich jest scenariusz **stream-based**, w którym dane napływają do systemu w sposób ciągły. W takich przypadkach zachodzi potrzeba okresowej aktualizacji modelu, na przykład w systemach rekomendacyjnych, gdzie zmieniające się trendy wpływają na jakość predykcji. Dodatkowo, na początkowym etapie wdrażania systemu często dysponujemy ograniczoną liczbą oznaczonych danych, a nowe obserwacje mogą znacząco poprawić jego skuteczność. W tym kontekście metody selekcji próbek pozwalają identyfikować najbardziej informatywne dane do dalszego etykietowania.

- Drugim istotny scenariusz to **pool-based**, w którym dostępny jest duży zbiór danych nieetykietowanych, natomiast proces ich ręcznej anotacji jest kosztowny lub czasochłonny. W takim przypadku początkowo oznaczana jest niewielka część danych, na podstawie której trenowany jest model. Następnie, w sposób iteracyjny, model dokonuje predykcji dla nieoznaczonych próbek, a miary niepewności wykorzystywane są do wyboru tych przypadków, które powinny zostać przekazane do eksperta w celu uzyskania etykiety. Pozwala to na stopniowe rozszerzanie zbioru treningowego o najbardziej informatywne przykłady.

W ramach niniejszego projektu przyjęto symulację drugiego ze scenariuszy. W tym celu wykorzystany zostanie w pełni oznaczony zbiór danych, który jednak będzie traktowany jako nieetykietowany na etapie inicjalizacji procesu. Etykiety będą wykorzystywane wyłącznie do symulacji odpowiedzi eksperta oraz aktualizacji modelu w kolejnych iteracjach.

Takie podejście umożliwia przeprowadzenie kontrolowanych eksperymentów oraz ocenę efektywności różnych strategii uczenia aktywnego. W szczególności analizowana będzie liczba zapytań do eksperta wymagana do osiągnięcia określonego poziomu skuteczności modelu, mierzonego za pomocą wybranej metryki.
