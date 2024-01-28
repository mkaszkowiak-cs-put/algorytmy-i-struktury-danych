Markdown jest dostosowany pod [Obsidian](https://obsidian.md/), więc w podglądzie Githuba część rzeczy może się psuć

---

# Algorytmy i struktury danych
Maciej Kaszkowiak (maciej@kaszkowiak.org)
czerwiec 2022

## Algorytmy sortowania
![[Pasted image 20220617160402.png]]
### Sortowanie przez wybieranie (Selection Sort)
Algorytm wybiera najmniejszy element z nieposortowanego zbioru i zamienia z obecnie iterowanym elementem po lewej. 
Zaczyna od lewej, sprawdza w prawo.

O(n^2) czasowo, niestabilny, in-place, O(1) pamięciowo
**Niestabilny! Np. 4 2 3 4 1 zamieni się w 1 2 3 4 4!**

```python
def selectionSort(T):
	size = len(T)
	for replacing_index in range(size):
		min_index = replacing_index
		for check_index in range(replacing_index + 1, size):
			if T[check_index] < T[min_index]:
				min_index = check_index

		T[replacing_index], T[min_index] = T[min_index], T[replacing_index]
	return T
```


### Sortowanie przez wstawianie (Insertion Sort)
Algorytm bierze lewy element z nieposortowanego zbioru i próbuje wstawić do posortowanego zbioru po prawej.
Zaczyna od lewej, sprawdza w lewo.

O(n^2) avg i worst-case, O(n) optymistycznie, O(1) pamięciowo, stabilny, in-place.
**O(n) optymistycznie - 1 2 3 4 5**

```python
def insertionSort(T):
	size = len(T)
	
	for element_index in range(1, size):
		element = T[element_index]
		new_position = element_index
		
		for compared in range(element_index - 1, -1, -1):
			if T[compared] > element:
				T[compared + 1] = T[compared]
				new_position -= 1
			else:
				break
		
		T[new_position] = element

	return T
```

### Sortowanie przez zamianę (Bubble Sort)
Bubble Sort porównuje i zamienia dwa sąsiednie elementy. Tablica posortowana tworzy się po prawej stronie. Dzięki zastosowaniu flagi bubble sort może mieć wydajność O(n) dla posortowanej tablicy.

O(n^2) avg i worst-case, O(n) optymistycznie, O(1) pamięciowo, stabilny, in-place.
**O(n) optymistycznie - 1 2 3 4 5**
```python
def bubbleSort(T):
	size, swapped = len(T), False
	
	for iteration in range(size):
		swapped = False
		unsorted_max = size - iteration
		for pair in range(unsorted_max - 1):
			if T[pair] > T[pair + 1]:
				T[pair], T[pair + 1] = T[pair + 1], T[pair]
				swapped = True

		if not swapped:
			break
	return T
# kod nietestowany - dodalem flage swapped w notatkach
```

### Sortowanie szybkie (Quick sort)
O(n log n) avg i best-case, O(n^2) worst-case, O(log n) pamięciowo, in-place, niestabilny.

**O(n^2) worst-case - np. pivot ostatni element i 1 2 3 4 5 6**
**O(log n) pamięciowo ze względu na rekursję**
**niestabilny!**
```python
def quickSortPartition(T, begin, end):
	"""
	W tej implementacji pivot jest ostatnim elementem listy.
	pivot_pos początkowo wskazuje na pierwszy element. 

	Jeśli porównywany element jest mniejszy od pivota,
	  (czyli powinien znaleźć się na lewo od pivota)
	zostaje zamieniony z pivot_pos, a pivot_pos się zwiększa

    Tablica 4 1 2 5 7 1 3 po quicksortpartition zamieni się w:
    <4> 1   2   5   7   1   3 (4 >  3)
    1  <4>  2   5   7   1   3 (1 <  3)
    1   2  <4>  5   7   1   3 (2 <  3)
    1   2  <4>  5   7   1   3 (5 >  3)
    1   2  <4>  5   7   1   3 (7 >  3)
    1   2   1  <5>  7   4   3 (1 <  3)
    1   2   1  <5>  7   4   3 (3 == 3)

	Na koniec pivot zostaje wstawiony na pivot_pos
	1   2   1  <3>  7   4   5

	Wynikiem funkcji jest pozycja pivota - indeks 3.
 
	Aby wybrać inny element do podziału niż ostatni, 
	należy w pierwszym kroku przenieść go na koniec tablicy.
	"""
	pivot_pos = begin
	for item_pos in range(begin, end+1):
		if T[item_pos] < T[end]:
			T[item_pos], T[pivot_pos] = T[pivot_pos], T[item_pos]
			pivot_pos += 1

	T[pivot_pos], T[end] = T[end], T[pivot_pos]
	return pivot_pos

def quickSortInPlace(T, start=0, end=-1): 
	# Sortujemy tablicę co najmniej 1-elementową
	if start > end:
		return T

	# Dzielimy tablicę na części mniejsze i większe od pivota
	pivot_pos = quickSortPartition(T, start, end)

	# Rekursywnie sortujemy lewą i prawą część
	quickSortInPlace(T, start, pivot_pos - 1)
	quickSortInPlace(T, pivot_pos + 1, end)

	# Zwracamy posortowaną tablicę
	return T

quickSortInPlace(T, 0, len(T) - 1)
```

### Sortowanie stogowe (Heap Sort)
O(n log n) złożoność czasowa, O(1) pamięciowa, in-place, niestabilny.

**niestabilne!**
![[heap.png|400]]
```python
"""Kod służy do tworzenia max heapu."""

def heapify(T, size, root):
	"""
	Heapify porównuje element oraz jego dzieci.
	
	Jeśli element jest większy od dzieci 
	  (zgodnie z założeniem max heap)
	to wszystko jest OK.

	Jeśli dziecko jest większe od rodzica,
	to ich wartości są zamieniane
	oraz heapify zostaje wykonane dla dziecka.

	Jedno wywołanie heapify to O(log n) - wysokość kopca.
	"""
	left = 2 * root + 1
	right = 2 * root + 2
	biggest_index = root
	
	if left < size and T[left] > T[biggest_index]:
		biggest_index = left
	
	if right < size and T[right] > T[biggest_index]:
		biggest_index = right

	if biggest_index != root:
		T[biggest_index], T[root] = T[root], T[biggest_index]
		heapify(T, size, biggest_index)

def heapSort(T):
	"""
	Heap Sort działa poprzez:
	1. Utworzenie kopca
		N/2 wywołań * log N heapify
	2. Wstawianie max elementu w posortowaną część
		N wywołań * log N heapify

	Heapify przy tworzeniu kopca wywoływane jest do N/2,
	ponieważ to ostatni możliwy element który może mieć dzieci. 
	"""
	size = len(T)
	for i in range(size // 2, -1, -1):
		heapify(T, size, i)
 
	for i in range(size):
		recently_sorted = size - 1 - i
		T[0], T[recently_sorted] = T[recently_sorted], T[0]

		heapify(T, recently_sorted, 0)

	return T
```

### Sortowanie przez scalanie (Merge Sort)
O(n log n) złożoność czasowa, O(n) pamięciowa, stabilny, not in-place

**O(n) pamięciowa - tworzy dodatkowe tablice!
nie jest in-place - potrzebuje dodatkowego miejsca!**
```python
def mergeSort(T):
	"""
	Merge Sort rekursywnie:
	- dzieli tablice na lewą i prawą część
	- sortuje połówki mergesortem
	- łączy posortowane tablice w wynik

	Warunek stopu to posortowana jedno-elementowa tablica.
	"""
	size = len(T)
	if size <= 1:
		return T

	middle = size // 2
	L = T[0:middle]
	R = T[middle:size]
	L, R = mergeSort(L), mergeSort(R)
	
	return merge(L, R)

def merge(L, R):
	"""
	Merge łączy posortowane tablice poprzez 
	porównywanie lewych elementów z obu tablic
	i dodawanie mniejszego do wyniku.

	Pozostała resztka z jednej tablicy zawsze będzie
	posortowana oraz większa od wyniku,
	więc jest dodawana na koniec.
	"""
	T = []
	while L and R:
		if L[0] > R[0]:
			T.append(R.pop(0))
		else:
			T.append(L.pop(0))

	T.extend(L)
	T.extend(R)
	return T
```

### Sortowanie przez zliczanie (Counting sort)
Złożoność czasowa O(n + k), pamięciowa O(n), stabilny, nie in-place

Pseudokod:
```python
def CountingSort(input, k): 
    count ← array of k + 1 zeros
    output ← array of same length as input
    
    for i = 0 to length(input) - 1 do
        j = key(input[i])
        count[j] += 1

    for i = 1 to k do
        count[i] += count[i - 1]

    for i = length(input) - 1 downto 0 do
        j = key(input[i])
        count[j] -= 1
        output[count[j]] = input[i]

    return output
```


### Sortowanie za pomocą malejących przyrostów (Shell Sort)
Wydajność zależy od wykorzystanego ciągu przerw.

Złożoność optymistyczna O(n log n), pesymistyczna O(n log^2 n), średnia jest nieznana!
Złożoność pamięciowa O(1), w miejscu, ale niestabilny.

Pseudokod:
```cpp
# Sort an array a[0...n-1].
gaps = [701, 301, 132, 57, 23, 10, 4, 1]  // Ciura gap sequence

# Start with the largest gap and work down to a gap of 1
# similar to insertion sort but instead of 1, gap is being used in each step
foreach (gap in gaps)
{
    # Do a gapped insertion sort for every elements in gaps
    # Each loop leaves a[0..gap-1] in gapped order
      for (i = gap; i < n; i += 1)
      {
          # save a[i] in temp and make a hole at position i
          temp = a[i]
          # shift earlier gap-sorted elements up until the correct location for a[i] is found
          for (j = i; a[j - gap] > temp; j -= gap)
          {
              a[j] = a[j - gap]
          }
          # put temp (the original a[i]) in its correct location
          a[j] = temp
       }
}
```



## Drzewa AVL, BST
### Przypomnienie terminów
#### Wysokość
Wysokość drzewa jest liczona bez korzenia (sam korzeń ma h=0).
Dla drzewa zdegenerowanego wysokość h = n-1
Dla drzewa wyważonego h = floor(log2 n)

#### Drzewo wyważone (zrównoważone, AVL)
Drzewo binarne T, w którym dla każdego węzła wi w T bezwzględna wartość różnicy między wysokością h jego lewego i prawego poddrzewa wynosi co najwyżej 1: 

**|hL(wi) - hP(wi) | ≤ 1**

Drzewko AVL przechowuje współczynnik równowagi który jest równy różnicy wysokości lewego i prawego poddrzewa.

#### Drzewo dokładnie wyważone (DDW)
Drzewo binarne T, w którym dla każdego węzła wi w T bezwzględna wartość różnicy między liczbą N elementów w jego lewym i prawym poddrzewie wynosi co najwyżej 1: 

**|NL(wi) - NP(wi) | ≤ 1**

### Metody przechodzenia
#### pre-order (wzdłużna)
**Korzeń**, Lewe drzewo, Prawe drzewo

#### in-order (poprzeczna)
Lewe drzewo, **Korzeń**, Prawe drzewo
np. sortowanie drzewa

#### post-order (wsteczna)
Lewe drzewo, Prawe drzewo, **Korzeń**
np. usuwanie drzewa

### Drzewa BST
Budowanie poprzez wyszukiwanie połówkowe:
- posortowanie ciągu wartości (N log N)
- dodanie środkowego elementu ciągu jako korzenia (mediana)
- lewe poddrzewo to mediana lewej części, prawe to mediana prawej

#### Usuwanie elementu z drzewa
**Liść** - robi po prostu YEEET
**Jeden potomek** - zamieniamy węzeł z potomkiem
**Dwa potomki** - wstawiamy wybrany sąsiadujący (in-order) węzeł oraz powtarzamy procedurę usuwania dla podkradzionego węzła

#### Równoważenie drzewa metodą z usuwaniem korzeni niezbalansowanych poddrzew
- przechodzimy BFS (poziomami), szukając aż znajdziemy węzeł ze współczynnikiem równowagi > 1
- usuwamy węzeł, jeśli ma dwa poddrzewa to zastępujemy poddrzewem o większej wysokości
- wstawiamy ponownie węzeł
- wracamy do punktu 1

Algorytm zakończy się gdy nie będzie węzłów z bf > 1 - drzewo będzie zrównoważone

#### Rotacja
Rotacja zachowuje kolejność elementów in-order. Jeden węzeł idzie do góry, natomiast drugi węzeł idzie na dół.

Lewy element lewego elementu pozostaje nietknięty.
Prawy element prawego elementu pozostaje nietknięty.

Przy rotacji w prawo Q zostaje prawym dzieckiem P, więc 
Q musi przejąć B jako swoje lewe dziecko.

Przy rotacji w lewo P zostaje lewym dzieckiem Q, więc P musi przyjąć B jako swoje prawe dziecko.

![[Pasted image 20220617172316.png|500]]
![[Pasted image 20220617172409.png|300]]

```python
def leftRotate(self, X):
	Y = X.right
	Temp = Y.left

	Y.left = X
	X.right = Temp

	# pominięto rekalkulację wysokości

	return Y

def rightRotate(self, X):
	Y = X.left
	Temp = Y.right

	Y.right = X
	X.left = Temp

	# pominięto rekalkulację wysokości
	return Y
```

#### Tworzenie drzewa AVL
TODO
```python
"""funkcja wywoływana w następujący sposób:"""
tree.root = tree.insertNode(tree.root, Node(value))

def insertNode(self, searchNode, node):
	if searchNode is None:
		return node
	elif searchNode > node:
		searchNode.left = self.insertNode(searchNode.left, node, balance)
	else:
		searchNode.right = self.insertNode(searchNode.right, node, balance)


	"""Po dodaniu: rekalkulacja wysokości"""
	searchNode.height = 1 + max(
		self.getHeight(searchNode.left),
		self.getHeight(searchNode.right)
	)

	"""balanceFactor przyjmowany jest jako left - right"""
	balanceFactor = self.getBf(searchNode)
	
	"""
	Rotacja w lewo zmniejsza wysokość prawego drzewa
	Rotacja w prawo zmniejsza wysokość lewego drzewa
	
	Tym samym:
	  BF > 1 => więcej itemów po lewej => rotacja w prawo
	  BF < -1 => więcej itemów po prawej => rotacja w lewo

	Ponadto:
	  - gdy dodano lewo > lewo: wszystko ok
	  - gdy dodano prawo > prawo: wszystko ok
	
	Ale przed tym:
	  - gdy dodano lewo > prawo: 
	     zmniejszamy wysokość prawego poddrzewa, rotacja w lewo
	  - gdy dodano prawo > lewo:
	     zmniejszamy wysokość lewego poddrzewa, rotacja w prawo 
	"""
	if balanceFactor > 1 and node < searchNode.left:
		searchNode = self.rightRotate(searchNode)

	elif balanceFactor > 1 and node > searchNode.left:
		searchNode.left = self.leftRotate(searchNode.left)
		searchNode = self.rightRotate(searchNode)

	elif balanceFactor < -1 and node < searchNode.right:
		searchNode.right = self.rightRotate(searchNode.right)
		searchNode = self.leftRotate(searchNode)

	elif balanceFactor < -1 and node > searchNode.right:
		searchNode = self.leftRotate(searchNode)

	return searchNode
```


### Struktura listy jedno- i dwukierunkowej
Zwykłe linked list. Każdy element zawiera wartość oraz wskaźnik na następny element. Lista dwukierunkowa zawiera również wskaźnik na poprzedni element. 


#### Równoważenie drzewa metodą DSW (Day-Stout-Warren)
![[Pasted image 20220618192822.png]]

Algorytm:
- dodaje tymczasowego node i przypisuje drzewo po prawej stronie
- zamienia rotacjami w prawo drzewo w winorośl
- zamienia rotacjami w lewo winorośl w zbalansowane drzewo
- usuwa tymczasowego node

```python
def treeToVine(self, root):
	tail = root
	rest = tail.right
	while rest:
		if rest.left is None:
			tail = rest
			rest = rest.right
		else:
			temp = rest.left
			rest.left = temp.right
			temp.right = rest
			rest = temp
			tail.right = temp

def vineToTree(self, root, size):
	leaves = size + 1 - 2 ** (math.log2(size + 1))
	self.compress(root, leaves)
	size = size - leaves
	while size > 1:
		self.compress(root, size / 2)
		size = size / 2

def compress(self, root, count):
	scanner = root
	for i in range(int(count)):
		child = scanner.right
		scanner.right = child.right
		scanner = scanner.right
		child.right = scanner.left
		scanner.left = child

def dsw(self):
	pseudoroot = Node(0)
	pseudoroot.right = self.root
	count_elements = len(self.root.inOrder())
	self.treeToVine(pseudoroot)
	self.vineToTree(pseudoroot, count_elements)
	self.root = pseudoroot.right
	del pseudoroot
```



## Algorytmy grafowe
### Wybrane pojęcia z teorii grafów
**graf prosty** - graf który nie zawiera krawędzi wielokrotnych ani pętli
**multigraf** - przeciwieństwo grafu prostego
**rząd grafu** - liczba N wierzchołków
**rozmiar grafu** - liczba M krawędzi
**digraf** - graf skierowany
**ścieżka Hamiltona** - ściezka przechodząca przez wszystkie wierzchołki grafu
**ścieżka Eulera** - ścieżka przechodząca przez wszystkie krawędzie grafu

**graf półhamiltonowski** - zawiera ścieżkę Hamiltona
**graf hamiltonowski** - zawiera cykl Hamiltona

### Reprezentacje grafu
![[Pasted image 20220618223320.png]]

#### Macierz sąsiedztwa
Rozmiar N^2, M(x)(y) = 1 jeśli istnieje krawędź, -1 w skierowanym jeśli wchodzi do niego łuk

#### Macierz incydencji
Rozmiar N \* M, wiersze = wierzchołki, kolumny = krawędzie
-1 jeśli łuk wychodzi z wierzchołka, 1 jeśli wchodzi do wierzchołka, 2 jeśli pętla własna

#### Lista krawędzi
Rozmiar M

#### Lista incydencji / Lista sąsiedztwa
Rozmiar N + M
Każdy wierzchołek zawiera listę sąsiadów
**tylko graf nieskierowany** - skierowany ma inną nazwę

#### Lista poprzedników / Lista następników
Rozmiar N + M
**tylko graf skierowany**

#### Macierz grafu XD
Rozmiar N \* (N + 4)

wartość $M[i][j]:$
$<-V, -1>$: nieincydentne
$<0, V>$:  jest taka krawędź
$<V+1, 2V>:$ jest odwrotna krawędź
$<2V+1, 3V>:$ są obie krawędzie - taka i odwrotna

Ostatnie 4 kolumny:
1. Pierwszy następnik (0 jeśli brak)
2. Pierwszy poprzednik (0 jeśli brak)
3. Pierwszy nieincydentny 
4. Pierwszy z cyklu


![[Pasted image 20220618233011.png]]

### Przechodzenie grafu

#### DFS, Depth First Search - w głąb
pre-order dla drzewa

#### BFS, Breadth First Search - wszerz
level-order dla drzewa

### Sortowanie topologiczne
Sortowanie topologiczne grafu skierowanego G=(V, E) polega na liniowym uporządkowaniu wierzchołków tego grafu w taki sposób, że dla każdej pary wierzchołków połączonych łukiem z u do v, w porządku topologicznym u występuje przed v. 
Sortowanie topologiczne grafu G jest możliwe tylko wówczas, gdy G jest grafem skierowanym acyklicznym (w skrócie jest to DAG – Directed Acyclic Graph).

#### Algorytm Kahna
```python
def topologicalSort(self):
	# Kahn's algorithm
	# Let's start with deepcopying the original matrix
	matrix = copy.copy(self.matrix)

	# Count incoming edges for each node
	degrees = [0] * self.size
	for original, edges in enumerate(matrix):
		for element, edge in enumerate(edges):
			if edge == -1:
				degrees[original] += 1

	# Check which nodes have no incoming edges
	queue = deque()
	for element, degree in enumerate(degrees):
		if degree == 0:
			queue.append(element)

	
	# In order to prove that there was no cycle in the graph,
	#   asserting visited nodes count == nodes count is enough 
	visited_count = 0

	topological_order = []

	while queue:
		element = queue.popleft()
		topological_order.append(element)

		# Remove all edges to neighbors for the current element
		# If a neighbor has no remaining edges, add it to the queue
		for neighbour, edge in enumerate(matrix[element]):
			if edge != 1:
				continue
			degrees[neighbour] -= 1
			if degrees[neighbour] == 0:
				queue.append(neighbour)

		visited_count += 1

	assert visited_count == self.size, "Graf zawiera cykl."

	return topological_order
```

#### Sortowanie topologiczne z wykorzystaniem DFS
TODO: wykrywanie cyklu
```python
def topologicalDFS(self):
	results, visited = [], [False] * self.size

	for node in range(self.size):
		if not visited[node]:
			self._topologicalDFS(node, visited, results)
	
	return results[::-1]

def _topologicalDFS(self, node, visited, results):
	visited[node] = True

	for neighbour, edge in enumerate(self.matrix[node]):
		if edge != 1 or visited[neighbour]:
			continue
		self._topologicalDFS(neighbour, visited, results)

	results.append(node)
```


### Ścieżka Hamiltona
Problem przeszukiwania: silnie NP-trudny
Problem decyzyjny: silnie NP-zupełny

#### Algorytm Robertsa-Floresa
```python
def hcycle(self, start_node):
	path, path_size = [start_node], 0
	visited = [False] * self.size

	def hamiltonian(node):
		nonlocal path, path_size, visited
		# assume suitable node will be found
		visited[node] = True
		path_size += 1
		
		for points_to in self.matrix[node]:
			# found full hcycle
			if points_to == start_node and path_size == self.size:
				return True
			
			# found suitable candidate for the path
			if not visited[points_to]:
				if hamiltonian(points_to):
					path.append(points_to)
					return True

		# no suitable nodes, backtracking
		visited[node] = False
		path_size -= 1
		return False

	if hamiltonian(start_node):
		return path[::-1]  # convert stack to list
	else:
		return False
```

### Ścieżka Eulera
Problem przeszukiwania: klasa P
Problem decyzyjny: klasa P

#### Warunek dostateczny i konieczny na istnienie cyklu Eulera 
1) Graf jest spójny (poza izolowanymi wierzchołkami)
2) (graf nieskierowany) 
    - stopień każdego wierzchołka jest parzysty
3) (graf skierowany) 
   - dla każdego wierzchołka stopień wejściowy jest równy stopniowi wyjściowemu

##### ... na istnienie ścieżki Eulera
1) Graf jest spójny (poza izolowanymi wierzchołkami)
2) (graf nieskierowany) 
    - stopień każdego wierzchołka **z wyjątkiem dwóch** jest parzysty
3) (graf skierowany) 
   - dla każdego wierzchołka **z wyjątkiem dwóch** stopień wejściowy jest równy stopniowi wyjściowemu

**Dwa wierzchołki, które nie muszą spełniać warunku to wierzchołek startowy i wierzchołek końcowy ścieżki Eulera.**

#### Algorytm Hierholzera
```python
"""
Zwróci cykl Eulera jeśli rozmiar wynikowej tablicy będzie OK

Algorytm po kolei dla każdego wierzchołka:
- usuwa krawędzie z jego sąsiadami
- wykonuje rekursywnie procedurę usuwania krawędzi
- wrzuca wierzchołek na stos 
""" 
def ecycle(self):
	# using deepcopy due to removing edges
	graph = copy.deepcopy(self) 

	results = []
	def euler(node):
		nonlocal graph, results
		# iterate over succesors
		for neighbor in graph.matrix[node]:
			graph.removeEdge(node, neighbor)
			euler(neighbor)
		results.append(node)

	euler(0)

	return results[::-1]
```

#### Algorytm Fleury’ego 
Algorytm Fleury'ego przechodzi kolejne krawędzie grafu i na każdym kroku sprawdza, czy do kolejnego następnika prowadzi krawędź która jest mostem. Krawędź-most jest wybierana tylko jeśli nie ma innej krawędzi. 

Most jest to krawędź w grafie, która po usunięciu zwiększa liczbę spójnych składowych grafu.

### Klasy złożoności
![[Pasted image 20220619024334.png|250]]
#### P (deterministic polynomial)
O(N^x)
Deterministyczna maszyna Turinga rozwiązuje problemy w czasie wielomianowym względem rozmiaru danych wejściowych.

#### NP (nondeterministic polynomial)
O(N^x), O(x^N)

Niedeterministyczna maszyna Turinga rozwiązuje problemy w czasie wielomianowym względem rozmiaru danych wejściowych.

Poprawność rozwiązania problemu z klasy NP może być sprawdzona w czasie wielomianowym.

#### NP-zupełne (NP-complete)
O(x^N), O(x \* N)

Problemy z klasy NP, do których w czasie wielomianowym da się przetransferować dowolny inny problem z klasy NP.

Problem NP-zupełny to problem, dla którego nie znaleziono algorytmu, który rozwiąże dowolną instancję tego problemu w czasie wielomianowym.

Przykład: problem plecakowy, problem sumy podzbioru, problem podziału zbioru liczb.

#### Silnie NP-zupełne (strongly NP-complete)
O(x^n)

Problemy nieliczbowe NP-zupełne lub takie liczbowe problemy NP-zupełne, które nawet przy ograniczeniu maksymalnej wartości występujących w nim liczb pozostają NP-zupełne. 

Przykład: problem cyklu Hamiltona, problem trójpodziału, problem komiwojażera

#### Przedstawianie, że problem jest NP-zupełny
Można to pokazać przeprowadzając dowód NP-zupełności poprzez transformację wielomianową. Co jest potrzebne do dowodu? 
- Nasz nowy problem A (dla którego chcemy dowieść, że jest NP-zupełny) 
- Znany problem B, o którym wiadomo, że jest NP-zupełny 

Dowód składa się z dwóch kroków: 
Krok 1. Pokazujemy, że problem A należy do klasy NP. 
Krok 2. Wykonujemy transformację wielomianową znanego problemu B do naszego nowego problemu A. Zapisujemy to tak: B $\alpha$ A. 

Transformacja wielomianowa jest to funkcja f: B→A, która spełnia warunki: 
a) dla każdej instancji problemu B odpowiedzią (rozwiązaniem) jest TAK wtedy i tylko wtedy gdy dla każdej instancji problemu A odpowiedzią jest TAK; 
b) czas obliczania funkcji f przez DTM dla każdej instancji problemu B jest ograniczony od góry przez wielomian N(B). 


## Problem plecakowy
Problem optymalizacyjny: Znajdź podzbiór elementów, które zmieszczą się do plecaka a suma ich wartości będzie maksymalna
NP-trudny

Problem decyzyjny: Czy istnieje taki podzbiór elementów w plecaku, aby suma ich wartości była równa co najmniej b?
NP-zupełny


### Rodzaje problemu plecakowego
#### Dyskretny 
(liczba dodawanych elementów jest całkowita)

##### Binarny
każdy element występuje tylko raz
##### Ograniczony
zdefiniowana jest liczba elementów każdego typu
##### Nieograniczony
brak ograniczenia na liczbę elementów każdego typu
##### Wielowymiarowy
plecak oraz elementy mają co najmniej 2 wymiary

#### Ciągły 
(nie ma wymogu, aby liczba dodawanych elementów była całkowita)

### Algorytm zachłanny
Algorytm zachłanny (ang. greedy algorithm) to ogólna technika algorytmiczna, która polega na rozwiązywaniu problemu optymalizacyjnego w następujący sposób: 
1. Algorytm w każdym kroku dokonuje wyboru na podstawie oceny sąsiedztwa. 
2. Wybiera przejście do lokalnego optimum (zawsze przechodzi do najlepszego rozwiązania sąsiedniego = decyzja zachłanna). 
3. Jeśli w danym kroku nie można polepszyć rozwiązania, algorytm zatrzymuje się i zwraca rozwiązanie bieżące jako rozwiązanie problemu.

#### Wskazówki projektowania algorytmu zachłannego
- określ stan początkowy
- określ warunki dopuszczalnego rozwiązania
- ustal warunek stopu
- zdefiniuj funkcję celu (jak obliczyć wartość dla rozwiązania)
- ustal funkcję wyboru (jak wybrać następny element)

### Programowanie dynamiczne
#### Na czym polega metoda PD?
1) rozwiąż problem dla jednego elementu przy różnych wartościach parametru sterującego i zapamiętaj wyniki
2) dodaj następny element do problemu
3) zbuduj rozwiązanie problemu powiększonego o nowy składnik dokonując optymalnego wyboru w oparciu o poprzednio zapisane rozwiązania

#### Kiedy możemy zastosować PD? 

PD opiera się na podziale rozwiązywanego problemu na podproblemy względem kilku parametrów (np. liczba elementów w plecaku lub pojemność plecaka). W odróżnieniu od techniki ‚dziel i zwyciężaj’, w PD podproblemy nie są rozłączne i cechuje je własność optymalnej struktury.

Dany problem ma własność optymalnej struktury jeśli jego optymalne rozwiązanie jest funkcją optymalnych rozwiązań podproblemów. Optymalne rozwiązanie problemu dla k-tego etapu jest jednocześnie rozwiązaniem optymalnych dla wszystkich etapów po nim następujących: k + 1, k + 2 ,…, n. Zatem optymalne rozwiązanie pierwszego etapu stanowi podstawę do uzyskania optymalnego rozwiązania na każdym kolejnym etapie dla całego problemu

#### Co jest potrzebne aby zaprojektować algorytm techniką PD? 
Należy sformułować równanie rekurencyjne Bellmana dla problemu (tj. równanie opisujące optymalną wartość funkcji celu dla problemu, które wykorzysta wyznaczone optymalne rozwiązania podproblemów o mniejszych rozmiarach). Poszczególne rozwiązania dla mniejszych podproblemów zapisywane są w tablicy.

![[Pasted image 20220619031515.png]]
![[Pasted image 20220619031613.png]]

TODO: odczytywanie wartości
