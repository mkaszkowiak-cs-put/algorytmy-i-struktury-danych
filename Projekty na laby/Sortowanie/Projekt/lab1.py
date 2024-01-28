import copy
import sys
import resource
import math
import random
from timeit import default_timer as timer
from datetime import timedelta

comparison, swap = 0, 0

class SortingAlgorithms:
	@staticmethod
	def bubbleSort(T):
		global comparison, swap
		size = len(T)
		for iteration in range(size):
			unsorted_max = size - iteration
			for pair in range(unsorted_max - 1):
				comparison += 1
				if T[pair] > T[pair + 1]:
					T[pair], T[pair + 1] = T[pair + 1], T[pair]
					swap += 1
		return T

	@staticmethod
	def selectionSort(T):
		global comparison, swap
		size = len(T)
		for replacing_index in range(size):
			min_index = replacing_index
			for check_index in range(replacing_index + 1, size):
				comparison += 1
				if T[check_index] < T[min_index]:
					min_index = check_index

			swap += 1
			T[replacing_index], T[min_index] = T[min_index], T[replacing_index]
		return T

	@staticmethod
	def mergeSort(T):
		global comparison, swap
		size = len(T)
		if size <= 1:
			return T

		middle = size // 2
		L = T[0:middle]
		R = T[middle:size]
		L, R = SortingAlgorithms.mergeSort(L), SortingAlgorithms.mergeSort(R)
		return SortingAlgorithms.__merge(L, R)

	@staticmethod
	def __merge(L, R):
		global comparison, swap
		T = []
		while L and R:
			comparison += 1
			if L[0] > R[0]:
				T.append(R.pop(0))
			else:
				T.append(L.pop(0))

		T.extend(L)
		T.extend(R)
		return T

	@staticmethod
	def quickSortPartition(T, begin, end):
		global comparison, swap
		pivot_pos = begin
		for item_pos in range(begin, end+1):
			comparison += 1
			if T[item_pos] < T[end]:
				swap += 1
				T[item_pos], T[pivot_pos] = T[pivot_pos], T[item_pos]
				pivot_pos += 1

		swap += 1
		T[pivot_pos], T[end] = T[end], T[pivot_pos]
		return pivot_pos

	@staticmethod
	def quickSortInPlace(T, start=0, end=None):
		global comparison, swap
		if end is None:
			end = len(T) - 1

		# should I include element comparisons only?
		# I'll assume so and won't increment comparison variable below 
		if start > end:
			return T

		pivot_pos = SortingAlgorithms.quickSortPartition(T, start, end)
		SortingAlgorithms.quickSortInPlace(T, start, pivot_pos - 1)
		SortingAlgorithms.quickSortInPlace(T, pivot_pos + 1, end)
		return T

	@staticmethod
	def quickSortSimple(T):
		global comparison, swap
		if len(T) <= 1:
			return T

		comparison += len(T) - 1
		pivot = T[-1]
		smaller = [el for el in T[:-1] if el <= pivot]
		bigger = [el for el in T[:-1] if el > pivot]
		return SortingAlgorithms.quickSortSimple(smaller) + [pivot] + SortingAlgorithms.quickSortSimple(bigger)


	@staticmethod
	def __heapify(T, size, root):
		global comparison, swap
		# passing size down to heapify() to:
		# - reduce unnecessary calls to len();
		# - allow heapify() to limit itself to a chunk of an array

		left = 2 * root + 1
		right = 2 * root + 2
		biggest_index = root
		
		comparison += 3
		# check if child nodes are bigger than root
		if left < size and T[left] > T[biggest_index]:
			biggest_index = left
		
		if right < size and T[right] > T[biggest_index]:
			biggest_index = right

		# if a child node is bigger than root, swap them and continue heapifying
		if biggest_index != root:
			swap += 1
			T[biggest_index], T[root] = T[root], T[biggest_index]
			SortingAlgorithms.__heapify(T, size, biggest_index)

	@staticmethod
	def heapSort(T):
		global comparison, swap
		# in order to create a heap, we need to heapify elements from size//2 to 0
		# why n//2? because it's the last possible node that could have children 
		size = len(T)
		for i in range(size // 2, -1, -1):
			SortingAlgorithms.__heapify(T, size, i)

		# now that the heap should have the biggest element extracted
		#  we can retrieve the biggest element -> move it to the end -> heapify -> ... 
		for i in range(size):
			recently_sorted = size - 1 - i
			swap += 1
			T[0], T[recently_sorted] = T[recently_sorted], T[0]

			SortingAlgorithms.__heapify(T, recently_sorted, 0)

		return T

	@staticmethod
	def insertionSort(T):
		global comparison, swap
		size = len(T)
		
		for element_index in range(1, size):
			element = T[element_index]
			new_position = element_index
			
			for compared in range(element_index - 1, -1, -1):
				comparison += 1
				if T[compared] > element:
					swap += 1
					T[compared + 1] = T[compared]
					new_position -= 1
				else:
					break
			
			swap += 1
			T[new_position] = element

		return T

class Timer:
	@staticmethod
	def timeSorts(functions, unsorted_array):
		sorted_array = sorted(unsorted_array)
		global comparison, swap
		for name, fun in functions.items():
			comparison, swap = 0, 0
			array_instance = copy.deepcopy(unsorted_array)

			error = None
			start = timer()
			try:
				results = fun(array_instance)
				assert results == sorted_array, "Sorting results arent correct!"
			except Exception as e:
				error = str(e)
			end = timer()

			if error is None:
				print(
					timedelta(seconds=end-start), int(timedelta(seconds=end-start).total_seconds()*1000000), "us", name, " | ", 
					comparison, "comparisons, ", swap, "swaps")
			else:
				print("[!]", error, name)

	def measureSort(fun, unsorted_array):
		global comparison, swap
		comparison, swap = 0, 0
		array_instance = copy.deepcopy(unsorted_array)

		error = None
		start = timer()
		try:
			results = fun(array_instance)
		except Exception as e:
			return None
		end = timer()

		return [int(timedelta(seconds=end-start).total_seconds()*1000000), comparison, swap]

class ArrayGenerator:
	""" Liczby w każdym ciągu należą do przedziału <1,10×n>. """
	@staticmethod
	def sortedAscArray(size):
		T = []
		for i in range(size):
			T.append((i+1) * size)

		return T

	@staticmethod
	def sortedDescArray(size):
		# shift by 1
		T = []
		for el in ArrayGenerator.sortedAscArray(size)[::-1]:
			T.append(el - 1)
		return T

	@staticmethod
	def randomArray(size):
		T = []
		for _ in range(size):
			T.append(random.randrange(-size, size))
		return T

	@staticmethod
	def shapedV(size):
		T = []
		T.extend(ArrayGenerator.sortedAscArray(size // 2))
		T.extend(ArrayGenerator.sortedDescArray((size // 2) + (size % 2)))
		return T

	@staticmethod
	def shapedA(size):
		T = []
		T.extend(ArrayGenerator.sortedDescArray(size // 2))
		T.extend(ArrayGenerator.sortedAscArray((size // 2) + (size % 2)))
		return T

	@staticmethod
	def predefinedArray(size):
		global preset_array
		return preset_array


if __name__ == "__main__":
	resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
	sys.setrecursionlimit(10**6)

	generators = {
		#"Predefined array": ArrayGenerator.predefinedArray,
		"Tablica rosnąca": ArrayGenerator.sortedAscArray,
		"Tablica malejąca": ArrayGenerator.sortedDescArray,
		"Losowe wartości": ArrayGenerator.randomArray,
		"V-kształtna": ArrayGenerator.shapedV,
		"A-kształtna": ArrayGenerator.shapedA
	}

	functions = {
		"Insertion sort": SortingAlgorithms.insertionSort,
		"Heap sort": SortingAlgorithms.heapSort,
		"Bubble sort": SortingAlgorithms.bubbleSort,
		# "Quick sort (in place)": SortingAlgorithms.quickSortInPlace,
		"Quick sort": SortingAlgorithms.quickSortSimple,
		"Merge sort": SortingAlgorithms.mergeSort,
		"Selection sort": SortingAlgorithms.selectionSort
	}

	"""
	presets = [
		[10,9,8,7,6,5,4,3,2,1],
		[2,4,8,10,12,10,8,4,2,1],
		[1,2,3,4,5,6,7,8,9,10]
	]

	print("Predefiniowane tablice:")
	for n, p in enumerate(presets):
		print(f"\n\nTablica #{n}\n")
		preset_array = p
		Timer.timeSorts(functions, ArrayGenerator.predefinedArray(10))
	"""
	n = []
	# todo: range(15)
	for i in range(15):
		n.append(round(2 ** (i*4/5+2)))

	#print(n)
	#sys.exit(1)

	print("Zadanie 1,,,,,,")
	for sort_name, sort in functions.items():
		print(f"{sort_name},,,,,,")
		print("N, " + ", ".join(generators.keys()))
		for count in n:
			val = []
			for gen_name, gen in generators.items():
				results = Timer.measureSort(sort, gen(count))
				val.append(str(results[0]))
			print(str(count)+', '+", ".join(val))
		print(",,,,,,")

	print("Zadanie 2,,,,,,")
	for gen_name, gen in generators.items():
		print(f"{gen_name},,,,,,")
		print("N, " + ", ".join(functions.keys()))
		for count in n:
			val = []
			for sort_name, sort in functions.items():
				results = Timer.measureSort(sort, gen(count))
				val.append(str(results[0]))
			print(str(count)+', '+", ".join(val))
		print(",,,,,,")


	print("Zadanie 3,,,,,,")
	for sort_name, sort in functions.items():
		print(f"{sort_name},,,,,,")
		print("N, " + ", ".join(generators.keys()))
		for count in n:
			val = []
			for gen_name, gen in generators.items():
				results = Timer.measureSort(sort, gen(count))
				val.append(str(results[1] + results[2]))
			print(str(count)+', '+", ".join(val))
		print(",,,,,,")
	"""
	size = 15000
	print(f"\n---\nElements count: {size}\n---")
	for name, generator in generators.items():
		print(f"\n---\n{name}\n---")
		Timer.timeSorts(functions, generator(size))
	"""