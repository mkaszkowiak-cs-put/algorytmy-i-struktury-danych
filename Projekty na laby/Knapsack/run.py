from copy import deepcopy  # :)
from timeit import default_timer as timer
from datetime import timedelta
from random import randint


class Item:
	def __init__(self, size, value, identificator=None):
		self.identificator = identificator
		self.size = size
		self.value = value
		self.ratio = self.value / self.size

	def __str__(self):
		txt = f"#{self.identificator} " if self.identificator is not None else ""
		return txt + f"Space {self.size}, value {self.value}"

	def __repr__(self):
		return str(self)

class Knapsack:
	def __init__(self, max_size):
		self.items = []
		self.items_size = 0
		self.max_size = max_size

	def __str__(self):
		txt = f"[*] Knapsack results:\nSpace: {self.items_size} / {self.max_size}\n"
		txt += f"Value: {self.getValue()}\n"
		txt += "\n".join(map(str, self.items)) + "\n"
		return txt

	def addItem(self, item):
		assert item.size > 0
		assert self.canFit(item)
		self.items.append(item)
		self.items_size += item.size
		return self

	def canFit(self, item):
		return self.max_size >= (self.items_size + item.size)

	def getValue(self):
		value = 0
		for i in self.items:
			value += i.value
		return value

class AbstractKnapsackSolver:
	def solve(self, items, size_limit):  # method stub
		raise NotImplementedError


class GreedyKnapsackSolver(AbstractKnapsackSolver):
	def solve(self, items, size_limit):
		# orders by value/size ratio
		knapsack = Knapsack(size_limit)
		items = sorted(items, key=lambda s: s.ratio)[::-1]
		for i in items:
			if knapsack.canFit(i):
				knapsack.addItem(i)

		return knapsack


class BruteforceKnapsackSolver(AbstractKnapsackSolver):
	def solve(self, items, size_limit):
		# storing best_value there for efficiency
		best_knapsack, best_value = Knapsack(size_limit), 0
		# 1 item  = 0 / 1
		# 2 items = 00 / 01 / 10 / 11
		# 2^n combinations, with each item represented by a bit
		# using bit shifting & bitwise and to determine items
		items_size = len(items)
		for combination in range(2**items_size):
			knapsack, value = Knapsack(size_limit), 0
			# go over all items for a combination
			for item_index in range(items_size):
				# check if item belongs to the combination
				if combination & (1 << item_index):
					# debug: print(f"{combination} - {item_index} adding")
					item = items[item_index]

					# if an item fits - add it, if not, 
					# then go straight to the next combination
					# no point checking the remaining items
					if knapsack.canFit(item):
						knapsack.addItem(item)
						value += item.value
					else:
						break
			if value > best_value:
				best_value, best_knapsack = value, knapsack
		
		return best_knapsack


class DynamicProgrammingKnapsackSolver(AbstractKnapsackSolver):
	def solve(self, items, size_limit):
		# initing the matrix - items+1 rows, size+1 columns
		matrix = []
		for _ in range(len(items) + 1):
			row = [0] * (size_limit + 1)
			matrix.append(row)

		# filling the DP matrix
		for row in range(1, len(items) + 1):
			for column in range(1, size_limit + 1):
				item = items[row - 1]  # 0-indexed
				if item.size > column:
					matrix[row][column] = matrix[row - 1][column]
				else:
					matrix[row][column] = max(
						matrix[row - 1][column],
						matrix[row - 1][column - item.size] + item.value
					)
		# optimal value is now stored in matrix[-1][-1]
		# we now retrieve the solution
		knapsack, row, column = Knapsack(size_limit), len(items), size_limit
		while row:
			item_added = matrix[row][column] != matrix[row - 1][column]
			if item_added:
				item = items[row - 1]  # 0-indexed
				knapsack.addItem(item)
				column = column - item.size
			
			row = row - 1

		return knapsack

# source: Algorytm zachłanny III: kontrprzykład
powerpoint_items = [
	Item(size=2, value=5, identificator=1),
	Item(size=1, value=4, identificator=2),
	Item(size=4, value=12, identificator=3),
	Item(size=1, value=2, identificator=4),
	Item(size=3, value=10, identificator=5)
]

# source: Algorytm zachłanny II: przykład
powerpoint_items_2 = [
	Item(size=2, value=4, identificator=1),
	Item(size=1, value=3, identificator=2),
	Item(size=4, value=6, identificator=3),
	Item(size=4, value=8, identificator=4)
]

# I almost forgot to use deepcopy in this implementation!
deepcopy(1)

sizes = []
start_size = 2
for i in range(15):
	sizes.append(start_size)
	start_size += 1

items = []
for size in sizes:
	items_row = []
	for n in range(size):
		items_row.append(Item(size=randint(1, 10), value=randint(1,1000), identificator=n))
	items.append(items_row)


if __name__ == '__main__':
	RUN_SHOWCASE = False

	if RUN_SHOWCASE:
		solvers = []
		for solver in solvers:
			print(str(solver.__name__))
			knapsack = solver().solve(powerpoint_items, 7)
			print(knapsack)

	# this code is going to be ugly, sorry
	solvers = [
		(BruteforceKnapsackSolver, "Algorytm siłowy"), 
		(GreedyKnapsackSolver, "Algorytm zachłanny"), 
		(DynamicProgrammingKnapsackSolver, "Algorytm programowania dynamicznego")
	]

	print("Wykorzystany algorytm,Rozmiar plecaka,Liczba przedmiotów,Czas obliczeń [us]")
	for knapsack_size in range(100, 13*1000, 1000):

		for item_array in items:
			for solver_class, solver_name in solvers:
				solver = solver_class()
				start = timer()
				result = solver.solve(item_array, knapsack_size)
				end = timer()
				time_us = int(timedelta(seconds=end-start).total_seconds()*1000000)
				print(",".join(map(str, [solver_name,knapsack_size,len(item_array),time_us])))
