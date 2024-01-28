import copy
import math
import sys
from timeit import default_timer as timer
from datetime import timedelta
from collections import deque


class AdjacencyMatrix:
	def __init__(self, size, directed=True):
		self.matrix = []
		self.size = None
		self.directed = directed
		self.initMatrix(size)

	def initMatrix(self, size):
		self.size = size
		# I cannot use list comprehension due to deepcopy issues
		for _ in range(size):
			row = []
			for _ in range(size):
				row.append(0)
			self.matrix.append(row)
		return self

	def addEdge(self, start, end):
		assert 0 <= start < self.size and 0 <= end < self.size
		# Order is important so A->A edges are 1
		self.matrix[end][start] = -1 if self.directed else 1 
		self.matrix[start][end] = 1
		return self

	def removeEdge(self, start, end):
		assert 0 <= start < self.size and 0 <= end < self.size 
		self.matrix[end][start] = 0
		self.matrix[start][end] = 0
		return self

	def hasEdge(self, start, end):
		assert 0 <= start < self.size and 0 <= end < self.size 
		return self.matrix[start][end]

	def DFS(self, node, visited={}, ancestors={}):
		if node in visited:
			return visited
		visited[node] = 1
		ancestors[node] = 1
		for element, edge in enumerate(self.matrix[node]):
			if edge != 1:
				continue
			assert element not in ancestors, "Graf zawiera cykl."
			visited = self.DFS(element, visited, copy.copy(ancestors))
		return list(visited)

	def __str__(self):
		res = []
		def showDigit(digit):
			if digit == 0:
				return "---"
			if digit == 1:
				return "OUT"
			return "IN"
		for i in self.matrix:
			res.append('\t'.join(map(showDigit, i)))
		return "\n".join(res)

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

			for neighbour, edge in enumerate(matrix[element]):
				if edge != 1:
					continue
				degrees[neighbour] -= 1
				if degrees[neighbour] == 0:
					queue.append(neighbour)

			visited_count += 1

		assert visited_count == self.size, "Graf zawiera cykl."

		return topological_order

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


class GraphMatrix:
	""" Top secret matrix implementation recently declassified by CIA"""
	def __init__(self, adjacency_matrix):
		self.matrix = []
		self.size = None
		self.initMatrix(adjacency_matrix)

	def __str__(self):
		res = []
		def showDigit(digit):
			if digit is None or digit < 0:
				return "---"
			if digit < self.size:
				return "OUT"
			return "IN"

		for i in self.matrix:
			res.append('\t'.join(map(str, i)))
		trademark = ""
		return trademark + "\n" + "\n".join(res)

	def initMatrix(self, adjacency_matrix):
		N = adjacency_matrix.size
		MX = adjacency_matrix
		# I'll introduce a variable N denoting the the nodes count
		# and MX as adjacency matrix due to their vast usage 
		
		# Generate an empty GraphMatrix
		self.size = N
		self.matrix = []
		for _ in range(N):
			row = []
			for _ in range(N + 3):
				# I'm not filling it with 0's
				# otherwise it screws up my 0-indexed elements
				row.append(None)
			self.matrix.append(row)

		incomingEdges, outcomingEdges, unconnected = [], [], []
		for _ in range(N):
			incomingEdges.append([])
			outcomingEdges.append([])
			unconnected.append([])

		for element, edges in enumerate(MX.matrix):
			incomingEdges.append(0)
			for neighbour, edge in enumerate(edges):
				if edge == 1:
					outcomingEdges[element].append(neighbour)
				elif edge == -1:
					incomingEdges[element].append(neighbour)
				elif edge == 0:
					# prevent duplication on A->A
					if neighbour not in unconnected:
						unconnected[element].append(neighbour)


		for element, points_to in enumerate(outcomingEdges):
			if not points_to:
				continue
			self.matrix[element][N] = points_to[0]
			last_element = points_to[-1]  # sheesh
			for neighbour in points_to:
				self.matrix[element][neighbour] = last_element

		for element, points_from in enumerate(incomingEdges):
			if not points_from:
				continue
			self.matrix[element][N + 1] = points_from[0]
			last_element = points_from[-1]  # woah
			for neighbour in points_from:
				self.matrix[element][neighbour] = last_element + N

		for element, unrelated in enumerate(unconnected):
			if not unrelated:
				continue
			self.matrix[element][N + 2] = unrelated[0]
			last_element = unrelated[-1]  # nice
			for node in unrelated:
				self.matrix[element][node] = -last_element

	def DFS(self, node, visited={}, ancestors={}):
		if node in visited:
			return visited
		visited[node] = 1
		ancestors[node] = 1
		for element, edge in enumerate(self.matrix[node][:-3]):
			outgoingEdge = edge is not None and self.size > edge >= 0
			if not outgoingEdge:
				continue
			assert element not in ancestors, "Graf zawiera cykl."
			visited = self.DFS(element, visited, copy.copy(ancestors))
		return list(visited)

	def topologicalSort(self):
		# Kahn's algorithm
		# Let's start with deepcopying the original matrix
		matrix = copy.copy(self.matrix)

		# Count incoming edges for each node
		degrees = [0] * self.size
		for original, edges in enumerate(matrix):
			for element, edge in enumerate(edges[:-3]):
				if (edge is not None) and (edge >= self.size):
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

			for neighbour, edge in enumerate(matrix[element][:-3]):
				if not (edge is not None and self.size > edge >= 0):
					continue
				degrees[neighbour] -= 1
				if degrees[neighbour] == 0:
					queue.append(neighbour)

			visited_count += 1

		assert visited_count == self.size, "Graf zawiera cykl."

		return topological_order

	def topologicalDFS(self):
		results, visited = [], [False] * self.size
		
		for node in range(self.size):
			if not visited[node]:
				self._topologicalDFS(node, visited, results)
		
		return results[::-1]

	def _topologicalDFS(self, node, visited, results):
		visited[node] = True

		for neighbour, edge in enumerate(self.matrix[node][:-3]):
			if edge is None or not (self.size > edge >= 0) or visited[neighbour]:
				continue
			self._topologicalDFS(neighbour, visited, results)

		results.append(node)


## ---- tests ----

def generateGraph(size):
	# wspolczynnik nasycenia ~= 50%
	m = AdjacencyMatrix(size)
	skip = False
	for start in range(size):
		for end in range(start, size):
			if not skip:
				m.addEdge(start, end)
				skip = True
			else:
				skip = False
	return m

sizes = []
start_size = 100
for i in range(15):
	sizes.append(start_size)
	start_size += 80

print("Liczba elementow,macierz grafu DFS [ns],macierz grafu Kahn [ns],", end="")
print("macierz incydencji DFS [ns], macierz incydencji Kahn [ns]")
for size in sizes:
	adjacency_matrix = generateGraph(size)
	graph_matrix = GraphMatrix(adjacency_matrix)

	start = timer()
	graph_matrix.topologicalDFS()
	end = timer()
	gm_dfs_ns = int(timedelta(seconds=end-start).total_seconds()*1000000)

	start = timer()
	graph_matrix.topologicalSort()
	end = timer()
	gm_kahn_ns = int(timedelta(seconds=end-start).total_seconds()*1000000)

	start = timer()
	adjacency_matrix.topologicalDFS()
	end = timer()
	am_dfs_ns = int(timedelta(seconds=end-start).total_seconds()*1000000)

	start = timer()
	adjacency_matrix.topologicalSort()
	end = timer()
	am_kahn_ns = int(timedelta(seconds=end-start).total_seconds()*1000000)


	print(",".join(map(str, [size, gm_dfs_ns, gm_kahn_ns, am_dfs_ns, am_kahn_ns])))
	del adjacency_matrix
	del graph_matrix
