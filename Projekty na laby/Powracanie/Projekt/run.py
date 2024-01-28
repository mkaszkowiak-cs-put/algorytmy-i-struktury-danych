import copy
import math
import sys
from timeit import default_timer as timer
from datetime import timedelta
from collections import deque


class SuccessorList:
	# Directed graph
	def __init__(self, size):
		self.matrix = []
		self.size = None
		self.directed = True
		self.initMatrix(size)

	def initMatrix(self, size):
		for _ in range(size):
			self.matrix.append([])
		self.size = size

	def addEdge(self, start, end):
		# [!] no duplicate detection
		self.matrix[start].append(end)
		return self

	def e(self, start, end):
		return self.addEdge(start, end)

	def removeEdge(self, start, end):
		self.matrix[start].remove(end)
		return self 

	def hasEdge(self, start, end):
		for edge in self.matrix[start]:
			if edge == end:
				return True
		return False

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



	def __str__(self):
		res = ""
		for node, edges in enumerate(self.matrix):
			res += f"{node}: {', '.join(map(str, edges))}\n"
		return res

	def __repr__(self):
		return str(self)

class AdjacencyMatrix:
	def __init__(self, size):
		self.matrix = []
		self.size = None
		self.directed = False
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
		# Order is important so A->A edges are 1
		self.matrix[end][start] = -1 if self.directed else 1 
		self.matrix[start][end] = 1
		return self

	def e(self, start, end):
		return self.addEdge(start, end)

	def removeEdge(self, start, end):
		self.matrix[end][start] = 0
		self.matrix[start][end] = 0
		return self

	def hasEdge(self, start, end):
		return self.matrix[start][end]

	def hcycle(self, start_node):
		path, path_size = [start_node], 0
		visited = [False] * self.size

		def hamiltonian(node):
			nonlocal path, path_size, visited
			# assume suitable node will be found
			visited[node] = True
			path_size += 1
			
			for points_to, edge in enumerate(self.matrix[node]):
				if edge != 1:
					continue
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

	def ecycle(self):
		# using deepcopy due to removing edges
		graph = copy.deepcopy(self) 

		results = []
		def euler(node):
			nonlocal graph, results
			# iterate over succesors
			for neighbor, edge in enumerate(graph.matrix[node]):
				if edge != 1:
					continue
				graph.removeEdge(node, neighbor)
				euler(neighbor)
			results.append(node)

		euler(0)

		return results[::-1]

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


## ---- tests ----

def generateGraph(size, saturation=5, graphType=AdjacencyMatrix):
	m = graphType(size)
	n = 0
	for start in range(size):
		for end in range(start, size):
			if start == end:
				# dont count as viable edges
				# the task defines max edges as n(n-1), not n^2
				continue

			if n < saturation:
				m.addEdge(start, end)
			n = n + 1
			n = n % 10

	return m

sizes = []
start_size = 3
for i in range(15):
	sizes.append(start_size)
	start_size += 1

print("Liczba wierzchołków,Nasycenie grafu [%],Reprezentacja grafu,Czas wyszukiwania cyklu Eulera [us],Czas wyszukiwania cyklu Hamiltona [us]")
for size in sizes:
	for graphType in (SuccessorList, AdjacencyMatrix):
		for saturation in range(1, 10):  # 10-90%
			
			graph = generateGraph(size, saturation, graphType)
			# maybe initing variables beforehand will help with more precise timing
			start, end = timer(), timer()
			
			start = timer()
			graph.ecycle()
			end = timer()
			euler_us = int(timedelta(seconds=end-start).total_seconds()*1000000)

			start = timer()
			graph.hcycle(0)
			end = timer()
			hamilton_us = int(timedelta(seconds=end-start).total_seconds()*1000000)

			graphStr = "Lista następników (skierowany)" if graph.directed else "Macierz incydencji (nieskierowana)"

			print(", ".join(map(str, [size, saturation*10, graphStr, euler_us, hamilton_us])))



"""
print("directed: hamilton cycle")
m = SuccessorList(6).e(0,1).e(1,4).e(1,2).e(2,3).e(4,2).e(4,3).e(3,5).e(5,0)
print(m)
print(m.hcycle(0))

print("directed: no hamilton cycle - should show False")
m = SuccessorList(6).e(0,1).e(1,4).e(1,2).e(2,3).e(4,2).e(4,3).e(3,5)
print(m)
print(m.hcycle(0))

print("directed: euler cycle")
m = SuccessorList(5).e(1,0).e(0,3).e(3,4).e(4,0).e(0,2).e(2,1)
print(m)
print(m.ecycle())

print("directed: no euler cycle (not enough nodes will be found)")
m = SuccessorList(5).e(0,3).e(3,4).e(4,0).e(0,2).e(2,1)
print(m)
print(m.ecycle())

print("undirected: hamilton cycle")
m = AdjacencyMatrix(6).e(0,1).e(1,4).e(1,2).e(2,3).e(4,2).e(4,3).e(3,5).e(5,0)
print(m)
print(m.hcycle(0))

print("undirected: no hamilton cycle - should show False")
m = AdjacencyMatrix(6).e(0,1).e(1,4).e(1,2).e(2,3).e(4,2).e(4,3).e(3,5)
print(m)
print(m.hcycle(0))

print("undirected: euler cycle")
m = AdjacencyMatrix(5).e(1,0).e(0,3).e(3,4).e(4,0).e(0,2).e(2,1)
print(m)
print(m.ecycle())

print("undirected: no euler cycle (not enough nodes will be found)")
m = AdjacencyMatrix(5).e(0,3).e(3,4).e(4,0).e(0,2).e(2,1)
print(m)
print(m.ecycle())
"""
	#start = timer()
	#adjacency_matrix.topologicalSort()
	#end = timer()
	#am_kahn_us = int(timedelta(seconds=end-start).total_seconds()*1000000)
