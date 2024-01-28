from functools import total_ordering
import copy
import math
import random
from timeit import default_timer as timer
from datetime import timedelta



@total_ordering
class Node:
	def __init__(self, value=None):
		self.left = None
		self.right = None
		self.value = value
		# height represents height of the node and all of nodes below it
		# imagine A->B->C, A gets height 3, B gets height 2, C gets height 1
		# it's the largest height of l/r subtree + 1
		# you need to keep track of it during tree modifications
		self.height = 1

	def deleteValue(self, value):
		"""Returns expected node after deletion, will crash on incorrect value
		you need to recalculate heights after use!"""
		if value < self.value:
			self.left = self.left.deleteValue(value)
			return self
		if value > self.value:
			self.right = self.right.deleteValue(value)
			return self

		# Leaf?
		if self.left is None and self.right is None:
			return None

		# 1 child only?
		if self.left is not None and self.right is None:
			return self.left

		if self.right is not None and self.left is None:
			return self.right

		# node has 2 children
		# lets get left-most element of right subtree
		parentNode, childNode = self, self.right
		while True:
			if childNode.left is not None:
				parentNode = childNode
				childNode = childNode.left
			else:
				break

		# replace our deleted value
		self.value = childNode.value

		# now we either have a leaf or a node with single child
		# test for leaf first
		if childNode.left is None and childNode.right is None:
			parentNode.left = None
		else:  # left is guaranteed to be None
			parentNode.left = childNode.right

		# seems ok?
		return self

	def findValue(self, value):
		if self.value == value:
			return self
		
		if value < self.value and self.left:
			result = self.left.findValue(value)
			if result:
				return result
		
		if value > self.value and self.right:
			result = self.right.findValue(value)
			if result:
				return result
		
		return None


	def balanceFactor(self):
		left = 0 if self.left is None else self.left.height
		right = 0 if self.right is None else self.right.height
		return left - right

	def inOrder(self):
		nodes = []
		if self.left is not None:
			nodes.extend(self.left.inOrder())

		nodes.append(self)

		if self.right is not None:
			nodes.extend(self.right.inOrder())

		return nodes

	def preOrder(self):
		nodes = []

		nodes.append(self)

		if self.left is not None:
			nodes.extend(self.left.inOrder())
		
		if self.right is not None:
			nodes.extend(self.right.inOrder())

		return nodes

	def postOrder(self):
		nodes = []

		if self.left is not None:
			nodes.extend(self.left.inOrder())
		
		if self.right is not None:
			nodes.extend(self.right.inOrder())

		nodes.append(self)

		return nodes

	def min(self):
		leftmost, leftmost_parent = self, self
		#print("Szukanie najmniejszego elementu: ", end="")
		while leftmost:
			#print(repr(leftmost), end=", ")
			leftmost_parent = leftmost
			leftmost = leftmost.left

		#print("found")
		return leftmost_parent

	def max(self):
		rightmost, rightmost_parent = self, self
		#print("Szukanie największego elementu: ", end="")
		while rightmost:
			#print(repr(rightmost), end=", ")
			rightmost_parent = rightmost
			rightmost = rightmost.right

		#print("found")
		return rightmost_parent

	def balanceFactor(self):
		lh = 0 if self.left is None else self.left.height
		rh = 0 if self.right is None else self.right.height
		return lh - rh


	def recalculateHeight(self):
		self.height = 1 + max(
			0 if self.left is None else self.left.height,
			0 if self.right is None else self.right.height
		)

	def recalculateHeights(self):
		if self.left:
			self.left.recalculateHeights()
		if self.right:
			self.right.recalculateHeights()
		self.recalculateHeight()

	def __eq__(self, other):
		return other.value == self.value

	def __lt__(self, other):
		return self.value < other.value

	def __str__(self):
		leftString = "" if self.left is None else "L(" + str(self.left) + ")"
		rightString = "" if self.right is None else "R(" + str(self.right) + ")"
		return f"#{self.value} {leftString} {rightString}"

	def __repr__(self):
		return "#" + str(self.value)

class AVLTree:
	def __init__(self):
		self.root = None

	def insertNode(self, searchNode, node, balance=False):
		if searchNode is None:
			return node
		elif searchNode > node:
			searchNode.left = self.insertNode(searchNode.left, node, balance)
		else:
			searchNode.right = self.insertNode(searchNode.right, node, balance)

		searchNode.height = 1 + max(
			self.getHeight(searchNode.left),
			self.getHeight(searchNode.right)
		)

		if balance:
			balanceFactor = self.getBf(searchNode)
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

	def getHeight(self, node):
		return 0 if node is None else node.height

	def getBf(self, node):
		return 0 if node is None else node.balanceFactor()

	def insertValue(self, value):
		self.root = self.insertNode(self.root, Node(value))

	def deleteValue(self, value):
		if not self.root:
			return
		self.root = self.root.deleteValue(value)
		if self.root:
			self.root.recalculateHeights()

	def insertBinarySearch(self, valuesList):
		valuesList = sorted(valuesList)
		return self._insertMedian(valuesList)

	def insertBalanced(self, valuesList):
		for value in valuesList:
			self.root = self.insertNode(self.root, Node(value), True)

	def _insertMedian(self, valuesList):
		if not valuesList:
			return
		
		median = (len(valuesList) - 1) // 2
		self.insertValue(valuesList[median])
		self._insertMedian(valuesList[:median])
		self._insertMedian(valuesList[median+1:])

	def leftRotate(self, X):
		Y = X.right
		Temp = Y.left

		Y.left = X
		X.right = Temp

		lh = 0 if X.left is None else X.left.height
		rh = 0 if X.right is None else X.right.height 

		X.height = max(lh, rh) + 1

		lh = 0 if Y.left is None else Y.left.height
		rh = 0 if Y.right is None else Y.right.height 

		Y.height = max(lh, rh) + 1

		return Y

	def rightRotate(self, X):
		Y = X.left
		if not Y:
			return X
		Temp = Y.right

		Y.right = X
		X.left = Temp

		lh = 0 if X.left is None else X.left.height
		rh = 0 if X.right is None else X.right.height 

		X.height = max(lh, rh) + 1

		lh = 0 if Y.left is None else Y.left.height
		rh = 0 if Y.right is None else Y.right.height 

		Y.height = max(lh, rh) + 1
		return Y

	"""DSW algorithm implemented according to the Wikipedia pseudocode"""
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
			x = None
			# ensure unique
			while x is None or x in T:
				x = random.randrange(-size, size)
			T.append(x)
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

if False:
	testList = [5, 7, 10, 11, 12, 13, 14, 15, 16, 3, 2, 1, 4, 8, 9, 6]

	bst = AVLTree()
	#bst.insertBinarySearch(testList)
	for value in testList:
		bst.root = bst.insertNode(bst.root, Node(value))


	print("Drzewo BST\n", bst.root)
	print("Elementy in-order:", bst.root.inOrder())

	bst.dsw()

	print("Drzewo BST\n", bst.root)
	print("Elementy in-order:", bst.root.inOrder())


	tree = AVLTree()
	tree.insertBalanced(testList)
	print("\nDrzewo AVL\n", tree.root)
	print("Elementy in-order:", tree.root.inOrder())
	print("Balance factor:", tree.root.balanceFactor())

	# wyszukanie w drzewie elementu o najmniejszej i największej wartości 
	#   i wypisanie ścieżki poszukiwania 
	#   (od korzenia do elementu szukanego),
	print("")
	minElement = tree.root.min()
	print("Najmniejszy element", minElement)
	maxElement = tree.root.max()
	print("Najwiekszy element", maxElement)

	# usunięcie elementu drzewa o wartości klucza podanej przez użytkownika
	userInput = 3
	print(f"\nUsuwanie elementu #{userInput} z drzewa AVL:")
	tree.root = tree.root.deleteValue(userInput)
	print(tree.root)

	userInput = 5
	print(f"\nSzukanie elementu #{userInput} w drzewie AVL:")
	print(tree.root.findValue(userInput))

	print("\nWypisanie drzewa w poszczególnych porządkach:")
	print("- in-order: "+" ".join([repr(x) for x in tree.root.inOrder()]))
	print("- pre-order: "+" ".join([repr(x) for x in tree.root.preOrder()]))
	print("- post-order: "+" ".join([repr(x) for x in tree.root.postOrder()]))

	print("\nUsuwanie kolejnych elementów metodą postOrder: ")
	tree2 = copy.deepcopy(tree)
	print("Usuwane elementy po kolei: ", end="")
	for node in tree2.root.postOrder():
		print(node.value, end=", ")
		tree2.deleteValue(node.value)

	print("\nDrzewo po usunięciu: "+repr(tree2.root))



tree = AVLTree()
_n = []
for i in range(17):  # todo: 17
	_n.append(round(2 ** (i*4/5+2)))

elements = []
for i in _n:
	el = ArrayGenerator.randomArray(i)
	elements.append(el)

avls = []
bsts = []
print("Tworzenie drzewa,,")
print("N elementow, czas AVL ns, czas BST ns")
for i, el in enumerate(elements):
	start = timer()
	tree = AVLTree()
	tree.insertBalanced(el)
	end = timer()

	avls.append(tree)
	delta_ns = int(timedelta(seconds=end-start).total_seconds()*1000000)

	valuesList = sorted(el)
	start = timer()
	tree = AVLTree()
	tree._insertMedian(valuesList)
	end = timer()

	bsts.append(tree)
	delta_bst = int(timedelta(seconds=end-start).total_seconds()*1000000)


	print(f"{_n[i]},{delta_ns},{delta_bst}")

print(",,")
print("In order,,")
print("N elementow, czas AVL ns, czas BST ns")
for i, _ in enumerate(_n):
	avl = avls[i]
	bst = bsts[i]

	start = timer()
	res = avl.root.inOrder()
	end = timer()
	delta_ns = int(timedelta(seconds=end-start).total_seconds()*1000000)
	# print(avl.root.height, avl.root.balanceFactor())

	start = timer()
	res = bst.root.inOrder()
	end = timer()
	delta_bst = int(timedelta(seconds=end-start).total_seconds()*1000000)
	# print(bst.root.height)

	print(f"{_n[i]},{delta_ns},{delta_bst}")

print(",,")
print("Minimum,,")
print("N elementow, czas AVL ns, czas BST ns")
for i, _ in enumerate(_n):
	avl = avls[i]
	bst = bsts[i]

	start = timer()
	res = avl.root.min()
	end = timer()
	delta_ns = int(timedelta(seconds=end-start).total_seconds()*1000000)

	start = timer()
	res = bst.root.min()
	end = timer()
	delta_bst = int(timedelta(seconds=end-start).total_seconds()*1000000)

	print(f"{_n[i]},{delta_ns},{delta_bst}")

print(",,")
print("Rownowazenie,,")
print("N elementow, BST,")
for i, _ in enumerate(_n):
	avl = avls[i]
	bst = bsts[i]
	start = timer()
	res = bst.dsw()
	end = timer()
	delta_bst = int(timedelta(seconds=end-start).total_seconds()*1000000)

	print(f"{_n[i]},{delta_bst},")




#print(elements)
