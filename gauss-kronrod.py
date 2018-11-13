#!/usr/bin/env python

"""

Gauss-Kronrod 7-15 automatic adaptive quadratures

	* total of 15 function evaluations for each subinterval

	The integral is then estimated by the Kronrod rule K15
	and the error can be estimated as |G7−K15|


	Gauss nodes					Weights

	±0.949107912342759		∗	0.129484966168870
	±0.741531185599394		∗	0.279705391489277
	±0.405845151377397		∗	0.381830050505119
	 0.000000000000000		∗	0.417959183673469

	Kronrod nodes				Weights

	±0.991455371120813			0.022935322010529
	±0.949107912342759		∗	0.063092092629979
	±0.864864423359769			0.104790010322250
	±0.741531185599394		∗	0.140653259715525
	±0.586087235467691			0.169004726639267
	±0.405845151377397		∗	0.190350578064785
	±0.207784955007898			0.204432940075298
	 0.000000000000000		∗	0.209482141084728

"""

__author__  = "LJ Brown"
__file__ = "gauss-kronrod.py"

import numpy as np

# tmp for testing
from quadratures import *
from composite_quadrature import *


class HashTable:

	def __init__(self):

		# create empty hash table
		self.hash_table = {}

		# create root and initalize nf
		#self.nf = 0
		#self.nf = len(self.hash_table)

	# hash table for checking if nodes have already been evaluated
	# f(xi)
	def reuse_evaluation(self, f, x):

		# if node has already been evaluated
		# return value
		if x in self.hash_table.values():
			# additional function evaluations == 0
			#self.nf += 0
			return self.hash_table[x]

		# otherwise compute and return
		value = f(x)
		self.hash_table[x] = value
		# additional function evaluations == 1
		#self.nf += 1
		return value

	def get_nf(self):
		return len(self.hash_table)


from itertools import product
def G7K15(f, a, b, hash_table=None):
	""" 

	Usage: 
		[InG, InK, nf, err] = GKK15(f, a, b) 

	InG            -- Gauss intergral approximation
	InK            -- Kronrod intergral approximation
	nf = 15 	   -- total of 15 function evaluations
	err = |G7−K15| -- approximation of error over interval [a,b]

	"""

	# K
	x0  = 0.5*(a+b) + 0.5*(b-a)*(-0.991455371120813)
	x1  = 0.5*(a+b) + 0.5*(b-a)*( 0.991455371120813)
	# GK
	x2  = 0.5*(a+b) + 0.5*(b-a)*(-0.949107912342759)
	x3  = 0.5*(a+b) + 0.5*(b-a)*( 0.949107912342759)
	# K
	x4  = 0.5*(a+b) + 0.5*(b-a)*(-0.864864423359769)
	x5  = 0.5*(a+b) + 0.5*(b-a)*( 0.864864423359769)
	# GK
	x6  = 0.5*(a+b) + 0.5*(b-a)*(-0.741531185599394)
	x7  = 0.5*(a+b) + 0.5*(b-a)*( 0.741531185599394)
	# K
	x8  = 0.5*(a+b) + 0.5*(b-a)*(-0.586087235467691)
	x9  = 0.5*(a+b) + 0.5*(b-a)*( 0.586087235467691)
	# GK
	x10 = 0.5*(a+b) + 0.5*(b-a)*(-0.405845151377397)
	x11 = 0.5*(a+b) + 0.5*(b-a)*( 0.405845151377397)
	# K
	x12 = 0.5*(a+b) + 0.5*(b-a)*(-0.207784955007898)
	x13 = 0.5*(a+b) + 0.5*(b-a)*( 0.207784955007898)
	# GK
	x14 = 0.5*(a+b) + 0.5*(b-a)*( 0.000000000000000)

	# K
	w0  = 0.022935322010529
	w1  = 0.022935322010529
	# GK
	w2  = 0.063092092629979
	w3  = 0.063092092629979
	# K
	w4  = 0.104790010322250
	w5  = 0.104790010322250
	# GK
	w6  = 0.140653259715525
	w7  = 0.140653259715525
	# K
	w8  = 0.169004726639267
	w9  = 0.169004726639267
	# GK
	w10 = 0.190350578064785
	w11 = 0.190350578064785
	# K
	w12 = 0.204432940075298
	w13 = 0.204432940075298
	# GK
	w14 = 0.209482141084728


	G_xs = [x2, x3,  x6, x7, x10, x11,  x14] 
	K_xs = [x0, x1, x4, x5, x8, x9, x12, x13]

	G_ws = [w2, w3,  w6, w7, w10, w11,  w14]
	K_ws = [w0, w1, w4, w5, w8, w9, w12, w13]

	#If provided a hash table, save these locations in a tree structure usinga hash of the x value to check if the value has already been computed
	# for that particular node
	if hash_table == None:

		# calculate integral approximations, reusing Gauss evaluations
		InG = 0.5*(b-a)*( w2*f(x2) + w3*f(x3) + w6*f(x6) + w7*f(x7) + w10*f(x10) + w11*f(x11) + w14*f(x14) )
		InK = InG + 0.5*(b-a)*( w0*f(x0) + w1*f(x1) + w4*f(x4) + w5*f(x5) + w8*f(x8) + w9*f(x9) + w12*f(x12) + w13*f(x13) )

		# total of 15 funciton calls
		nf = 15

	else:

		G_fxs = []
		K_fxs = []

		for x in G_xs:
			fx = hash_table.reuse_evaluation(f, x)
			G_fxs.append(fx)

		for x in K_xs:
			fx = hash_table.reuse_evaluation(f, x)
			K_fxs.append(fx)

		nf = hash_table.get_nf()

		# calculate integral approximations, reusing all function evaluations

		InG = 0.5*(b-a)*sum([w*fx for w,fx in zip(G_ws,G_fxs)])
		InK = InG + 0.5*(b-a)*sum([w*fx for w,fx in zip(K_ws,K_fxs)])


	# error approximation
	#err = abs(InG - InK)
	#err = abs(InG - InK)/abs(InK)
	#err = 0.5*abs(InG - InK)/abs(b-a)*abs(InK)
	#err =abs(InG - InK) * (abs(b-a) * 0.5) #**2
	err = abs(InK/InG)*abs(InG - InK) #* (abs(b-a) * 0.5) #abs(InG/InK) #* (abs(b-a) * 0.5) #**2
	#print(err)

	return [InG, InK, nf, err]


# binary tree node
class Node: 
	
	def __init__(self, data_dict): 

		# data_dict contains:
		#
		#   interval begin and end point [a,b]
		#       a, b
		#	local error --|InG−InK| -- approximation of error over subinterval
		#      local_err
		#   Gauss integral of subinterval
		#       InG
		#   Gauss integral of subinterval
		#       InG
		#
		self.data_dict = data_dict

		# node children
		self.left = None
		self.right = None

# Gloabl error is the sum of the local errors in leaf nodes
# pass in the root node to start recursion
def global_error(node): 

	if node is None: 
		return 0

	# leaf node
	if ( node.left is None and node.right is None ): 
		return node.data_dict['local_error']

	else: 
		return global_error(node.left) + global_error(node.right)


# Get leaf node with largest local error  
# returns leaf node with maximum local error in given binary tree 
def find_max_local_error(node, min_h=0.9): #1e-01

	# leaf node
	if ( node.left is None and node.right is None ): 		
		# make sure nodes subinterval width does not fall bellow minimum width
		if abs(node.data_dict['b'] - node.data_dict['a'])/2 < min_h:
			#print("max local error node cannot be divided any further")
			#print(abs(node.data_dict['b'] - node.data_dict['a'])/2)
			#print(node.data_dict['a'])
			#print(node.data_dict['b'])
			return Node({'local_error':0})
		else:
	  		return node
  
	# Return node containgmaximum:
	# 1) max local error in left subtree 
	# 2) max local error in right subtree 

	l_max_node = find_max_local_error(node.left)
	r_max_node = find_max_local_error(node.right)

	if l_max_node == None:
		if r_max_node == None:
			return None
		else:
			if r_max_node.data_dict['local_error'] != 0:
				return r_max_node
	else:
		if l_max_node.data_dict['local_error'] != 0:
			if r_max_node == None:
					return l_max_node
			else:
				if r_max_node.data_dict['local_error'] != 0:
					if (l_max_node.data_dict['local_error'] > r_max_node.data_dict['local_error']):
						return l_max_node
					return r_max_node
		else:
			if r_max_node == None:
				if r_max_node.data_dict['local_error'] != 0:
					return r_max_node
			return None

	#if (l_max_node.data_dict['local_error'] > r_max_node.data_dict['local_error']):
	#  return l_max_node

	#if r_max_node.data_dict['local_error'] != 0:
	#	return r_max_node

	#return None

# if tolerence is not met, subdivide interval in node with largest local error
# return number of function evaluations and updated tree's root node
def subdivide(root, f, hash_table): #, min_h=1e-07):


	# find node with maximum local error
	max_error_node = find_max_local_error(root)

	#print(max_error_node)
	if max_error_node is None:
		return root, 0

	# create children by splitting the interval in two
	a, b = max_error_node.data_dict['a'], max_error_node.data_dict['b']

	c = a + (b-a)/2

	#print("a: %s, c: %s, b: %s" % (a,c,b))

	left_a, left_b = a, c
	right_a, right_b = c, b

	# compute integral approximation for each node
	InG_left, InK_left, nf_left, local_error_left = G7K15(f, left_a, left_b, hash_table)
	InG_right, InK_right, nf_right, local_error_right = G7K15(f, right_a, right_b, hash_table)

	# create nodes and add them to tree
	data_dict_l = {
		'a' : left_a,
		'b' : left_b,
		'InG' : InG_left,
		'InK' : InK_left,
		'local_error' : local_error_left

	}

	data_dict_r = {
		'a' : right_a,
		'b' : right_b,
		'InG' : InG_right,
		'InK' : InK_right,
		'local_error' : local_error_right
	}

	#print(local_error_left)
	#print(local_error_right)

	max_error_node.left = Node(data_dict_l)
	max_error_node.right = Node(data_dict_r)

	# return total function evaluations
	nf = nf_left + nf_right

	# return root
	return root, nf

def create_root(f, a, b, hash_table):

	InG, InK, nf, local_error = G7K15(f, a, b)

	data_dict = {
		'a' : a,
		'b' : b,
		'InG' : InG,
		'InK' : InK,
		'local_error' : local_error
	}

	root = Node(data_dict)

	# return root node and number of function evaluations
	return root, nf


def integral_approximation(node):

	if node is None: 
		return 0

	# leaf node
	if ( node.left is None and node.right is None ): 
		return node.data_dict['InK']

	else: 
		return integral_approximation(node.left) + integral_approximation(node.right)


# count number of leaf nodes in binary tree 
def count_leaf_nodes(node): 

    if node is None: 
        return 0 
    if(node.left is None and node.right is None): 
        return 1 
    else: 
        return count_leaf_nodes(node.left) + count_leaf_nodes(node.right) 


"""
class AdaptiveQuadrature:

	def __init__:

		# create empty hash table
		self.hash_table = {}

		# create root and initalize nf
		self.root, self.nf = create_root(f, a, b, self.hash_table)

	# hash table for checking if nodes have already been evaluated
	# f(xi)
	def reuse_evaluation(self, f, x):

		# if node has already been evaluated
		# return value
		if x in self.hash_table.values():
			# additional function evaluations == 0
			#self.nf += 0
			return self.hash_table[x]

		# otherwise compute and return
		value = f(x)
		self.hash_table[x] = value
		# additional function evaluations == 1
		self.nf += 1
		return value



	def G7K15_adaptive_quadrature(self, f, a, b, tol, maxit=10000):

		# check inputs
		if (b < a):
			raise ValueError('composite_gauss_quadrature error: b < a!')

		# create empty hash table
		#self.hash_table = {}
		#root, self.nf = create_root(f, a, b, hash_table)

		for i in range(maxit):

			# exit if tolerence is met
			g_err = global_error(root)
			n_sub_intervals = count_leaf_nodes(root)
			g_err = global_error(root)/n_sub_intervals**3

			# tmp display global error estimate
			if i % 500 == 0:
				print(g_err)

			if g_err <= tol:
				return [integral_approximation(root), self.nf]

			# otherwise subdivide and try again
			root, nf_local = subdivide(root, f, self.hash_table)
			self.nf += nf_local
			
		print("failure to meet tolernce")
		return [integral_approximation(root), nf]
"""

def G7K15_adaptive_quadrature(f, a, b, tol, maxit=200):

	# check inputs
	if (b < a):
		raise ValueError('composite_gauss_quadrature error: b < a!')

	# create empty hash table
	hash_table = HashTable()
	#root, nf = create_root(f, a, b, hash_table)
	root, _ = create_root(f, a, b, hash_table)

	for i in range(maxit):

		# exit if tolerence is met
		g_err = global_error(root)
		n_sub_intervals = count_leaf_nodes(root)
		g_err = global_error(root)/n_sub_intervals**3

		# tmp display global error estimate
		if i % 50 == 0:
			print(g_err)

		if g_err <= tol:
			#return [integral_approximation(root), self.nf]
			return [integral_approximation(root), hash_table.get_nf()]

		# otherwise subdivide and try again
		#root, nf_local = subdivide(root, f, self.hash_table)
		#self.nf += nf_local
		root, _ = subdivide(root, f, hash_table)
		
	print("failure to meet tolernce")
	#return [integral_approximation(root), nf]
	return [integral_approximation(root), hash_table.get_nf()]

if __name__ == "__main__":
	# set the integration interval, integrand function and parameters
	a = -5.0
	b = 4.0
	c = 0.5
	d = 5.0

	def f(x):
		return np.exp(c*x) + np.sin(d*x)

	# set the true integral
	Itrue = 1.0/c*(np.exp(c*b) - np.exp(c*a)) - 1.0/d*(np.cos(d*b)-np.cos(d*a))



	# quadrature paramters

	ms = [2, 20, 200, 2000]

	# tmp - test against profs method
	pqd = Gauss8


	# 
	# Test composite quadrature method
	#

	print("\n\n\nTesting composite G7K15 quadrature method...")

	for m in ms:

		print("\nm : ", m)

		# professors approx
		pIapprox, pnf = composite_quadrature(f, a, b, pqd, m)

		# my approx
		tol = abs(b-a)/m
		Iapprox, nf = G7K15_adaptive_quadrature(f, a, b, tol)

		# display true integral value
		print("\nI = ", Itrue)

		# display my approx
		print("My Integral Aprroximation : ", Iapprox)

		# display professors approx
		print("Professors Integral Aprroximation : ", pIapprox)

		print("Diffrence : ", abs(Iapprox - pIapprox))

		# display my nf 
		print("\nMy nf : ", nf)

		# display professors nf 
		print("Professors nf : ", pnf)

		print("Diffrence : ", abs(nf - pnf))


		# check error agains I true from class test function
		true_error = Itrue - Iapprox
		print("\nMy Error : ", true_error)

		p_true_error = Itrue - pIapprox
		print("Professors Error : ", p_true_error)

		print("Diffrence : ", abs(true_error-p_true_error))




