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
__file__ = "adaptive_G7K15_quadrature.py"

import numpy as np

# tmp for testing
from quadratures import *
from composite_quadrature import *


class HashTable:

	def __init__(self):

		# create empty hash table
		self.hash_table = {}

	# hash table for checking if nodes have already been evaluated
	# f(xi)
	def reuse_evaluation(self, f, x):

		# if node has already been evaluated
		# return value from hash table
		if x in self.hash_table.values():

			# additional function evaluations == 0
			return self.hash_table[x]

		# otherwise value by calling function
		# additional function evaluations == 1
		value = f(x)

		# add value to hash table
		self.hash_table[x] = value
		
		return value

	def get_nf(self):

		# return the total number of function calls performed
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

	#If provided a hash table, save these locations in a tree structure using hash of the x value to check if the value has already been computed
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


	#
	# error approximation
	# 

	# importaint find good error approximation

	if abs(b-a) < 1:
		#err = abs(InG - InK) * abs(b-a)**8 / 2 # order 8 difference?
		#err = abs(InG - InK) * abs(b-a)**15 / 2 
		#err = abs(InG - InK) * abs(b-a)**7 #15 # order 8 difference?
		#err = abs(InG - InK) * abs(b-a)**12.86
		err = abs(InG - InK) * abs(b-a)**12.86
	else:
		err = abs(InG - InK) / abs(b-a) * 2

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


def sum_leaf_entries(node, entry_key):

	if node is None: 
		return 0

	# leaf node
	if ( node.left is None and node.right is None ): 
		return node.data_dict[entry_key]

	else: 
		return sum_leaf_entries(node.left, entry_key) + sum_leaf_entries(node.right, entry_key)

# count number of leaf nodes in binary tree 
def count_leaf_nodes(node): 

    if node is None: 
        return 0 

    if(node.left is None and node.right is None): 
        return 1 
    else: 
        return count_leaf_nodes(node.left) + count_leaf_nodes(node.right) 


# Gloabl error is the sum of the local errors in leaf nodes
# pass in the root node to start recursion
def global_error(root): 

	#return sum_leaf_entries(root, 'local_error') / count_leaf_nodes(root)
	return sum_leaf_entries(root, 'local_error')


def integral_approximation(root):

	return sum_leaf_entries(root, 'InK')


def return_leaf_nodes(node):

	if node is not None: 

		# leaf node
		if ( node.left is None and node.right is None ): 
			return [node]

		else: 
			return return_leaf_nodes(node.left) + return_leaf_nodes(node.right)


# if tolerence is not met, subdivide interval in node with largest local error
# return number of function evaluations and updated tree's root node
import random
def subdivide(root, f, hash_table, min_h=0.187):#min_h=1e-01): #min_h=0.25): #2):


	# find node with maximum local error
	leaf_nodes = return_leaf_nodes(root)

	# don't do this randomly!
	random.shuffle(leaf_nodes)

	divide = False
	for ln in leaf_nodes:
		if abs(ln.data_dict['b'] - ln.data_dict['a'])/2 >= min_h:
			max_error_node = ln
			divide = True

	if divide == False:
		print("Tree NOT Dividing")
		return None, 0

	for ln in leaf_nodes:

		# ensure the subinterval is not too small
		if abs(ln.data_dict['b'] - ln.data_dict['a'])/2 >= min_h:

			if ln.data_dict['local_error'] >= max_error_node.data_dict['local_error']:
				max_error_node = ln

	# create children by splitting the interval in two
	a, b = max_error_node.data_dict['a'], max_error_node.data_dict['b']
	c = a + (b-a)/2

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

	max_error_node.left = Node(data_dict_l)
	max_error_node.right = Node(data_dict_r)

	# return total function evaluations
	nf = nf_left + nf_right

	# return root
	return root, nf

def create_root(f, a, b, hash_table):

	InG, InK, nf, local_error = G7K15(f, a, b, hash_table)

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

def create_split_root(f, a, b, hash_table):

	data_dict = {
		'a' : a,
		'b' : b,
		'InG' : None,
		'InK' : None,
		'local_error' : abs(b-a)
	}

	root = Node(data_dict)

	# start with two subintervals
	root, _ = subdivide(root, f, hash_table)

	# return root node and number of function evaluations
	return root, hash_table.get_nf()


def G7K15_adaptive_quadrature(f, a, b, tol, maxit=1000, output=False):

	# check inputs
	if (b < a):
		raise ValueError('composite_gauss_quadrature error: b < a!')

	# create empty hash table
	hash_table = HashTable()


	#root, _ = create_split_root(f, a, b, hash_table)
	root, _ = create_root(f, a, b, hash_table)

	for i in range(maxit):

		# exit if tolerence is met
		g_err = global_error(root)
		n_sub_intervals = count_leaf_nodes(root)

		# display global error estimate
		if output:

			if i % 25 == 0:
				print("global error approximation: %s" % g_err)
				print("number of subintervals: %s" % n_sub_intervals)

		if g_err <= tol:

			if output:
				print("tolerence threshold met.")
				print("number of subintervals used: %s" %  count_leaf_nodes(root))

			return [integral_approximation(root), hash_table.get_nf()]

		# otherwise subdivide and try again
		new_root, _ = subdivide(root, f, hash_table)

		# tree stopped dividing
		if new_root == None:

			if output:
				print("tree stopped dividing")
				print("number of subintervals used: %s" %  count_leaf_nodes(root))

			return [integral_approximation(root), hash_table.get_nf()]
		root = new_root
		
	if output:
		print("failure to meet tolernce")
		print("number of subintervals used: %s" %  count_leaf_nodes(root))

	return [integral_approximation(root), hash_table.get_nf()]




if __name__ == "__main__":
	# set the integration interval, integrand function and parameters
	a = -1.0
	b = 4.0
	c = 0.5
	d = 1.0

	r1 = random.randint(-10, 10)
	r2 = random.randint(-10, 10)
	a = min(r1,r2)
	b= max(r1,r2)


	def f(x):
		return np.exp(c*x) + np.sin(d*x)

	# set the true integral
	Itrue = 1.0/c*(np.exp(c*b) - np.exp(c*a)) - 1.0/d*(np.cos(d*b)-np.cos(d*a))

	"""
		# test function
	def tanh(x):
		y = np.exp(-2.0 * x)
		return (1.0 - y) / (1.0 + y)

	def f(x):
		return tanh(x)
	"""
	#from scipy.integrate import quad

	# check against scipy
	#Itrue = quad(f, a, b)[0]



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



