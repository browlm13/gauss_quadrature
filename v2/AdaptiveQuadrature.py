#!/usr/bin/env python

"""


	Adaptive Quadrature


"""

__author__  = "LJ Brown"
__file__ = "AdaptiveQuadrature.py"

import numpy as np

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

# count number of leaf nodes
def count_leaf_nodes(node): 

    if node is None: 
        return 0 

    if(node.left is None and node.right is None): 
        return 1 

    else: 
        return count_leaf_nodes(node.left) + count_leaf_nodes(node.right) 


# Gloabl error is the sum of the local errors in leaf nodes
def global_error(root): return sum_leaf_entries(root, 'local_error') 

# integral approximation is the sum of the higher order method approximations
def integral_approximation(root): return sum_leaf_entries(root, 'In2')


def return_leaf_nodes(node):

	if node is not None: 

		# leaf node
		if ( node.left is None and node.right is None ): 
			return [node]

		else: 
			return return_leaf_nodes(node.left) + return_leaf_nodes(node.right)

import random
from itertools import product

class AdaptiveQuadrature:

	def __init__(self, low_order_method, high_order_method, min_h=1e-06, method_order_difference=None, variable_h=False):

		# method should contain a member with the O(h^q) order approximation, q, called order
		self.low_order_method = low_order_method
		self.high_order_method = high_order_method

		self.min_h = min_h
		self.method_order_difference = method_order_difference
		self.variable_h = variable_h

	def create_root(self, a, b):

		# compute first approximation
		In1, In2, local_error = self.quad(a, b)

		data_dict = {
			'a' : a,
			'b' : b,
			'In1' : In1,
			'In2' : In2,
			'local_error' : local_error
		}

		self.root = Node(data_dict)

	# if tolerence is not met, subdivide interval in node with largest local error
	# return number of function evaluations and updated tree's root node
	def sub_divide(self): #, min_h=0.187, tol=None, err=None):#min_h=1e-01): #min_h=0.25): #2):


		# allow decrease in self.min_h for..
		if self.variable_h:
				self.max_local_error = getattr(self, "max_local_error", None)
				if self.max_local_error is None:
					self.max_local_error = np.inf

				if self.tol/self.max_local_error >= 1e-03:
					#print("DECREASING min_h")
					self.min_h = self.min_h/2
		


		# find node with maximum local error
		leaf_nodes = return_leaf_nodes(self.root)

		# don't do this randomly!
		random.shuffle(leaf_nodes)

		divide = False
		for ln in leaf_nodes:
			if abs(ln.data_dict['b'] - ln.data_dict['a'])/2 >= self.min_h:
				max_error_node = ln
				divide = True

		if divide == False:
			print("Tree NOT Dividing")
			return False

		self.max_local_error = max_error_node.data_dict['local_error']

		for ln in leaf_nodes:

			# ensure the subinterval is not too small
			if abs(ln.data_dict['b'] - ln.data_dict['a'])/2 >= self.min_h:

				if ln.data_dict['local_error'] >= max_error_node.data_dict['local_error']:
					max_error_node = ln

		# create children by splitting the interval in two
		a, b = max_error_node.data_dict['a'], max_error_node.data_dict['b']
		c = a + (b-a)/2

		# compute integral approximation for each node using lower order and higher order methods for each
		a_left, b_left = a, c
		a_right, b_right = c, b

		In1_left, In2_left, local_error_left = self.quad(a_left, b_left)
		In1_right, In2_right, local_error_right = self.quad(a_right, b_right)

		# create nodes and add them to tree
		data_dict_l = {
			'a' : a_left,
			'b' : b_left,
			'In1' : In1_left,
			'In2' : In2_left,
			'local_error' : local_error_left

		}

		data_dict_r = {
			'a' : a_right,
			'b' : b_right,
			'In1' : In1_right,
			'In2' : In2_right,
			'local_error' : local_error_right
		}

		max_error_node.left = Node(data_dict_l)
		max_error_node.right = Node(data_dict_r)

		# tree is still dividing
		return True

	def quad(self, a, b):
		""" 

		Usage: 
			[In1, In2, err] = self.quad(f, a, b) 

		In1            -- Intergral approximation from low order method
		In2            -- Intergral approximation from high order method
		err 		   -- approximation of error over interval [a,b]

		"""

		ws1, xs1 = self.low_order_method.get_weights_and_nodes(a, b)
		ws2, xs2 = self.high_order_method.get_weights_and_nodes(a, b)

		# use hash table to retreive any previously computed values
		# to  minimize number of function calls -- nf -- hash_table.get_nf()
		fxs1 = []
		fxs2 = []
		for x in xs1:
			fx = self.hash_table.reuse_evaluation(self.f, x)
			fxs1.append(fx)

		for x in xs2:
			fx = self.hash_table.reuse_evaluation(self.f, x)
			fxs2.append(fx)

		# calculate integral approximations, reusing all function evaluations
		#In1 = 0.5*(b-a)*sum([w*fx for w,fx in zip(ws1,fxs1)]) 
		#In2 = 0.5*(b-a)*sum([w*fx for w,fx in zip(ws2,fxs2)])
		In1 = sum([w*fx for w,fx in zip(ws1,fxs1)]) 
		In2 = sum([w*fx for w,fx in zip(ws2,fxs2)])

		#
		# error approximation
		# 

		# importaint find good error approximation

		# if method_order_difference memeber is defined use that
		method_order_difference = getattr(self, "method_order_difference", None)
		if method_order_difference is None:
			method_order_difference = self.high_order_method.order - self.low_order_method.order

		if abs(b-a) < 1:
			err = abs(In1 - In2) * abs(b-a)**method_order_difference 
		else:
			err = abs(In2 - In1) / abs(b-a) * 2


		return [In1, In2, err]


	def adaptive_quadrature(self, f, a, b, tol=1e-10, maxit=1000, output=False):
		
		self.f = f
		self.tol = tol
		self.output = output

		# check inputs
		if (b < a):
			raise ValueError('adaptive_quadrature error: b < a!')

		# create empty hash table
		self.hash_table = HashTable()

		# create root node of tree
		self.create_root(a, b)

		for i in range(maxit):

			# exit if tolerence is met
			g_err = global_error(self.root)
			n_sub_intervals = count_leaf_nodes(self.root)

			# display global error estimate
			if self.output:

				if i % 5 == 0:
					print("global error approximation: %s" % g_err)
					print("number of subintervals: %s" % n_sub_intervals)

			if g_err <= self.tol:

				if self.output:
					print("tolerence threshold met.")
					print("number of subintervals used: %s" %  count_leaf_nodes(self.root))

				return [integral_approximation(self.root), self.hash_table.get_nf()]

			# otherwise subdivide and try again
			err = abs(g_err - tol)
			still_dividing = self.sub_divide()

			# tree stopped dividing
			if not still_dividing:

				if self.output:
					print("tree stopped dividing")
					print("number of subintervals used: %s" %  count_leaf_nodes(self.root))

				return [integral_approximation(self.root), self.hash_table.get_nf()]
			
		if self.output:
			print("failure to meet tolernce")
			print("number of subintervals used: %s" %  count_leaf_nodes(self.root))

		return [integral_approximation(self.root), self.hash_table.get_nf()]

"""
#
#  Lobatto
#

import warnings
warnings.filterwarnings("ignore")


# generate the Legendre polynomial function of order n recusrivley
#    P(0,X) = 1
#    P(1,X) = X
#    P(N,X) = ( (2*N-1)*X*P(N-1,X)-(N-1)*P(N-2,X) ) / N
def L(n):

	if (n==0):
		# P(0,X) = 1
		return lambda x: x*0+1.0

	elif (n==1):
		# P(1,X) = X
		return lambda x: x

	else:
		# P(N,X) = ( (2*N-1)*X*P(N-1,X)-(N-1)*P(N-2,X) ) / N
		return lambda x: ( (2.0*n-1.0) * x * L(n-1)(x)-(n-1) * L(n-2)(x) ) / n


# generate derivative function of the Legendre polynomials of 
# order n recursivley
#    P'(0,X) = 0
#    P'(1,X) = 1
#    P'(N,X) = ( (2*N-1)*(P(N-1,X)+X*P'(N-1,X)-(N-1)*P'(N-2,X) ) / N
def dL(n):

	#[TODO]: allow evaluation at 0

	if (n==0):
		return lambda x: x*0

	elif (n==1):
		return lambda x: x*0+1.0

	else:
		# (1 − x2)pn′(x) = n[−xpn(x) + pn−1(x)]
		return lambda x: (n/(x**2-1.0))*(x*L(n)(x)-L(n-1)(x))



#
#   get coefficients approximation of polynomial
#
from newton_interp import *
def coef_approximation(p, order):


	# maintain parity of order +1
	n = 50 + order +1
	r =  1
	xs = np.linspace(-r, r, num=n)
	ys = p(xs)



	# [TODO]: fix coeffients method
	# replace with 'c = coeffients(xs, ys)'
	degree = n #n #order + 1
	#print(degree)
	c = np.polyfit(xs,ys,degree)
	#c = coeffients(xs, ys)

	return c

def polynomial_derivative(coefficients):

	# compute coefficients for first derivative of p with coefficients c
	# [TODO]: create own method
	c_prime = np.polyder(coefficients)

	return c_prime


# generate second derivative of the Lengendre polynomials of
# order n recursivley
#    P"(0,X) = 0
#    P"(1,X) = 0
#    P"(N,X) = ( (2*N-1)*(2*P'(N-1,X)+X*P"(N-1,X)-(N-1)*P'(N-2,X) ) / N
def ddL(n):

	if (n==0):
		return lambda x: 0.0

	elif (n==1):
		return lambda x: 0.0

	else:
		# P"(N,X) = ( (2*N-1)*(2*P'(N-1,X)+X*P"(N-1,X)-(N-1)*P'(N-2,X) ) / N
		#return lambda x: ( (2*n-1) * (2 * dL(n-1)(x)) + x * ddL(n-1)(x) - ((n-1) * dL(n-2)(x)) ) / n
		#return lambda x: grad(dL(n))(x)

		# approximate by fitting polynomial and taking derivatives
		#c_om1 = coef_approximation(L(order-1), order-1)
		c_om1 = coef_approximation(L(n), n)
		c_prime = polynomial_derivative(c_om1)
		c_double_prime = polynomial_derivative(c_prime)

		# [TODO]: create own poly eval function
		return lambda x: np.polyval(c_double_prime, x)


from newton import *
# find an approximation for roots of the derivative of the legendre polynomial 
# of given order
# on the unit interval [-1, 1]
def unit_lobatto_nodes(order, tol=1e-15, output=True):
	roots=[]

	#print("Finding roots of the derivative of the Legendre polynomial of order ", order)

	# The polynomials are alternately even and odd functions
	# so evaluate only half the number of roots

	# order - 1, lobatto polynomial is derivative of legendre polynomial of order n-1
	order = order-1

	for i in range(1,int(order/2) +1):

		# initial guess, x0, for ith root 
		# the approximate values of the abscissas.
		# these are good initial guesses
		#x0=np.cos(np.pi*(i-0.25)/(order+0.5)) 
		x0=np.cos(np.pi*(i+0.1)/(order+0.5))  # not sure why this inital guess is better

		# call newton to find the roots of the lobatto polynomial
		#Ffun, Jfun = dL(order-1), ddL(order-1) 
		Ffun, Jfun = dL(order), ddL(order) 
		ri, _ = newton( Ffun, Jfun, x0 )

		roots.append(ri)

	# remove roots close to zero
	cleaned_roots = []
	tol = 1e-08
	for r in roots:
		if abs(r) >= tol:
			cleaned_roots += [r]
	roots = cleaned_roots

	# use symetric properties to find remmaining roots
	# the nodal abscissas are computed by finding the 
	# nonnegative zeros of the Legendre polynomial pm(x) 
	# with Newton’s method (the negative zeros are obtained from symmetry).
	roots = np.array(roots)
	
	# add -1 and 1 to tail ends
	# check parity of order + 1

	# even. no center 
	#if order % 2==0:
	if (order + 1) % 2==0:
		roots = np.concatenate( ([-1.0], -1.0*roots, roots[::-1], [1.0]) )

	# odd. center root is 0.0
	else:
		roots = np.concatenate( ([-1.0], -1.0*roots, [0.0], roots[::-1], [1.0] ) )

	return roots


# find weights for the lobatto polynomial
# of given order 
# on the unit interval [-1, 1]
def unit_lobatto_weights_and_nodes(order):

	# find roots of legendre polynomial  on unit interval
	nodes = unit_lobatto_nodes(order)

	# calculate weights for unit interval
	# Ai = 2 / [ (1 - xi^2)* (p'n+1(xi))^2 ]  -- Gauss Legendre Weights
	#weights = 2.0/( (1.0-nodes**2) * dL(order)(nodes)**2 )

	# wi = 2/(n(n-1) * Pn-1(xi)^2)
	weights = 2.0/( (order*(order-1)) * L(order-1)(nodes)**2 )

	return weights, nodes


# given unit weights and nodes on interval [-1,1] map to interval [a,b]
def project_weights_and_nodes(a, b, unit_weights, unit_nodes):

	# project onto interval [a,b]
	nodes = 0.5*(b-a)*unit_nodes + 0.5*(a+b)
	weights = 0.5*(b-a)*unit_weights

	return weights, nodes


class LobattoQuadrature:

	def __init__(self, order):

		self.order = order

		self.unit_weights, self.unit_nodes = unit_lobatto_weights_and_nodes(self.order)


	def get_weights_and_nodes(self, a, b):

		# project onto interval [a,b]
		weights, nodes = project_weights_and_nodes(a, b, self.unit_weights, self.unit_nodes)

		return weights, nodes
"""

"""
if __name__ == "__main__":

	# for testing
	from quadratures import *
	from composite_quadrature import *

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


	# Adaptive Quadrature Class

	low_order_method = LobattoQuadrature(5)
	high_order_method = LobattoQuadrature(7)

	ad = AdaptiveQuadrature(low_order_method, high_order_method)


	# quadrature paramters

	# number of subintervals for other method
	ms = [2, 20, 200, 2000]


	# set up tolerances
	low = 4
	high = len(ms) + low
	ts = np.arange(low,high)
	tols = [10.0**(-t) for t in ts]

	# tmp - test against profs method
	pqd = Gauss8


	# 
	# Test composite quadrature method
	#

	print("\n\n\nTesting composite G7K15 quadrature method...")

	for tol, m in zip(tols, ms):

		print("\nm : ", m)

		# professors approx
		pIapprox, pnf = composite_quadrature(f, a, b, pqd, m)

		# my approx
		Iapprox, nf = ad.adaptive_quadrature(f, a, b, tol=1e-10, maxit=1000, min_h=1e-06, output=False)

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

"""

