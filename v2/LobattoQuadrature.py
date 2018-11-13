#
#  Lobatto
#

import warnings
warnings.filterwarnings("ignore")

import numpy as np


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

