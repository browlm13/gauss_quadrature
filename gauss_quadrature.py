import numpy as np

from newton import *

# tmp for testing
from quadratures import *
from composite_quadrature import *

# generate the Legendre polynomial function of order n recusrivley
def L(n):

	if (n==0):
		return lambda x: x*0+1.0

	elif (n==1):
		return lambda x: x

	else:
		return lambda x: ((2.0*n-1.0)*x*L(n-1)(x)-(n-1)*L(n-2)(x))/n

# generate derivative function of the Legendre polynomials of 
# order n recursivley
def dL(n):

	if (n==0):
		return lambda x: x*0

	elif (n==1):
		return lambda x: x*0+1.0

	else:
		return lambda x: (n/(x**2-1.0))*(x*L(n)(x)-L(n-1)(x))


# find an approximation for roots of the legendre polynomial 
# of given order
# on the unit interval [-1, 1]
def unit_lengendre_roots(order, tol=1e-15, output=True):
	roots=[]

	print("Finding roots of Legendre polynomial of order ", order)

	# The polynomials are alternately even and odd functions
	# so evaluate only half the number of roots
	for i in range(1,int(order/2) +1):

		# initial guess, x0, for ith root 
		x0=np.cos(np.pi*(i-0.25)/(order+0.5)) 

		# call newton to find the roots of the legendre polynomial
		Ffun, Jfun = L(order), dL(order)
		ri, _ = newton( Ffun, Jfun, x0 )

		roots.append(ri)

	# use symetric properties to find remmaining roots
	roots = np.array(roots)

	# even. no center
	if order % 2==0:
		roots = np.concatenate( (-1.0*roots, roots[::-1]) )

	# odd. center root is 0.0
	else:
		roots = np.concatenate( (-1.0*roots, [0.0], roots[::-1]) ) 

	return roots

# find weights for the roots of the legendre polynomial 
# of given order 
# on the unit interval [-1, 1]
def unit_gauss_weights_and_nodes(order):

	# find roots of legendre polynomial  on unit interval
	nodes = unit_lengendre_roots(order)

	# calculate weights for unit interval
	weights = 2.0/( (1.0-nodes**2)*(dL(order)(nodes)**2) )

	return weights, nodes


# given unit weights and nodes on interval [-1,1] map to interval [a,b]
def project_weights_and_nodes(a, b, unit_weights, unit_nodes):

	# project onto interval [a,b]
	nodes = 0.5*(b-a)*unit_nodes + 0.5*(a+b)
	weights = 0.5*(b-a)*unit_weights

	return weights, nodes


def gauss_quadrature(f, a, b, order, weights=None, nodes=None):

	# if weights and nodes are None, compute them for the interval [a,b]
	if weights is None: 

		assert nodes is None

		# find weights for the legendre polynomial on the unit interval
		unit_weights, unit_nodes = unit_gauss_weights_and_nodes(order)

		# project onto interval [a,b]
		weights, nodes = project_weights_and_nodes(a, b, unit_weights, unit_nodes)

	# compute integral approximation
	Iapprox = sum( weights * f(nodes) )

	# record the number of function calls for f as nf
	nf = order

	return Iapprox, nf


def composite_gauss_quadrature(f, a, b, order, m):

	# m in number of sub intervals

	# check inputs
	if (b < a):
		raise ValueError('composite_gauss_quadrature error: b < a!')
	if (m < 1):
		raise ValueError('composite_gauss_quadrature error: m < 1!')

	# set up subinterval width
	h = 1.0*(b-a)/m

	# initialize results
	Imn = 0.0
	nf = 0

	# compute the unit weights and nodes for the lobatto quadrature 
	# of given order
	unit_weights, unit_nodes = unit_gauss_weights_and_nodes(order)

	# [TODO]: reuse calculation of end nodes and weights
	# iterate over subintervals
	for i in range(m):

		# define subintervals start and stop points
		ai, bi = a+i*h, a+(i+1)*h

		# project onto interval [a,b]
		weights, nodes = project_weights_and_nodes(ai, bi, unit_weights, unit_nodes)

		# call lobatto quadrature formula on this subinterval
		In, nlocal = gauss_quadrature(f, ai, bi*h, order, weights=weights, nodes=nodes)

		# increment outputs
		Imn += In
		nf  += nlocal

	return Imn, nf



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



	# gauss quadrature paramters and composite  gauss quadrature paramters

	order = 8
	ms = [2, 20, 200, 2000]

	# tmp - test against profs method
	pqd = Gauss8


	#
	# Test gauss quadrature method
	#

	print("\n\n\nTesting gauss quadrature method...\n")

	# my approx
	Iapprox, nf = gauss_quadrature( f, a, b, order )

	# professors approx
	pIapprox, pnf = pqd(f, a, b)

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

	# 
	# Test composite gauss quadrature method
	#

	print("\n\n\nTesting composite gauss quadrature method...")

	for m in ms:

		print("\nm : ", m)
		print("order : ", order)

		# my approx
		Iapprox, nf = composite_gauss_quadrature(f, a, b, order, m )

		# professors approx
		pIapprox, pnf = composite_quadrature(f, a, b, pqd, m)

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
