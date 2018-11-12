import numpy as np

from newton import *

# tmp for testing
from quadratures import *
from composite_quadrature import *

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
	print(degree)
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
# incorrect!!!
#    P"(N,X) = ( (2*N-1)*(2*P'(N-1,X)+X*P"(N-1,X)-(N-1)*P'(N-2,X) ) / N
def ddL(n):

	if (n==0):
		return lambda x: 0.0

	elif (n==1):
		return lambda x: 0.0

	else:
		# P"(N,X) = ( (2*N-1)*(2*P'(N-1,X)+X*P"(N-1,X)-(N-1)*P'(N-2,X) ) / N
		#return lambda x: ( (2*n-1) * (2 * dL(n-1)(x)) + x * ddL(n-1)(x) - ((n-1) * dL(n-2)(x)) ) / n

		# approximate by fitting polynomial and taking derivatives
		c_om1 = coef_approximation(L(n), n)
		c_prime = polynomial_derivative(c_om1)
		c_double_prime = polynomial_derivative(c_prime)

		# [TODO]: create own poly eval function
		return lambda x: np.polyval(c_double_prime, x)

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
		# the approximate values of the abscissas.
		# these are good initial guesses
		x0=np.cos(np.pi*(i-0.25)/(order+0.5)) 

		# call newton to find the roots of the legendre polynomial
		Ffun, Jfun = L(order), dL(order)
		ri, _ = newton( Ffun, Jfun, x0 )

		roots.append(ri)

	# use symetric properties to find remmaining roots
	# the nodal abscissas are computed by finding the 
	# nonnegative zeros of the Legendre polynomial pm(x) 
	# with Newton’s method (the negative zeros are obtained from symmetry).
	roots = np.array(roots)

	# even. no center
	if order % 2==0:
		roots = np.concatenate( (-1.0*roots, roots[::-1]) )

	# odd. center root is 0.0
	else:
		roots = np.concatenate( (-1.0*roots, [0.0], roots[::-1]) ) 

	return roots


# find an approximation for roots of the derivative of the legendre polynomial 
# of given order
# on the unit interval [-1, 1]
def unit_lobatto_nodes(order, tol=1e-15, output=True):
	roots=[]

	print("Finding roots of the derivative of the Legendre polynomial of order ", order)

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


# find weights for the roots of the legendre polynomial 
# of given order 
# on the unit interval [-1, 1]
def unit_gauss_weights_and_nodes(order):

	# find roots of legendre polynomial  on unit interval
	nodes = unit_lengendre_roots(order)

	# calculate weights for unit interval
	# Ai = 2 / [ (1 - xi^2)* (p'n+1(xi))^2 ]  -- Gauss Legendre Weights
	weights = 2.0/( (1.0-nodes**2) * dL(order)(nodes)**2 )

	return weights, nodes

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

def lobatto_quadrature(f, a, b, order, weights=None, nodes=None):

	# if weights and nodes are None, compute them for the interval [a,b]
	if weights is None: 

		assert nodes is None

		# find weights for the legendre polynomial on the unit interval
		unit_weights, unit_nodes = unit_lobatto_weights_and_nodes(order)

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


def composite_lobatto_quadrature(f, a, b, order, m):

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
	unit_weights, unit_nodes = unit_lobatto_weights_and_nodes(order)

	# [TODO]: reuse calculation of end nodes and weights
	# iterate over subintervals
	for i in range(m):

		# define subintervals start and stop points
		ai, bi = a+i*h, a+(i+1)*h

		# project onto interval [a,b]
		weights, nodes = project_weights_and_nodes(ai, bi, unit_weights, unit_nodes)

		# call lobatto quadrature formula on this subinterval
		In, nlocal = lobatto_quadrature(f, ai, bi*h, order, weights=weights, nodes=nodes)

		# increment outputs
		Imn += In
		nf  += nlocal

	return Imn, nf


"""
import matplotlib.pyplot as plt

y_approx = lambda coeffs, xs: np.polyval(coeffs, xs)
x = np.linspace(-1.05, 1.05, num=80)


#
# show polynomial interpolation accuracy
#

order = 5
c = coef_approximation(L(order), order)


y = L(order)(x)
y_hats = y_approx(c, x)

L_roots = unit_lengendre_roots(order)
L_root_ys = L(order)(L_roots)
L_root_y_hats = y_approx(c, L_roots)

#plt.scatter(L_roots, L_root_ys, c='g', s=30)
plt.scatter(L_roots, L_root_y_hats, c='b', s=50,  marker='X')

plt.plot(x,y, c='g', linewidth=3)
plt.plot(x, y_hats, c='r', linestyle='--', linewidth=2)


plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.show()


#
# show derivative approximation
#

# take derivatice of this representation
#c_om1 = coef_approximation(L(order-1), order-1)
om1 = order-1
c_om1 = coef_approximation(L(om1), om1)
c_prime = polynomial_derivative(c_om1)


dL_y = dL(order-1)(x)
dL_y_hats = np.polyval(c_prime, x) #y_approx(c_prime, x)

dL_roots = unit_lobatto_nodes(order)
dL_roots = dL_roots[1:-1]



dL_root_ys = dL(order)(dL_roots)
dL_root_y_hats = np.polyval(c_prime, dL_roots) #y_approx(c_prime, dL_roots)

#plt.scatter(dL_roots, dL_root_ys, c='g', s=30)
plt.scatter(dL_roots, dL_root_y_hats, c='b', s=50, marker='X')

plt.plot(x, dL_y, c='g', linewidth=3)
plt.plot(x, dL_y_hats, c='r', linestyle='--', linewidth=2)

plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.show()


#
#  show 2nd derivative approximation
#

om1 = order-1
c_om1 = coef_approximation(L(om1), om1)
c_prime = polynomial_derivative(c_om1)
c_double_prime = polynomial_derivative(c_prime)

ddL_y_hats = np.polyval(c_double_prime, x) #y_approx(c_prime, x)

plt.plot(x, ddL_y_hats, c='r', linestyle='--', linewidth=2)

plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.show()
"""

"""

order = 5

ws, xs = unit_lobatto_weights_and_nodes(order)

for w,x in zip(ws,xs):
	print("w: %s, x: %s" % (w,x))
"""



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

	order = 6
	ms = [2, 20, 200, 2000]

	# tmp - test against profs method
	pqd = Gauss8


	#
	# Test gauss quadrature method
	#

	print("\n\n\nTesting lobatto quadrature method...\n")

	# my approx
	Iapprox, nf = lobatto_quadrature( f, a, b, order )

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

	print("\n\n\nTesting composite lobatto quadrature method...")

	for m in ms:

		print("\nm : ", m)
		print("order : ", order)

		# my approx
		Iapprox, nf = composite_lobatto_quadrature(f, a, b, order, m )

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

