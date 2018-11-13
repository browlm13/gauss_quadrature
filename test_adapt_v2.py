#!/usr/bin/env python3

# composite_test.py
#
# Script to run adaptive quadrature demo:
#   Tests the adaptive_trao() function that approximates 
#       \int_a^b f(x) dx
#   to a specified error tolerance.
#
# Daniel R. Reynolds
# Math5315
# Fall 2018

# imports
from numpy import *
from adaptive_trap import *
from scipy.integrate import quad
from pylab import *

#from adaptive_G7K15_quadrature import G7K15_adaptive_quadrature
from adaptive_lobatto_quadrature import lobatto_adaptive_quadrature

from AdaptiveQuadrature import *

# set the integration interval, integrand function and parameters
a = -3.0
b = 3.0

def f(x):
	return (cos(x**4) - exp(-x/3))

# true solution (from Mathematica)
Itrue = -5.388189110199078390464788757549832333192362851501884776675107808988626164717563118491875769130202907

"""
a = -1.0
b = 4.0
c = 0.5
d = 1.0

r1 = random.randint(-4, 0)
r2 = random.randint(0, 4)
a = min(r1,r2)
b= max(r1,r2)


def f(x):
	return np.exp(c*x) + np.sin(d*x)

# set the true integral
Itrue = 1.0/c*(np.exp(c*b) - np.exp(c*a)) - 1.0/d*(np.cos(d*b)-np.cos(d*a))
"""

# Adaptive Quadrature Class

low_order_method = LobattoQuadrature(4)
high_order_method = LobattoQuadrature(6)

ad = AdaptiveQuadrature(low_order_method, high_order_method)

# plot the function over this interval
x = linspace(a,b,1000)
plt.figure()
plt.plot(x,f(x))
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("integrand")
plt.show()

# set up tolerances
m = arange(4,10)

# Python's built-in routine
Iref, Ierr, infodict = quad(f, a, b, epsabs=1e-9, epsrel=0, full_output=1)
print("integral (tol = 1e-9):")
print('                I = {0:.16f},  err = {1:.2e},  nf = {2:d}'.format(Iref, abs(Itrue-Iref), infodict['neval']))
		
# approximate integral for each tolerance, output results
print("adaptive_trap:")

tols = []
err1s = []
n1s = []
err2s = []
n2s = []

for i in range(m.size):

	# perform approximation
	#I1, n1 = adaptive_trap(f, a, b, 10.0**(-m[i]))
	Iref, Ierr, infodict = quad(f, a, b, epsabs=10.0**(-m[i]), epsrel=0, full_output=1)
	I1 = Iref
	n1 = infodict['neval']

	#I1, n1 = G7K15_adaptive_quadrature(f, a, b, 10.0**(-m[i])) #, output=True)
	I1, n1 = ad.adaptive_quadrature(f, a, b, tol=10.0**(-m[i]), maxit=1000, min_h=1e-06, output=False)


	# output results
	print('\nM1:  tol = 1e-{0:2d}:  I = {1:.16f},  err = {2:.2e},  nf = {3:d}'.format(m[i], I1, abs(Itrue-I1), n1))
   
	#I2, n2 = G7K15_adaptive_quadrature(f, a, b, 10.0**(-m[i])) #, output=True)
	#I2, n2	= lobatto_adaptive_quadrature(f, a, b, 10.0**(-m[i])) #, output=True)
	I2, n2	= lobatto_adaptive_quadrature(f, a, b, 10.0**(-m[i])) #, output=True)
	print('M2:  tol = 1e-{0:2d}:  I = {1:.16f},  err = {2:.2e},  nf = {3:d}'.format(m[i], I2, abs(Itrue-I2), n2))
   
	tols += [10.0**(-m[i])]
	err1s += [abs(Itrue-I1)]
	n1s += [n1]
	err2s += [abs(Itrue-I2)]
	n2s += [n2]


def magnitude(x):
	"""order of magnitude of x"""
	return int(math.log10(x))

print("\nTRIAL RESULTS:")
nf_winners = []
disqualifications = []
total_trial = len(m)
for tol, e1, n1, e2, n2 in zip(tols, err1s, n1s, err2s, n2s):

	ndiff = n1 - n2

	# with in tol
	method1_sucsess = True
	if e1 !=0:
		method1_sucsess = magnitude(tol) >= magnitude(e1)

	method2_sucsess = True
	if e2 !=0:
		method2_sucsess = magnitude(tol) >= magnitude(e2)


	print("Method 1 output bellow tolerance threshold: ", method1_sucsess)
	print("Method 2 output bellow tolerance threshold: ", method2_sucsess)

	print("Method 1 nf - Method 2 nf: ", ndiff)

	if ndiff < 0:
		nwinner = "Method 1"
		nf_winners += ["Method 1"]
	elif ndiff > 0:
		nwinner = "Method 2"
		nf_winners += ["Method 2"]
	else:
		nwinner = "Tie"
		nf_winners += [["Method 1", "Method 2"]]

	print("nf winner: ", nwinner)

	disquals = ""
	ds = []
	if method1_sucsess == False:
		disquals += "\tMethod 1"
		ds += ["Method 1"]
	if method2_sucsess == False:
		disquals += "\tMethod 2"
		ds += ["Method 2"]
	if len(disquals) == 0:
		disquals = "None"
	disqualifications += ds

	print("Disqualifications: ", disquals)

print("\n\nFinal Results:")

print("\nnf winners: ")
print(nf_winners)

print("\nDisqualifications: ")
print(disqualifications)

# end of script


