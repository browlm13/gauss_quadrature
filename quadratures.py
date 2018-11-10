# quadratures.py
#
# Collection of single-interval quadrature functions.  Each of these
# apply a specific quadrature rule to approximate an integral
#    int_a^b f(x) dx
# using a given quadrature rule over the interval [a,b].
#
# All of these require that f have the calling syntax
#    y = f(x)
# where y has the same size and shape as x, in the case
# that x is [numpy] array-valued.
#
# Inputs:  f        function to integrate
#          a        left end point of integration
#          b        right end point of integration
#
# Outputs: In       value of the numerical integral
#          nf       # of calls to f
#
# Daniel R. Reynolds
# Math 5315 / CSE 7365
# Fall 2018

# imports
from numpy import *

def midpoint(f, a, b):
    """ Usage: [In,nf] = midpoint(f, a, b) """
    In = (b-a)*f(0.5*(a+b))
    nf = 1
    return [In, nf]


def trapezoid(f, a, b):
    """ Usage: [In,nf] = trapezoid(f, a, b) """
    In = 0.5*(b-a)*(f(a) + f(b) )
    nf = 2
    return [In, nf]


def Simpson(f, a, b):
    """ Usage: [In,nf] = Simpson(f, a, b) """
    In = (b-a)/6.0*(f(a) + 4.0*f(0.5*(a+b)) + f(b) )
    nf = 3
    return [In, nf]


def Gauss3(f, a, b):
    """ Usage: [In,nf] = Gauss3(f, a, b) """
    x0 = 0.5*(a+b) + 0.5*(b-a)*(-sqrt(3.0/5.0))
    x1 = 0.5*(a+b) + 0.5*(b-a)*(0.0)
    x2 = 0.5*(a+b) + 0.5*(b-a)*(sqrt(3.0/5.0))
    w0 = 5.0/9.0
    w1 = 8.0/9.0
    w2 = 5.0/9.0
    In = 0.5*(b-a)*( w0*f(x0) + w1*f(x1) + w2*f(x2) )
    nf = 3
    return [In, nf]


def Gauss5(f, a, b):
    """ Usage: [In,nf] = Gauss5(f, a, b) """
    x0 = 0.5*(a+b) + 0.5*(b-a)*0.0
    x1 = 0.5*(a+b) + 0.5*(b-a)*(-sqrt(5.0-2.0*sqrt(10.0/7.0))/3.0)
    x2 = 0.5*(a+b) + 0.5*(b-a)*( sqrt(5.0-2.0*sqrt(10.0/7.0))/3.0)
    x3 = 0.5*(a+b) + 0.5*(b-a)*(-sqrt(5.0+2.0*sqrt(10.0/7.0))/3.0)
    x4 = 0.5*(a+b) + 0.5*(b-a)*( sqrt(5.0+2.0*sqrt(10.0/7.0))/3.0)
    w0 = 128.0/225.0
    w1 = (322.0+13.0*sqrt(70.0))/900.0
    w2 = (322.0+13.0*sqrt(70.0))/900.0
    w3 = (322.0-13.0*sqrt(70.0))/900.0
    w4 = (322.0-13.0*sqrt(70.0))/900.0
    In = 0.5*(b-a)*( w0*f(x0) + w1*f(x1) + w2*f(x2) + w3*f(x3) + w4*f(x4) )
    nf = 5
    return [In, nf]


def Gauss8(f, a, b):
    """ Usage: [In,nf] = Gauss8(f, a, b) """
    x0 = 0.5*(a+b) + 0.5*(b-a)*(-0.18343464249564980493)
    x1 = 0.5*(a+b) + 0.5*(b-a)*( 0.18343464249564980493)
    x2 = 0.5*(a+b) + 0.5*(b-a)*(-0.52553240991632898581)
    x3 = 0.5*(a+b) + 0.5*(b-a)*( 0.52553240991632898581)
    x4 = 0.5*(a+b) + 0.5*(b-a)*(-0.79666647741362673959)
    x5 = 0.5*(a+b) + 0.5*(b-a)*( 0.79666647741362673959)
    x6 = 0.5*(a+b) + 0.5*(b-a)*(-0.96028985649753623168)
    x7 = 0.5*(a+b) + 0.5*(b-a)*( 0.96028985649753623168)
    w0 = 0.36268378337836198296
    w1 = 0.36268378337836198296
    w2 = 0.31370664587788728733
    w3 = 0.31370664587788728733
    w4 = 0.22238103445337447054
    w5 = 0.22238103445337447054
    w6 = 0.10122853629037625915
    w7 = 0.10122853629037625915
    In = 0.5*(b-a)*( w0*f(x0) + w1*f(x1) + w2*f(x2) + w3*f(x3) + w4*f(x4) + w5*f(x5) + w6*f(x6) + w7*f(x7) )
    nf = 8
    return [In, nf]


# end of functions
