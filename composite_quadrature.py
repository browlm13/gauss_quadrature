# composite_quadrature.py
#
# Daniel R. Reynolds
# Math 5315 / CSE 7365
# Fall 2018

def composite_quadrature(f, a, b, qd, m):
    """ Usage: In, nf = composite_quadrature(f, a, b, qd, m)
    
        This routine numerically approximates the integral
           int_a^b f(x) dx
        using a composite quadrature rule with m uniformly sized
        subintervals.  We require that f have the calling syntax
           y = f(x)
        where y has the same size and shape as x, in the case
        that x is array-valued.  We require that the single-interval
        quadrature method, qd, have calling syntax
           In, nf = qd(f, ai, bi
        where f is the integrand function, [ai,bi] is the subinterval
        to integrate over, In is the numerical integral over this
        subinterval, and nf is the number of calls that qd made to
        the function f

        Inputs:  f        integrand function
                 a        left end point of integration
                 b        right end point of integration
                 qd       single-interval quadrature function
                 m        number of subintervals

        Outputs: Imn      value of the composite quadrature result
                 nf       total number of calls to f
    """

    # check inputs
    if (b < a):
        raise ValueError('composite_quadrature error: b < a!')
    if (m < 1):
        raise ValueError('composite_quadrature error: m < 1!')

    # set up subinterval width
    h = 1.0*(b-a)/m

    # initialize results
    Imn = 0.0
    nf = 0

    # iterate over subintervals
    for i in range(m):

        # call quadrature formula on this subinterval
        In, nlocal = qd(f, a+i*h, a+(i+1)*h)

        # increment outputs
        Imn += In
        nf  += nlocal

    return [Imn, nf]