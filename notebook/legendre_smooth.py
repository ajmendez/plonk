import numpy as np
from numpy.polynomial.legendre import legvander,legder
from scipy.linalg import qr,solve_triangular,lu_factor,lu_solve

class legendre_smooth(object):
    def __init__(self,n,k,a,m):
        self.k2 = 2*k

        # Uniform grid points
        x = np.linspace(-1,1,n)

        # Legendre Polynomials on grid 
        self.V = legvander(x,m-1)

        # Do QR factorization of Vandermonde for least squares 
        self.Q,self.R = qr(self.V,mode='economic')

        I = np.eye(m)
        D = np.zeros((m,m))
        D[:-self.k2,:] = legder(I,self.k2)

        # Legendre modal approximation of differential operator
        self.A = I-a*D

        # Store LU factors for repeated solves   
        self.PLU = lu_factor(self.A[:-self.k2,self.k2:])

    def fit(self,z):

        # Project data onto orthogonal basis 
        Qtz = np.dot(self.Q.T,z)

        # Compute expansion coefficients in Legendre basis
        zhat = solve_triangular(self.R,Qtz,lower=False)

        # Solve differential equation       
        yhat = np.zeros(len(zhat))
        q = np.dot(self.A[:-self.k2,:self.k2],zhat[:self.k2])
        r = zhat[:-self.k2]-q
        yhat[:self.k2] = zhat[:self.k2]
        yhat[self.k2:] = lu_solve(self.PLU,r)
        y = np.dot(self.V,yhat)
        return y