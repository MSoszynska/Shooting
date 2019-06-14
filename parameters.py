from __future__ import print_function
from fenics import*

# Define right hand side
def f(t):
    f = Expression('exp(-10.0*(pow(x[0] - 0.5, 2) + pow(x[1] - \
                    0.5, 2)))*sign', sign = 1.0, degree = 1)
    if (int(t) + 0.1 < t):
        f.sign = 0.0
    return f

# Define problem parameters
nu = Constant(0.1)
beta = Constant((5.0, 0.0))
zeta = Constant(500.0)
delta = Constant(0.01)
alpha = Constant(0.01)
gamma = Constant(1000.0)

# Define time step
dt = 0.01

# Define number of global and fractional time steps
N = 100
M = 1
K = 1

# Define number of mesh cells
nx = 80
ny = 20

# Define relaxation parameters
tau = Constant(0.7)
tol_relax = 1.0e-12
maxiter_relax = 10

# Define parameters for Newton's method
eps = 1.0e-6
tol_newton = 1.0e-6
maxiter_newton = 15

# Define parameters for GMRES method
tol_gmres = 1.0e-6
maxiter_gmres = 25

