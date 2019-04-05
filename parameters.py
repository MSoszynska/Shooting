from fenics import*

# Define problem parameters
nu = Constant(0.1)
beta = Constant((5.0, 0.0))
zeta = Constant(500.0)
delta = Constant(0.01)
alpha = Constant(0.01)
gamma = Constant(1000.0)

# Define time step
dt = 0.0025

# Define number of global and fractional time steps
N = 10
M = 1
K = 1

# Define number of mesh cells
nx = 80
ny = 20

# Define relaxation parameters
tau = Constant(0.1)
tol = 1.0e-6
maxit = 15

# Define parameters for Newton's method
eps = 1.0e-6
tol_newton = 1.0e-8
maxiter_newton = 15

# Define parameters for GMRES method
tol_gmres = 1.0e-8
maxiter_gmres = 25

