from fenics import Expression, Constant

# Define right hand side
def f(t):
    f = Expression('exp(-10.0*(pow(x[0] - 0.5, 2) + pow(x[1] - \
                    0.5, 2)))*sign', sign = 1.0, degree = 1)
    if (int(t) + 0.1 < t):
        f.sign = 0.0
    return f

class Parameters:

    def __init__(self, 
                 
                # Define problem parameters
                nu = Constant(0.1), 
                beta = Constant((5.0, 0.0)), 
                zeta = Constant(500.0), 
                delta = Constant(0.01), 
                alpha = Constant(0.01), 
                gamma = Constant(1000.0), 

                # Define time step
                dt = 0.01, 

                # Define number of global and fractional time steps
                N = 10,  
                M = 1, 
                K = 1, 

                # Define number of mesh cells
                nx = 80,  
                ny = 20, 

                # Define relaxation parameters
                tau = Constant(0.7), 
                tol_relax = 1.0e-12, 
                maxiter_relax = 20, 

                # Define parameters for Newton's method
                eps = 1.0e-6, 
                tol_newton = 1.0e-8, 
                maxiter_newton = 15, 

                # Define parameters for GMRES method
                tol_gmres = 1.0e-6, 
                maxiter_gmres = 25):
    
        self.nu = nu
        self.beta = beta
        self.zeta = zeta
        self.delta = delta
        self.alpha = alpha
        self.gamma = gamma
        
        self.dt = dt
        
        self.N = N
        self.M = M
        self.K = K
        
        self.nx = nx
        self.ny = ny
        
        self.tau = tau
        self.tol_relax = tol_relax
        self.maxiter_relax = maxiter_relax
        
        self.eps = eps
        self.tol_newton = tol_newton
        self.maxiter_newton = maxiter_newton
        
        self.tol_gmres = tol_gmres
        self.maxiter_gmres = maxiter_gmres



