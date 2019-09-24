from fenics import Expression, Constant

# Define right hand side
def f(t):

    f = Expression(
        'exp(-10.0 * (pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2))) * sign',
        sign = 1.0, degree = 1)
    if int(t) + 0.1 < t:

        f.sign = 0.0

    return f

# Define parameters
class Parameters:

    def __init__(self, 
                 
                # Define problem parameters
                nu = Constant(0.1), 
                beta = Constant((5.0, 0.0)), 
                zeta = Constant(500.0), 
                delta = Constant(0.01), 
                alpha = Constant(0.01), 
                gamma = Constant(1000.0), 

                # Define time step on the coarsest level
                dt = 0.02,

                # Define number of global time steps on the coarsest level
                N = 50,

                # Define number of fractional time-steps for fluid
                M = 2,

                # Define number of fractional time-steps for solid
                K = 1,

                # Define number of mesh cells
                nx = 80,  
                ny = 20, 

                # Define relaxation parameters
                tau = Constant(0.7), 
                abs_tol_relax = 1.0e-12,
                rel_tol_relax = 1.0e-6,
                maxiter_relax = 25, 

                # Define parameters for Newton's method
                eps = 1.0e-6, 
                abs_tol_newton = 1.0e-12,
                rel_tol_newton = 1.0e-6, 
                maxiter_newton = 15, 

                # Define parameters for GMRES method
                tol_gmres = 1.0e-6, 
                maxiter_gmres = 10,

                # Choose decoupling method
                relaxation = True,
                shooting = False,

                # Choose goal functional
                J1 = True,
                J2 = False):
    
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
        self.abs_tol_relax = abs_tol_relax
        self.rel_tol_relax = rel_tol_relax
        self.maxiter_relax = maxiter_relax
        
        self.eps = eps
        self.abs_tol_newton = abs_tol_newton
        self.rel_tol_newton = rel_tol_newton
        self.maxiter_newton = maxiter_newton
        
        self.tol_gmres = tol_gmres
        self.maxiter_gmres = maxiter_gmres

        self.relaxation = relaxation
        self.shooting = shooting

        self.J1 = J1
        self.J2 = J2
