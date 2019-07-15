from fenics import dot, grad, Measure, Function, \
                   FunctionSpace, Constant, project, \
                   DirichletBC, TrialFunction, \
                   TestFunction, split, solve, \
                   VectorFunctionSpace, inner
from spaces import boundary
from coupling import flip    
    
# Define solid function
def solid_problem(u, v, fluid, solid, interface, param):
    
    # Define variational form
    def a_s(u_s, v_s, phi_s, psi_s):
        return param.zeta*dot(grad(u_s), grad(phi_s))*solid.dx \
             + param.delta*dot(grad(v_s), grad(phi_s))*solid.dx \
             - v_s*psi_s*solid.dx \
             + param.delta*grad(v_s)[1]*phi_s*solid.ds
        
    # Store old solutions
    u_s_n_K = Function(solid.V.sub(0).collapse())
    v_s_n_K = Function(solid.V.sub(1).collapse())
    u_s_n_K.assign(u.s_n)
    v_s_n_K.assign(v.s_n)
        
    # Store old boundary values
    v_s_n_i_K = Function(solid.V.sub(1).collapse())
    v_s_n_i_K.assign(v.s_n_i)
    
    # Initialize interface values
    v_s_i = Function(solid.V.sub(1).collapse())
            
    # Compute fractional steps for solid problem
    for k in range(param.K):
            
        # Update boundary values
        v_s_i.assign(project((param.K - k - 1.0)/param.K*v.s_n_i + \
              (k + 1.0)/param.K*flip(v.f, \
              solid.V.sub(1).collapse(), param), solid.V.sub(1).collapse()))
                
        # Define Dirichlet boundary conditions
        bc_u_s_0 = DirichletBC(solid.V.sub(0), Constant(0.0), boundary)
        bc_v_s_0 = DirichletBC(solid.V.sub(1), Constant(0.0), boundary)
        bcs_s = [bc_u_s_0, bc_v_s_0]
                
        # Define trial and test functions
        U_s = TrialFunction(solid.V)
        (u_s, v_s) = split(U_s)
        Phi_s = TestFunction(solid.V)
        (phi_s, psi_s) = split(Phi_s)
                
        # Define scheme
        A_s = v_s*phi_s*solid.dx \
            + u_s*psi_s*solid.dx \
            + 0.5*param.dt/param.K*a_s(u_s, v_s, phi_s, psi_s)
        L_s = v_s_n_K*phi_s*solid.dx \
            + u_s_n_K*psi_s*solid.dx \
            - 0.5*param.dt/param.K*a_s(u_s_n_K, v_s_n_K, phi_s, psi_s) \
            + 0.5*param.dt/param.K*param.nu*dot(grad(v_s_i), \
                                                solid.n)*phi_s*solid.ds \
            + 0.5*param.dt/param.K*param.nu*dot(grad(v_s_n_i_K), \
                                                solid.n)*phi_s*solid.ds 
                
        # Solve solid problem
        U_s = Function(solid.V)
        solve(A_s == L_s, U_s, bcs_s)
        (u_s, v_s) = U_s.split(U_s)
                
        # Update solid solution
        u_s_n_K.assign(u_s)
        v_s_n_K.assign(v_s)
                
        # Update boundary condition
        v_s_n_i_K.assign(project(v_s_i, solid.V.sub(1).collapse()))
        
    # Assign final values
    u.s.assign(u_s)
    v.s.assign(v_s)
    
    return 
    
