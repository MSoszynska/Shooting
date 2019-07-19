from fenics import (Function, FunctionSpace, dot, grad,
                    project, DirichletBC, Constant, 
                    TrialFunction, split, TestFunction, solve)
from spaces import boundary
from forms import a_s
from coupling import fluid_to_solid


# Define solid function
def solid_problem(u, v, fluid, solid, interface, param):

    # Store old solutions
    u_s_n_K = Function(solid.V_u)
    v_s_n_K = Function(solid.V_v)
    u_s_n_K.assign(u.s_n)
    v_s_n_K.assign(v.s_n)
        
    # Store old boundary values
    v_s_n_i_K = Function(solid.V_v)
    v_s_n_i_K.assign(v.s_n_i)
    
    # Initialize interface values
    v_s_i = Function(solid.V_v)
    
    # Define Dirichlet boundary conditions
    bc_u_s_0 = DirichletBC(solid.V.sub(0), Constant(0.0), boundary)
    bc_v_s_0 = DirichletBC(solid.V.sub(1), Constant(0.0), boundary)
    bcs_s = [bc_u_s_0, bc_v_s_0]
            
    # Compute fractional steps for solid problem
    for k in range(param.K):
            
        # Update boundary values
        v_s_i.assign(project((param.K - k - 1.0)/param.K*v.s_n_i + 
              + (k + 1.0)/param.K*fluid_to_solid(v.f, 
              fluid, solid, param, 1), solid.V_v))
                
        # Define trial and test functions
        U_s = TrialFunction(solid.V)
        (u_s, v_s) = split(U_s)
        Phi_s = TestFunction(solid.V)
        (phi_s, psi_s) = split(Phi_s)
                
        # Define scheme
        A_s = (v_s*phi_s*solid.dx 
            + u_s*psi_s*solid.dx 
            + 0.5*param.dt/param.K*a_s(u_s, v_s, phi_s, psi_s, 
                                       solid, param))
        L_s = (v_s_n_K*phi_s*solid.dx 
            + u_s_n_K*psi_s*solid.dx 
            - 0.5*param.dt/param.K*a_s(u_s_n_K, v_s_n_K, phi_s, psi_s, 
                                       solid, param) 
            + 0.5*param.dt/param.K*param.nu*dot(grad(v_s_i), 
                                                solid.n)*phi_s*solid.ds 
            + 0.5*param.dt/param.K*param.nu*dot(grad(v_s_n_i_K),
                                                solid.n)*phi_s*solid.ds) 
                
        # Solve solid problem
        U_s = Function(solid.V)
        solve(A_s == L_s, U_s, bcs_s)
        (u_s, v_s) = U_s.split(U_s)
                
        # Update solid solution
        u_s_n_K.assign(u_s)
        v_s_n_K.assign(v_s)
                
        # Update boundary condition
        v_s_n_i_K.assign(project(v_s_i, solid.V_v))
        
    # Assign final values
    u.s.assign(u_s)
    v.s.assign(v_s)
    
    return 
