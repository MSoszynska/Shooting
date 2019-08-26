from fenics import (Function, FunctionSpace, dot, grad,
                    project, DirichletBC, Constant, 
                    TrialFunction, split, TestFunction, solve)
from spaces import boundary
from forms import a_s
from coupling import fluid_to_solid

# Define solid function
def solid_problem(u_f, v_f, u_s, v_s,
                  fluid, solid, param, save = False):

    # Store old solutions
    u_s_n_K = Function(solid.V_split[0])
    v_s_n_K = Function(solid.V_split[1])
    u_s_n_K.assign(u_s.old)
    v_s_n_K.assign(v_s.old)
        
    # Store old boundary values
    v_s_n_i_K = Function(solid.V_split[1])
    v_s_n_i_K.assign(v_s.i_old)
    
    # Initialize interface values
    v_s_i = Function(solid.V_split[1])
    
    # Define Dirichlet boundary conditions
    bc_u_s_0 = DirichletBC(solid.V.sub(0), Constant(0.0), boundary)
    bc_v_s_0 = DirichletBC(solid.V.sub(1), Constant(0.0), boundary)
    bcs_s = [bc_u_s_0, bc_v_s_0]
            
    # Compute fractional steps for solid problem
    for k in range(param.K):
            
        # Update boundary values
        v_s_i.assign(project((param.K - k - 1.0)/param.K*v_s.i_old +
              + (k + 1.0)/param.K*fluid_to_solid(v_f.new,
              fluid, solid, param, 1), solid.V_split[1]))
                
        # Define trial and test functions
        U_s_new = TrialFunction(solid.V)
        (u_s_new, v_s_new) = split(U_s_new)
        Phi_s = TestFunction(solid.V)
        (phi_s, psi_s) = split(Phi_s)
                
        # Define scheme
        A_s = (v_s_new*phi_s*solid.dx
            + u_s_new*psi_s*solid.dx
            + 0.5*param.dt/param.K*a_s(u_s_new, v_s_new, phi_s, psi_s,
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
        U_s_new = Function(solid.V)
        solve(A_s == L_s, U_s_new, bcs_s)
        (u_s_new, v_s_new) = U_s_new.split(U_s_new)
        
        # Append solutions to the arrays
        if save:
            
            u_s.array.append(u_s_new.copy(deepcopy = True))
            v_s.array.append(v_s_new.copy(deepcopy = True))
            
        # Update solid solution
        u_s_n_K.assign(u_s_new)
        v_s_n_K.assign(v_s_new)
                
        # Update boundary condition
        v_s_n_i_K.assign(project(v_s_i, solid.V_split[1]))
        
    # Save final values
    u_s.new.assign(u_s_new)
    v_s.new.assign(v_s_new)
    v_s.i_new.assign(v_s_i)
    
    return 
