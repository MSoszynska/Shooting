from fenics import (Function, FunctionSpace, project, 
                    DirichletBC, Constant, TrialFunction, split, 
                    TestFunction, solve)
from spaces import boundary
from parameters import f
from forms import a_f
from coupling import solid_to_fluid


# Define fluid function
def fluid_problem(u_f, v_f, u_s, v_s,
                  fluid, solid, param, t, save = False):
        
    # Store old solutions
    u_f_n_M = Function(fluid.V_split[0])
    v_f_n_M = Function(fluid.V_split[1])
    u_f_n_M.assign(u_f.old)
    v_f_n_M.assign(v_f.old)
        
    # Store old boundary values
    u_f_n_i_M = Function(fluid.V_split[0])
    v_f_n_i_M = Function(fluid.V_split[1])
    u_f_n_i_M.assign(u_f.i_old)
    v_f_n_i_M.assign(v_f.i_old)
    
    # Initialize interface values
    u_f_i = Function(fluid.V_split[0])
    v_f_i = Function(fluid.V_split[1])
    
    # Define Dirichlet boundary conditions
    bc_u_f_0 = DirichletBC(fluid.V.sub(0), Constant(0.0), boundary)
    bc_v_f_0 = DirichletBC(fluid.V.sub(1), Constant(0.0), boundary)
    bcs_f = [bc_u_f_0, bc_v_f_0]
            
    # Compute fractional steps for fluid problem
    for m in range(param.M):
                
        # Update boundary values
        u_f_i.assign(project((param.M - m - 1.0)/param.M*u_f.i_old
              + (m + 1.0)/param.M*solid_to_fluid(u_s.new,
              fluid, solid, param, 0), fluid.V_split[0]))
        v_f_i.assign(project((param.M - m - 1.0)/param.M*v_f.i_old
              + (m + 1.0)/param.M*solid_to_fluid(v_s.new,
              fluid, solid, param, 1), fluid.V_split[1]))
                
        # Define trial and test functions 
        U_f_new = TrialFunction(fluid.V)
        (u_f_new, v_f_new) = split(U_f_new)
        Phi_f = TestFunction(fluid.V)
        (phi_f, psi_f) = split(Phi_f)
                
        # Define scheme
        A_f = (v_f_new*phi_f*fluid.dx
            + 0.5*param.dt/param.M*a_f(u_f_new, v_f_new, phi_f, psi_f,
                                       fluid, param))
        L_f = (v_f_n_M*phi_f*fluid.dx 
            - 0.5*param.dt/param.M*a_f(u_f_n_M, v_f_n_M, phi_f, psi_f, 
                                       fluid, param) 
            + 0.5*param.dt/param.M*param.gamma/fluid.h*u_f_i*psi_f
                                                            *fluid.ds 
            + 0.5*param.dt/param.M*param.gamma/fluid.h*v_f_i*phi_f 
                                                            *fluid.ds 
            + 0.5*param.dt/param.M*param.gamma/fluid.h*u_f_n_i_M*psi_f 
                                                      *fluid.ds 
            + 0.5*param.dt/param.M*param.gamma/fluid.h*v_f_n_i_M*phi_f 
                                                      *fluid.ds 
            + param.dt/param.M*f(t)*phi_f*fluid.dx) 
                
        # Solve fluid problem
        U_f_new = Function(fluid.V)
        solve(A_f == L_f, U_f_new, bcs_f)
        (u_f_new, v_f_new) = U_f_new.split(U_f_new)
        
        # Append solutions to the arrays
        if save:
            
            u_f.array.append(u_f_new.copy(deepcopy = True))
            v_f.array.append(v_f_new.copy(deepcopy = True))
                    
        # Update fluid solution
        u_f_n_M.assign(u_f_new)
        v_f_n_M.assign(v_f_new)
                
        # Update boundary conditions
        u_f_n_i_M.assign(project(u_f_i, fluid.V_split[0]))
        v_f_n_i_M.assign(project(v_f_i, fluid.V_split[1]))
        
    # Save final values
    u_f.new.assign(u_f_new)
    v_f.new.assign(v_f_new)
    u_f.i_new.assign(u_f_i)
    v_f.i_new.assign(v_f_i)
    
    return 
