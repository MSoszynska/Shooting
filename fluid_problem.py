from fenics import dot, grad, Measure, Function, \
                   FunctionSpace, Constant, project, \
                   DirichletBC, TrialFunction, \
                   TestFunction, split, solve, \
                   VectorFunctionSpace, inner
from spaces import boundary
from parameters import f
from coupling import flip

# Define fluid function
def fluid_problem(u, v, fluid, solid, interface, param, t):
    
    # Define variational form
    def a_f(u_f, v_f, phi_f, psi_f):
        return param.nu*dot(grad(v_f), grad(phi_f))*fluid.dx \
             + dot(param.beta, grad(v_f))*phi_f*fluid.dx \
             + dot(grad(u_f), grad(psi_f))*fluid.dx \
             - grad(u_f)[1]*psi_f*fluid.ds \
             - param.nu*grad(v_f)[1]*phi_f*fluid.ds \
             + param.gamma/fluid.h*u_f*psi_f*fluid.ds \
             + param.gamma/fluid.h*v_f*phi_f*fluid.ds
        
    # Store old solutions
    u_f_n_M = Function(fluid.V.sub(0).collapse())
    v_f_n_M = Function(fluid.V.sub(1).collapse())
    u_f_n_M.assign(u.f_n)
    v_f_n_M.assign(v.f_n)
        
    # Store old boundary values
    u_f_n_i_M = Function(fluid.V.sub(0).collapse())
    v_f_n_i_M = Function(fluid.V.sub(1).collapse())
    u_f_n_i_M.assign(u.f_n_i)
    v_f_n_i_M.assign(v.f_n_i)
    
    # Initialize interface values
    u_f_i = Function(fluid.V.sub(0).collapse())
    v_f_i = Function(fluid.V.sub(1).collapse())
            
    # Compute fractional steps for fluid problem
    for m in range(param.M):
                
        # Update boundary values
        u_f_i.assign(project((param.M - m - 1.0)/param.M*u.f_n_i + \
              (m + 1.0)/param.M*flip(u.s, \
              fluid.V.sub(0).collapse(), param), fluid.V.sub(0).collapse()))
        v_f_i.assign(project((param.M - m - 1.0)/param.M*v.f_n_i + \
              (m + 1.0)/param.M*flip(v.s, \
              fluid.V.sub(1).collapse(), param), fluid.V.sub(1).collapse()))

            
        # Define Dirichlet boundary conditions
        bc_u_f_0 = DirichletBC(fluid.V.sub(0), Constant(0.0), boundary)
        bc_v_f_0 = DirichletBC(fluid.V.sub(1), Constant(0.0), boundary)
        bcs_f = [bc_u_f_0, bc_v_f_0]
                
        # Define trial and test functions 
        U_f = TrialFunction(fluid.V)
        (u_f, v_f) = split(U_f)
        Phi_f = TestFunction(fluid.V)
        (phi_f, psi_f) = split(Phi_f)
                
        # Define scheme
        A_f = v_f*phi_f*fluid.dx \
            + 0.5*param.dt/param.M*a_f(u_f, v_f, phi_f, psi_f)
        L_f = v_f_n_M*phi_f*fluid.dx \
            - 0.5*param.dt/param.M*a_f(u_f_n_M, v_f_n_M, phi_f, psi_f) \
            + 0.5*param.dt/param.M*param.gamma/fluid.h*u_f_i*psi_f\
                                                            *fluid.ds \
            + 0.5*param.dt/param.M*param.gamma/fluid.h*v_f_i*phi_f \
                                                            *fluid.ds \
            + 0.5*param.dt/param.M*param.gamma/fluid.h*u_f_n_i_M*psi_f \
                                                      *fluid.ds \
            + 0.5*param.dt/param.M*param.gamma/fluid.h*v_f_n_i_M*phi_f \
                                                      *fluid.ds \
            + param.dt/param.M*f(t)*phi_f*fluid.dx 
                
        # Solve fluid problem
        U_f = Function(fluid.V)
        solve(A_f == L_f, U_f, bcs_f)
        (u_f, v_f) = U_f.split(U_f)
                    
        # Update fluid solution
        u_f_n_M.assign(u_f)
        v_f_n_M.assign(v_f)
                
        # Update boundary conditions
        u_f_n_i_M.assign(project(u_f_i, fluid.V.sub(0).collapse()))
        v_f_n_i_M.assign(project(v_f_i, fluid.V.sub(1).collapse()))
        
    # Save final values
    u.f.assign(u_f)
    v.f.assign(v_f)
    
    return 
    
