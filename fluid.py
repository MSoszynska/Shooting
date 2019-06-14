from __future__ import print_function
from fenics import*
from parameters import*
from spaces import*
from coupling import dirichlet

# Define fluid function
def fluid(u_s, v_s, u_f_n, v_f_n, u_f_n_i, v_f_n_i, t):
    
    # Define variational form
    def a_f(u_f, v_f, phi_f, psi_f):
        return nu*dot(grad(v_f), grad(phi_f))*dx_f \
             + dot(beta, grad(v_f))*phi_f*dx_f \
             + dot(grad(u_f), grad(psi_f))*dx_f \
             - grad(u_f)[1]*psi_f*ds_f \
             - nu*grad(v_f)[1]*phi_f*ds_f \
             + gamma/h_f*u_f*psi_f*ds_f \
             + gamma/h_f*v_f*phi_f*ds_f
        
    # Store old solutions
    u_f_n_M = Function(V_f.sub(0).collapse())
    v_f_n_M = Function(V_f.sub(1).collapse())
    u_f_n_M.assign(u_f_n)
    v_f_n_M.assign(v_f_n)
        
    # Store old boundary values
    u_f_n_i_M = Function(V_f.sub(0).collapse())
    v_f_n_i_M = Function(V_f.sub(1).collapse())
    u_f_n_i_M.assign(u_f_n_i)
    v_f_n_i_M.assign(v_f_n_i)
    
    # Initialize interface values
    u_f_i = Function(V_f.sub(0).collapse())
    v_f_i = Function(V_f.sub(1).collapse())
            
    # Compute fractional steps for fluid problem
    for m in range(M):
                
        # Update boundary values
        u_f_i.assign(project((M - m - 1.0)/M*u_f_n_i + \
              (m + 1.0)/M*dirichlet(u_s, V_s.sub(0).collapse(), \
              V_f.sub(0).collapse()), V_f.sub(0).collapse()))
        v_f_i.assign(project((M - m - 1.0)/M*v_f_n_i + \
              (m + 1.0)/M*dirichlet(v_s, V_s.sub(1).collapse(), \
              V_f.sub(1).collapse()), V_f.sub(1).collapse()))
            
        # Define Dirichlet boundary conditions
        bc_u_f_0 = DirichletBC(V_f.sub(0), Constant(0.0), boundary)
        bc_v_f_0 = DirichletBC(V_f.sub(1), Constant(0.0), boundary)
        bcs_f = [bc_u_f_0, bc_v_f_0]
                
        # Define trial and test functions 
        U_f = TrialFunction(V_f)
        (u_f, v_f) = split(U_f)
        Phi_f = TestFunction(V_f)
        (phi_f, psi_f) = split(Phi_f)
                
        # Define scheme
        A_f = v_f*phi_f*dx_f \
            + 0.5*dt/M*a_f(u_f, v_f, phi_f, psi_f)
        L_f = v_f_n_M*phi_f*dx_f \
            - 0.5*dt/M*a_f(u_f_n_M, v_f_n_M, phi_f, psi_f) \
            + 0.5*dt/M*gamma/h_f*u_f_i*psi_f*ds_f \
            + 0.5*dt/M*gamma/h_f*v_f_i*phi_f*ds_f \
            + 0.5*dt/M*gamma/h_f*u_f_n_i_M*psi_f*ds_f \
            + 0.5*dt/M*gamma/h_f*v_f_n_i_M*phi_f*ds_f \
            + dt/M*f(t)*phi_f*dx_f 
                
        # Solve fluid problem
        U_f = Function(V_f)
        solve(A_f == L_f, U_f, bcs_f)
        (u_f, v_f) = U_f.split(U_f)
                    
        # Update fluid solution
        u_f_n_M.assign(u_f)
        v_f_n_M.assign(v_f)
                
        # Update boundary conditions
        u_f_n_i_M.assign(project(u_f_i, V_f.sub(0).collapse()))
        v_f_n_i_M.assign(project(v_f_i, V_f.sub(1).collapse()))
            
    return [u_f, v_f]
