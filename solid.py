from __future__ import print_function
from fenics import*
from parameters import*
from spaces import*
from coupling import neumann    
    
# Define solid function
def solid(v_f, u_s_n, v_s_n, v_s_n_i):
    
    # Define variational form
    def a_s(u_s, v_s, phi_s, psi_s):
        return zeta*dot(grad(u_s), grad(phi_s))*dx_s \
             + delta*dot(grad(v_s), grad(phi_s))*dx_s \
             - v_s*psi_s*dx_s \
             + delta*grad(v_s)[1]*phi_s*ds_s
        
    # Store old solutions
    u_s_n_K = Function(V_s.sub(0).collapse())
    v_s_n_K = Function(V_s.sub(1).collapse())
    u_s_n_K.assign(u_s_n)
    v_s_n_K.assign(v_s_n)
        
    # Store old boundary values
    v_s_n_i_K = Function(V_s.sub(1).collapse())
    v_s_n_i_K.assign(v_s_n_i)
    
    # Initialize interface values
    v_s_i = Function(V_s.sub(1).collapse())
            
    # Compute fractional steps for solid problem
    for k in range(K):
            
        # Update boundary values
        v_s_i.assign(project((K - k - 1.0)/K*v_s_n_i + \
              (k + 1.0)/K*neumann(v_f, V_f.sub(1).collapse(), \
              V_s.sub(1).collapse()), V_s.sub(1).collapse()))
                
        # Define Dirichlet boundary conditions
        bc_u_s_0 = DirichletBC(V_s.sub(0), Constant(0.0), boundary)
        bc_v_s_0 = DirichletBC(V_s.sub(1), Constant(0.0), boundary)
        bcs_s = [bc_u_s_0, bc_v_s_0]
                
        # Define trial and test functions
        U_s = TrialFunction(V_s)
        (u_s, v_s) = split(U_s)
        Phi_s = TestFunction(V_s)
        (phi_s, psi_s) = split(Phi_s)
                
        # Define scheme
        A_s = v_s*phi_s*dx_s \
            + u_s*psi_s*dx_s \
            + 0.5*dt/K*a_s(u_s, v_s, phi_s, psi_s)
        L_s = v_s_n_K*phi_s*dx_s \
            + u_s_n_K*psi_s*dx_s \
            - 0.5*dt/K*a_s(u_s_n_K, v_s_n_K, phi_s, psi_s) \
            + 0.5*dt/K*nu*dot(grad(v_s_i), n_s)*phi_s*ds_s \
            + 0.5*dt/K*nu*dot(grad(v_s_n_i_K), n_s)*phi_s*ds_s 
                
        # Solve solid problem
        U_s = Function(V_s)
        solve(A_s == L_s, U_s, bcs_s)
        (u_s, v_s) = U_s.split(U_s)
                
        # Update solid solution
        u_s_n_K.assign(u_s)
        v_s_n_K.assign(v_s)
                
        # Update boundary condition
        v_s_n_i_K.assign(project(v_s_i, V_s.sub(1).collapse()))
            
    return [u_s, v_s]
    
