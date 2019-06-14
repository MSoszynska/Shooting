from __future__ import print_function
from fenics import*
from parameters import*
import os
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import lgmres

def primal_shooting():
    
    print('Primal problem using shooting')
    
    # Set option to allow extrapolation
    parameters['allow_extrapolation'] = True

    # Define boundary
    def boundary(x, on_boundary):
        return on_boundary and (not near(x[1], 0.0))
    
    # Define interface
    class Interface(SubDomain):
        def inside(self, x, on_boundary):            
            return near(x[1], 0.0)
    interface = Interface()

    # Create meshes
    mesh_f = RectangleMesh(Point(0.0, 0.0), Point(4.0, 1.0), nx, ny)
    mesh_s = RectangleMesh(Point(0.0, -1.0), Point(4.0, 0.0), nx, ny)
    h_f = CellDiameter(mesh_f)
    h_s = CellDiameter(mesh_s)
    boundarymesh = BoundaryMesh(mesh_s, 'exterior')
    mesh_i = SubMesh(boundarymesh, interface)

    # Define measures
    dx_f = Measure('dx', domain = mesh_f)
    ds_f = Measure('ds', domain = mesh_f)
    dx_s = Measure('dx', domain = mesh_s)
    ds_s = Measure('ds', domain = mesh_s)
    dx_i = Measure('dx', domain = mesh_i)

    # Define function space
    W_f = FiniteElement('Lagrange', mesh_f.ufl_cell(), 1)
    V_f = FunctionSpace(mesh_f, W_f * W_f)
    W_s = FiniteElement('Lagrange', mesh_s.ufl_cell(), 1)
    V_s = FunctionSpace(mesh_s, W_s * W_s)
    W_i = FiniteElement('Lagrange', mesh_i.ufl_cell(), 1)
    V_i = FunctionSpace(mesh_i, W_i * W_i)

    # Define boundary values
    u_0 = Constant(0.0)
    v_0 = Constant(0.0)

    # Define initial boundary values
    u_f_n_i = project(u_0, V_f.sub(0).collapse())
    u_f_i = project(u_0, V_f.sub(0).collapse())
    v_f_n_i = project(v_0, V_f.sub(1).collapse())
    v_f_i = project(v_0, V_f.sub(1).collapse())
    v_s_n_i = project(v_0, V_s.sub(1).collapse())
    v_s_i = project(v_0, V_s.sub(1).collapse())

    # Define initial values for shooting method
    u_s_new = project(u_0, V_s.sub(0).collapse())
    v_s_new = project(v_0, V_s.sub(1).collapse())

    # Define right hand side
    f = Expression('exp(-10.0*(pow(x[0] - 0.5, 2) + pow(x[1] - \
                    0.5, 2)))*sign', sign = 1.0, degree = 1)
    N_external = int(1.0/(10.0*dt))

    # Define initial values for time loop
    u_f_n = project(u_0, V_f.sub(0).collapse())
    v_f_n = project(v_0, V_f.sub(1).collapse())
    u_s_n = project(u_0, V_s.sub(0).collapse())
    v_s_n = project(v_0, V_s.sub(1).collapse())

    # Find continuous gradient
    def gradient(u):
        dx_g = Measure('dx', domain = mesh_f)
        V_g = VectorFunctionSpace(mesh_f, 'Lagrange', 1)
        w = TrialFunction(V_g)
        v = TestFunction(V_g)
        A = inner(w, v)*dx_g
        L = inner(grad(u), v)*dx_g
        grad_u = Function(V_g)
        solve(A == L, grad_u)        
        return grad_u

    # Define variational forms
    def a_f(u_f, v_f, phi_f, psi_f):        
        return nu*dot(grad(v_f), grad(phi_f))*dx_f \
             + dot(beta, grad(v_f))*phi_f*dx_f \
             + dot(grad(u_f), grad(psi_f))*dx_f \
             - grad(u_f)[1]*psi_f*ds_f \
             - nu*grad(v_f)[1]*phi_f*ds_f \
             + gamma/h_f*u_f*psi_f*ds_f \
             + gamma/h_f*v_f*phi_f*ds_f
    def a_s(u_s, v_s, phi_s, psi_s):        
        return zeta*dot(grad(u_s), grad(phi_s))*dx_s \
             + delta*dot(grad(v_s), grad(phi_s))*dx_s \
             - v_s*psi_s*dx_s + \
             + delta*grad(v_s)[1]*phi_s*ds_s
        
    # Create directory
    directory = os.getcwd()
    os.makedirs('primal_shooting')
    os.chdir('primal_shooting')

    # Create pvd files
    u_f_pvd = File('solutions/fluid/displacement.pvd')
    v_f_pvd = File('solutions/fluid/velocity.pvd')
    u_s_pvd = File('solutions/solid/displacement.pvd')
    v_s_pvd = File('solutions/solid/velocity.pvd')

    # Create arrays 
    U_f_array = []
    U_f_array.append(project(u_0, V_f.sub(0).collapse()))
    V_f_array = []
    V_f_array.append(project(v_0, V_f.sub(1).collapse()))
    U_s_array = []
    U_s_array.append(project(u_0, V_s.sub(0).collapse()))
    V_s_array = []
    V_s_array.append(project(v_0, V_s.sub(1).collapse()))
    
    # Define fluid function
    def fluid(u_s, v_s, save):
        
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
            
        # Compute fractional steps for fluid problem
        for m in range(M):
                
            # Update boundary values
            u_f_i.assign(project((M - m - 1.0)/M*u_f_n_i + \
                                (m + 1.0)/M*u_s, V_f.sub(0).collapse()))
            v_f_i.assign(project((M - m - 1.0)/M*v_f_n_i + \
                                (m + 1.0)/M*v_s, V_f.sub(1).collapse()))
            
            # Define Dirichlet boundary conditions
            bc_u_f_0 = DirichletBC(V_f.sub(0), u_0, boundary)
            bc_v_f_0 = DirichletBC(V_f.sub(1), v_0, boundary)
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
                + dt/M*f*phi_f*dx_f 
                
            # Solve fluid problem
            U_f = Function(V_f)
            solve(A_f == L_f, U_f, bcs_f)
            (u_f, v_f) = U_f.split(U_f)
                
            if save == True:
                    
                # Save values
                U_f_array.append(u_f)
                V_f_array.append(v_f)
                    
                # Save solutions
                u_f.rename('Displacement', 'Fluid')
                u_f_pvd << u_f 
                v_f.rename('Velocity', 'Fluid')
                v_f_pvd << v_f
                    
            # Update fluid solution
            u_f_n_M.assign(u_f)
            v_f_n_M.assign(v_f)
                
            # Update boundary conditions
            u_f_n_i_M.assign(project(u_f_i, V_f.sub(0).collapse()))
            v_f_n_i_M.assign(project(v_f_i, V_f.sub(1).collapse()))
            
        return [u_f, v_f]
    
    # Define solid function
    def solid(v_f, save):
        
        # Store old solutions
        u_s_n_K = Function(V_s.sub(0).collapse())
        v_s_n_K = Function(V_s.sub(1).collapse())
        u_s_n_K.assign(u_s_n)
        v_s_n_K.assign(v_s_n)
        
        # Store old boundary values
        v_s_n_i_K = Function(V_s.sub(1).collapse())
        v_s_n_i_K.assign(v_s_n_i)
            
        # Compute fractional steps for solid problem
        for k in range(K):
            
            # Update boundary values
            v_s_i.assign(project((K - k - 1.0)/K*gradient(v_f_n)[1] + \
                                (k + 1.0)/K*gradient(v_f)[1], V_s.sub(1).collapse()))
                
            # Define Dirichlet boundary conditions
            bc_u_s_0 = DirichletBC(V_s.sub(0), u_0, boundary)
            bc_v_s_0 = DirichletBC(V_s.sub(1), v_0, boundary)
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
                + 0.5*dt/K*nu*v_s_i*phi_s*ds_s \
                + 0.5*dt/K*nu*v_s_n_i_K*phi_s*ds_s 
                
            # Solve solid problem
            U_s = Function(V_s)
            solve(A_s == L_s, U_s, bcs_s)
            (u_s, v_s) = U_s.split(U_s)
                
            if save == True:
                    
                # Save values
                U_s_array.append(u_s)
                V_s_array.append(v_s)
                    
                # Save solution
                u_s.rename('Displacement', 'Solid')
                u_s_pvd << u_s
                v_s.rename('Velocity', 'Solid')
                v_s_pvd << v_s   
                
            # Update solid solution
            u_s_n_K.assign(u_s)
            v_s_n_K.assign(v_s)
                
            # Update boundary condition
            v_s_n_i_K.assign(project(v_s_i, V_s.sub(1).collapse()))
            
        return [u_s, v_s]
    
    # Define shooting function
    def F(s, p):
        
        # Perform one iteration
        fluid_interface = fluid(s, p, False)
        solid_interface = solid(fluid_interface[1], False)
        
        # Define shooting function
        F_1 = project(interpolate(s, V_i.sub(0).collapse()) - \
              interpolate(solid_interface[0], V_i.sub(0).collapse()), \
              V_i.sub(0).collapse())
        F_2 = project(interpolate(p, V_i.sub(1).collapse()) - \
              interpolate(solid_interface[1], V_i.sub(1).collapse()), \
              V_i.sub(1).collapse())
        
        # Represent shooting function as an array
        F_1_array = F_1.vector().get_local()
        F_2_array = F_2.vector().get_local()
        F_array = np.concatenate((F_1_array, F_2_array), axis = None)
            
        return F_array
    
    # Define linear operator for linear solver in shooting method
    def shooting_newton(S, F_shooting):
        def shooting_gmres(D):
            
            # Define empty functions on interface
            U_eps = Function(V_i)
            (u_eps, v_eps) = U_eps.split(U_eps)
            
            # Split entrance vectors
            S_split = np.split(S, 2)
            D_split = np.split(D, 2)
            
            # Set values of functions on interface
            u_eps.vector().set_local(S_split[0] + eps*D_split[0])
            v_eps.vector().set_local(S_split[1] + eps*D_split[1])
            
            # Interpolate functions on solid subdomain
            u_eps_s = interpolate(u_eps, V_s.sub(0).collapse())
            v_eps_s = interpolate(v_eps, V_s.sub(1).collapse())
            
            # Compute shooting function
            F_eps = F(u_eps_s, v_eps_s)
            
            return (F_eps - F_shooting)/eps
        return shooting_gmres
    
    # Create time loop
    switch = True
    count = 0
    for i in range(N):
        
        # Update current time step and parameters
        print('Current time step: ', i + 1)
        j = 0
        num_iters = 0
        
        # Update external force
        if (count == N_external) and (switch == True):
            switch = False
            f.sign = 0.0
            print('No external force')
        if (i == 10*N_external):
            count = 0
            switch = True
            f.sign = 1.0
            print('External force')
        count += 1
        
        # Define initial values for Newton's method
        u_s_new.assign(u_s_n)
        v_s_new.assign(v_s_n)
        F_new = np.ones(2*nx + 2)
        
        # Define Newton's method
        while (j < maxiter_newton) and (np.linalg.norm(F_new) > tol_newton):
            
            j += 1
            num_iters += 1
            print('Current iteration of Newton\'s method: ', j)
            
            # Define right hand side
            F_new = F(u_s_new, v_s_new)
            
            # Define linear operator
            u_s_i_new = interpolate(u_s_new, V_i.sub(0).collapse())
            v_s_i_new = interpolate(v_s_new, V_i.sub(1).collapse())
            u_s_i_new_array = u_s_i_new.vector().get_local()
            v_s_i_new_array = v_s_i_new.vector().get_local()
            U_s_i_new_array = np.concatenate((u_s_i_new_array, v_s_i_new_array), axis = None)
            shooting_gmres_new = shooting_newton(U_s_i_new_array, F_new)
            shooting_new = LinearOperator((2*nx + 2, 2*nx + 2), matvec = shooting_gmres_new)
            
            # Solve linear system
            num_iters_gmres = 0
            res_norm_gmres = 0
            def callback(xk):
                nonlocal num_iters_gmres
                nonlocal res_norm_gmres
                num_iters_gmres += 1
                res_norm_gmres = np.linalg.norm(xk)
            D_new, exit_code = lgmres(shooting_new, -F_new, tol = tol_gmres, \
                                     maxiter = maxiter_gmres, callback = callback)
            num_iters += num_iters_gmres
            if (exit_code == 0):
                print('GMRES method converged successfully after', str(num_iters_gmres), 'iterations.')
            else:
                print('GMRES method failed to converge.')
                print('Norm of residual: ', str(res_norm_gmres))
                break
            
            # Advance solution
            D_new_split = np.split(D_new, 2) 
            u_s_i_new_array += D_new_split[0]
            v_s_i_new_array += D_new_split[1]
            u_s_i_new.vector().set_local(u_s_i_new_array)
            v_s_i_new.vector().set_local(v_s_i_new_array)
            u_s_new.assign(interpolate(u_s_i_new, V_s.sub(0).collapse()))
            v_s_new.assign(interpolate(v_s_i_new, V_s.sub(1).collapse()))
            
            # Check stop conditions
            if (np.linalg.norm(F_new) < tol_newton):
                print('Newton\'s method converged successfully after ', str(j), ' iterations.')
            elif (j == maxiter_newton):
                print('Newton\'s method failed to converge.')
                print('Norm of residual: ', str(np.linalg.norm(F_new)))
                
        # Solve fluid problem
        print('Fluid problem')
        fluid_interface = fluid(u_s_new, v_s_new, True)
        u_f = fluid_interface[0]
        v_f = fluid_interface[1]
            
        # Solve solid problem
        print('Solid problem')
        solid_interface = solid(v_f, True)
        u_s = solid_interface[0]
        v_s = solid_interface[1]
        
        # Save number of iterations
        num_iters += 1
        print('Number of solved linear systems: ', str(num_iters))
        iterations_txt = open('iterations.txt', 'a')
        iterations_txt.write(str(i + 1) + ' ' + str(num_iters) + ' ' + \
                             str(np.linalg.norm(F_new))+ '\r\n')
        iterations_txt.close()
            
        # Update solution
        u_f_n.assign(u_f)
        v_f_n.assign(v_f)
        u_s_n.assign(u_s)
        v_s_n.assign(v_s)
        
        # Update boundary conditions
        u_f_n_i.assign(u_f_i)
        v_f_n_i.assign(v_f_i)
        v_s_n_i.assign(v_s_i)
    
    os.chdir(directory)
    
    return [U_f_array, V_f_array, U_s_array, V_s_array]
        





