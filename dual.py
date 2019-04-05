from __future__ import print_function
from fenics import*
from parameters import*
import os
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import lgmres

def dual():
    
    print('Dual problem')
    
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
    mesh_f = RectangleMesh(Point(0.0, 0.0), Point(4.0, 1.0), 80, 20)
    mesh_s = RectangleMesh(Point(0.0, -1.0), Point(4.0, 0.0), 80, 20)
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
    z_0 = Constant(0.0)
    y_0 = Constant(0.0)
    
    # Define initial boundary values
    z_f_n_i = project(z_0, V_f.sub(0).collapse())
    z_f_i = project(z_0, V_f.sub(0).collapse())
    z_s_n_i = project(z_0, V_s.sub(0).collapse())
    z_s_i = project(z_0, V_s.sub(0).collapse())
    y_s_n_i = project(y_0, V_s.sub(1).collapse())
    y_s_i = project(y_0, V_s.sub(1).collapse())

    # Define initial values for shooting method
    z_s_new = project(z_0, V_s.sub(0).collapse())
    
    # Define initial values for time loop
    z_f_n = project(z_0, V_f.sub(0).collapse())
    y_f_n = project(y_0, V_f.sub(1).collapse())
    z_s_n = project(z_0, V_s.sub(0).collapse())
    y_s_n = project(y_0, V_s.sub(1).collapse())
    
    # Define characteristic function
    char_func = Expression('2.0 <= x[0] && x[0] <= 4.0 ? 1.0 : 0.0', \
                           degree = 0)

    # Define variational forms
    def a_f(xi_f, eta_f, z_f, y_f):
        return nu*dot(grad(eta_f), grad(z_f))*dx_f \
             + dot(beta, grad(eta_f))*z_f*dx_f \
             + dot(grad(xi_f), grad(y_f))*dx_f \
             - grad(xi_f)[1]*y_f*ds_f \
             - nu*grad(eta_f)[1]*z_f*ds_f \
             + gamma/h_f*xi_f*y_f*ds_f \
             + gamma/h_f*eta_f*z_f*ds_f
    def a_s(xi_s, eta_s, z_s, y_s):
        return zeta*dot(grad(xi_s), grad(z_s))*dx_s \
             + delta*dot(grad(eta_s), grad(z_s))*dx_s \
             - eta_s*y_s*dx_s
        
    # Create directory
    directory = os.getcwd()
    os.makedirs('dual')
    os.chdir('dual')

    # Create pvd files
    z_f_pvd = File('solutions/fluid/displacement.pvd')
    y_f_pvd = File('solutions/fluid/velocity.pvd')
    z_s_pvd = File('solutions/solid/displacement.pvd')
    y_s_pvd = File('solutions/solid/velocity.pvd')

    # Create arrays 
    Z_f_array = []
    Y_f_array = []
    Z_s_array = []
    Y_s_array = []
    
    # Define fluid function
    def fluid(z_s, save):
        
        # Store old solutions
        z_f_n_M = Function(V_f.sub(0).collapse())
        y_f_n_M = Function(V_f.sub(1).collapse())
        z_f_n_M.assign(z_f_n)
        y_f_n_M.assign(y_f_n)
        
        # Store old boundary values
        z_f_n_i_M = Function(V_f.sub(0).collapse())
        z_f_n_i_M.assign(z_f_n_i)
        
        # Compute fractional steps for fluid problem
        for m in range(M):
                
            # Update boundary values
            z_f_i.assign(project((M - m - 1.0)/M*z_f_n_i + \
                                (m + 1.0)/M*z_s, V_f.sub(0).collapse()))
                
            # Define Dirichlet boundary conditions
            bc_z_f_0 = DirichletBC(V_f.sub(0), z_0, boundary)
            bc_y_f_0 = DirichletBC(V_f.sub(1), y_0, boundary)
            bcs_f = [bc_z_f_0, bc_y_f_0]
                
            # Define trial and test functions 
            Z_f = TrialFunction(V_f)
            (z_f, y_f) = split(Z_f)
            Xi_f = TestFunction(V_f)
            (xi_f, eta_f) = split(Xi_f)
                
            # Define scheme
            if (i == 0) and (m == 0):
                A_f = eta_f*z_f*dx_f \
                    + 0.5*dt/M*a_f(xi_f, eta_f, z_f, y_f)
                L_f = 0.5*dt/M*nu*grad(eta_f)[1]*z_f_i*ds_f \
                    + 0.5*dt/M*char_func*eta_f*dx_f 
            else:
                A_f = eta_f*z_f*dx_f \
                    + 0.5*dt/M*a_f(xi_f, eta_f, z_f, y_f)
                L_f = eta_f*z_f_n_M*dx_f \
                    - 0.5*dt/M*a_f(xi_f, eta_f, z_f_n_M, y_f_n_M) \
                    + 0.5*dt/M*nu*grad(eta_f)[1]*z_f_i*ds_f \
                    + 0.5*dt/M*nu*grad(eta_f)[1]*z_f_n_i_M*ds_f \
                    + dt/M*char_func*eta_f*dx_f 
                
            # Solve fluid problem
            Z_f = Function(V_f)
            solve(A_f == L_f, Z_f, bcs_f)
            (z_f, y_f) = Z_f.split(Z_f)
                
            if save == True:
                    
                # Save values
                Z_f_array.append(z_f)
                Y_f_array.append(y_f)
                    
                # Save solutions
                z_f.rename('Displacement', 'Fluid')
                z_f_pvd << z_f 
                y_f.rename('Velocity', 'Fluid')
                y_f_pvd << y_f 
                    
            # Update fluid solution
            z_f_n_M.assign(z_f)
            y_f_n_M.assign(y_f)
                
            # Update boundary conditions
            z_f_n_i_M.assign(project(z_f_i, V_f.sub(0).collapse()))
            
        return [z_f, y_f]
    
    # Define solid function
    def solid(z_f, y_f, save):
        
        # Store old solutions
        z_s_n_K = Function(V_s.sub(0).collapse())
        y_s_n_K = Function(V_s.sub(1).collapse())
        z_s_n_K.assign(z_s_n)
        y_s_n_K.assign(y_s_n)
        
        # Store old boundary values
        z_s_n_i_K = Function(V_s.sub(0).collapse())
        y_s_n_i_K = Function(V_s.sub(1).collapse())
        z_s_n_i_K.assign(z_s_n_i)
        y_s_n_i_K.assign(y_s_n_i)
        
        # Compute fractional steps for solid problem
        for k in range(K):
            
            # Update boundary values
            z_s_i.assign(project((K - k - 1.0)/K*z_f_n + \
                                (k + 1.0)/K*z_f, V_s.sub(0).collapse()))
            y_s_i.assign(project((K - k - 1.0)/K*y_f_n + \
                                (k + 1.0)/K*y_f, V_s.sub(1).collapse()))
                
            # Define Dirichlet boundary conditions
            bc_z_s_0 = DirichletBC(V_s.sub(0), z_0, boundary)
            bc_y_s_0 = DirichletBC(V_s.sub(1), y_0, boundary)
            bcs_s = [bc_z_s_0, bc_y_s_0]
                
            # Define trial and test functions
            Z_s = TrialFunction(V_s)
            (z_s, y_s) = split(Z_s)
            Xi_s = TestFunction(V_s)
            (xi_s, eta_s) = split(Xi_s)
                
            # Define scheme
            if (i == 0) and (k == 0):
                A_s = eta_s*z_s*dx_s \
                    + xi_s*y_s*dx_s \
                    + 0.5*dt/K*a_s(xi_s, eta_s, z_s, y_s)
                L_s = 0.5*dt/K*gamma/h_s*xi_s*y_s_i*ds_s \
                    + 0.5*dt/K*gamma/h_s*eta_s*z_s_i*ds_s 
            else:
                A_s = eta_s*z_s*dx_s \
                    + xi_s*y_s*dx_s \
                    + 0.5*dt/K*a_s(xi_s, eta_s, z_s, y_s)
                L_s = eta_s*z_s_n_K*dx_s \
                    + xi_s*y_s_n_K*dx_s \
                    - 0.5*dt/K*a_s(xi_s, eta_s, z_s_n_K, y_s_n_K) \
                    + 0.5*dt/K*gamma/h_s*xi_s*y_s_i*ds_s \
                    + 0.5*dt/K*gamma/h_s*eta_s*z_s_i*ds_s \
                    + 0.5*dt/K*gamma/h_s*xi_s*y_s_n_i_K*ds_s \
                    + 0.5*dt/K*gamma/h_s*eta_s*z_s_n_i_K*ds_s 
                
            # Solve solid problem
            Z_s = Function(V_s)
            solve(A_s == L_s, Z_s, bcs_s)
            (z_s, y_s) = Z_s.split(Z_s)
                
            if save == True:
                    
                # Save values
                Z_s_array.append(z_s)
                Y_s_array.append(y_s)
                    
                # Save solution
                z_s.rename('Displacement', 'Solid')
                z_s_pvd << z_s
                y_s.rename('Velocity', 'Solid')
                y_s_pvd << y_s   
                
            # Update solid solution
            z_s_n_K.assign(z_s)
            y_s_n_K.assign(y_s)
                
            # Update boundary condition
            z_s_n_i_K.assign(project(z_s_i, V_s.sub(0).collapse()))
            y_s_n_i_K.assign(project(y_s_i, V_s.sub(1).collapse()))
        
        return [z_s, y_s]
    
    # Define shooting function
    def F(s):
        
        # Perform one iteration
        fluid_interface = fluid(s, False)
        solid_interface = solid(fluid_interface[0], fluid_interface[1], False)
        
        # Define shooting function
        F = project(interpolate(s, V_i.sub(0).collapse()) - \
            interpolate(solid_interface[0], V_i.sub(0).collapse()), \
            V_i.sub(0).collapse())
        
        # Represent shooting function as an array
        F_array = F.vector().get_local()
            
        return F_array
    
    # Define linear operator for linear solver in shooting method
    def shooting_newton(S, F_shooting):
        def shooting_gmres(D):
            
            # Define empty functions on interface
            u_eps = Function(V_i.sub(0).collapse())
            
            # Set values of functions on interface
            u_eps.vector().set_local(S + eps*D)
            
            # Interpolate functions on solid subdomain
            u_eps_s = interpolate(u_eps, V_s.sub(0).collapse())
            
            # Compute shooting function
            F_eps = F(u_eps_s)
            
            return (F_eps - F_shooting)/eps
        return shooting_gmres

    # Create time loop
    for i in range(N):
        
        # Update current time step and parameters
        print('Current time step: ', N - i)
        j = 0
        num_iters = 0
        
        # Define initial values for Newton's method
        z_s_new.assign(z_s_n)
        F_new = np.ones(nx + 1)
        
        # Define Newton's method
        while (j < maxiter_newton) and (np.linalg.norm(F_new) > tol_newton):
            
            j += 1
            num_iters += 1
            print('Current iteration of Newton\'s method: ', j)
            
            # Define right hand side
            F_new = F(z_s_new)
            
            # Define linear operator
            z_s_i_new = interpolate(z_s_new, V_i.sub(0).collapse())
            z_s_i_new_array = z_s_i_new.vector().get_local()
            shooting_gmres_new = shooting_newton(z_s_i_new_array, F_new)
            shooting_new = LinearOperator((nx + 1, nx + 1), matvec = shooting_gmres_new)
            
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
            
            # Advance solution
            z_s_i_new_array += D_new
            z_s_i_new.vector().set_local(z_s_i_new_array)
            z_s_new.assign(interpolate(z_s_i_new, V_s.sub(0).collapse()))
            
            # Check stop conditions
            if (np.linalg.norm(F_new) < tol_newton):
                print('Newton\'s method converged successfully after ', str(j), ' iterations.')
            elif (j == maxiter_newton):
                print('Newton\'s method failed to converge.')
                print('Norm of residual: ', str(np.linalg.norm(F_new)))
                
        # Solve fluid problem
        print('Fluid problem')
        fluid_interface = fluid(z_s_new, True)
        z_f = fluid_interface[0]
        y_f = fluid_interface[1]
            
        # Solve solid problem
        print('Solid problem')
        solid_interface = solid(z_f, y_f, True)
        z_s = solid_interface[0]
        y_s = solid_interface[1]
        
        # Save number of iterations
        num_iters += 1
        print('Number of solved linear systems: ', str(num_iters))
        iterations_txt = open('iterations.txt', 'a')
        iterations_txt.write(str(i + 1) + ' ' + str(num_iters) + ' ' + \
                             str(np.linalg.norm(F_new)) + '\r\n')
        iterations_txt.close()
            
        # Update solution
        z_f_n.assign(z_f)
        y_f_n.assign(y_f)
        z_s_n.assign(z_s)
        y_s_n.assign(y_s)
        
        # Update boundary conditions
        z_f_n_i.assign(z_f_i)
        z_s_n_i.assign(z_s_i)
        y_s_n_i.assign(y_s_i)
    
    # Add initial value
    Z_f_array.append(project(z_0, V_f.sub(0).collapse()))
    Y_f_array.append(project(y_0, V_f.sub(1).collapse()))
    Z_s_array.append(project(z_0, V_s.sub(0).collapse()))
    Y_s_array.append(project(y_0, V_s.sub(1).collapse()))
    
    os.chdir(directory)
    return [Z_f_array, Y_f_array, Z_s_array, Y_s_array]
