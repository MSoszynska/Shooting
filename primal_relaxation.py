from __future__ import print_function
from fenics import*
from parameters import*
import os
import numpy as np

def primal_relaxation():
    
    print('Primal problem using relaxation')
    
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
    u_f_n_i_M = project(u_0, V_f.sub(0).collapse())
    v_f_n_i_M = project(v_0, V_f.sub(1).collapse())
    v_s_n_i_K = project(v_0, V_s.sub(1).collapse())

    # Define initial values for relaxation
    u_s_old = project(u_0, V_s.sub(0).collapse())
    v_s_old = project(v_0, V_s.sub(1).collapse())
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
    u_s_n_K = project(u_0, V_s.sub(0).collapse())
    u_f_n_M = project(u_0, V_f.sub(0).collapse())
    v_f_n_M = project(v_0, V_f.sub(1).collapse())
    v_s_n_K = project(v_0, V_s.sub(1).collapse())

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
             - v_s*psi_s*dx_s
        
    # Create directory
    directory = os.getcwd()
    os.makedirs('primal_relaxation')
    os.chdir('primal_relaxation')

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
    
    # Create time loop
    switch = True
    count = 0
    for i in range(N):
        
        # Update current time step and parameters
        print('Current time step: ', i + 1)
        stop = False
        save = False
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
        
        # Compute solution
        while save == False:
            
            j += 1
            num_iters += 1
            print('Current iteration: ', j)
            
            # Store old solutions
            u_f_n_M.assign(u_f_n)
            v_f_n_M.assign(v_f_n)
            u_s_n_K.assign(u_s_n)
            v_s_n_K.assign(v_s_n)
            
            # Store old boundary values
            u_f_n_i_M.assign(u_f_n_i)
            v_f_n_i_M.assign(v_f_n_i)
            v_s_n_i_K.assign(v_s_n_i)
            
            print('Fluid problem')
            
            # Compute fractional steps for fluid problem
            for m in range(M):
                
                # Update boundary values
                u_f_i.assign(project((M - m - 1.0)/M*u_f_n_i + \
                                    (m + 1.0)/M*u_s_new, V_f.sub(0).collapse()))
                v_f_i.assign(project((M - m - 1.0)/M*v_f_n_i + \
                                    (m + 1.0)/M*v_s_new, V_f.sub(1).collapse()))
            
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
                L_f = v_f_n*phi_f*dx_f \
                    - 0.5*dt/M*a_f(u_f_n, v_f_n, phi_f, psi_f) \
                    + 0.5*dt/M*gamma/h_f*u_f_i*psi_f*ds_f \
                    + 0.5*dt/M*gamma/h_f*v_f_i*phi_f*ds_f \
                    + 0.5*dt/M*gamma/h_f*u_f_n_i*psi_f*ds_f \
                    + 0.5*dt/M*gamma/h_f*v_f_n_i*phi_f*ds_f \
                    + dt/M*f*phi_f*dx_f
                
                # Solve fluid problem
                U_f = Function(V_f)
                solve(A_f == L_f, U_f, bcs_f)
                (u_f, v_f) = U_f.split(U_f)
                
                if stop == True:
                    
                    # Save values
                    U_f_array.append(u_f)
                    V_f_array.append(v_f)
                    
                    # Save solutions
                    u_f.rename('Displacement', 'Fluid')
                    u_f_pvd << u_f 
                    v_f.rename('Velocity', 'Fluid')
                    v_f_pvd << v_f
                    
                    save = True
                    
                # Update fluid solution
                u_f_n.assign(u_f)
                v_f_n.assign(v_f)
                
                # Update boundary conditions
                u_f_n_i.assign(project(u_f_i, V_f.sub(0).collapse()))
                v_f_n_i.assign(project(v_f_i, V_f.sub(1).collapse()))
            
            print('Solid problem')
            
            # Compute fractional steps for solid problem
            for k in range(K):
            
                # Update boundary values
                v_s_i.assign(project((K - k - 1.0)/K*gradient(v_f_n_M)[1] + \
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
                L_s = v_s_n*phi_s*dx_s \
                    + u_s_n*psi_s*dx_s \
                    - 0.5*dt/K*a_s(u_s_n, v_s_n, phi_s, psi_s) \
                    + 0.5*dt/K*nu*v_s_i*phi_s*ds_s \
                    + 0.5*dt/K*nu*v_s_n_i*phi_s*ds_s 
                
                # Solve solid problem
                U_s = Function(V_s)
                solve(A_s == L_s, U_s, bcs_s)
                (u_s, v_s) = U_s.split(U_s)
                
                if stop == True:
                    
                    # Save values
                    U_s_array.append(u_s)
                    V_s_array.append(v_s)
                    
                    # Save solution
                    u_s.rename('Displacement', 'Solid')
                    u_s_pvd << u_s
                    v_s.rename('Velocity', 'Solid')
                    v_s_pvd << v_s   
                
                # Update solid solution
                u_s_n.assign(u_s)
                v_s_n.assign(v_s)
                
                # Update boundary condition
                v_s_n_i.assign(project(v_s_i, V_s.sub(1).collapse()))
                    
            # Define relaxation
            u_s_new.assign(project(tau*u_s + (1.0 - tau)*u_s_old, V_s.sub(0).collapse()))
            v_s_new.assign(project(tau*v_s + (1.0 - tau)*v_s_old, V_s.sub(1).collapse()))
            u_s_old.assign(u_s_new)
            v_s_old.assign(v_s_new)
                
            # Calculate errors on the interface
            u_error = project(interpolate(u_s, V_i.sub(0).collapse()) - \
                      interpolate(u_f, V_i.sub(0).collapse()), V_i.sub(0).collapse())
            v_error = project(interpolate(v_s, V_i.sub(1).collapse()) - \
                      interpolate(v_f, V_i.sub(1).collapse()), V_i.sub(1).collapse())
            u_error_vector = u_error.vector().get_local()
            v_error_vector = v_error.vector().get_local()
            U_error_vector = np.concatenate((u_error_vector, v_error_vector), axis = None)
            U_error = np.linalg.norm(U_error_vector)
            print('Error on the interface: ', str(U_error))
            
            # Define stop conditions
            if U_error < tol and save == False:
                
                print('Algorithm converged successfully after ', str(j), ' iterations.')
                print('In the next iteration solution will be saved.')
                stop = True
                
            elif j == maxit and save == False:
                
                print('Maximal number of iterations was reached.')
                print('In the next iteration solution will be saved.')
                stop = True
                
            # Assign old solutions 
            u_f_n.assign(u_f_n_M)
            v_f_n.assign(v_f_n_M)
            u_s_n.assign(u_s_n_K)
            v_s_n.assign(v_s_n_K)
            
            # Assign old boundary values
            u_f_n_i.assign(u_f_n_i_M)
            v_f_n_i.assign(v_f_n_i_M)
            v_s_n_i.assign(v_s_n_i_K)
            
        # Save number of iterations
        print('Number of solved linear systems: ', str(num_iters))
        iterations_txt = open('iterations.txt', 'a')
        iterations_txt.write(str(i + 1) + ' ' + str(num_iters) + ' ' + str(U_error)+ '\r\n')
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

        





