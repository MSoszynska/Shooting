from __future__ import print_function
from fenics import*
from parameters import*
import os
import numpy as np

def coupled():
    
    print('Fully coupled problem')
    
    # Define boundary
    def boundary(x, on_boundary):
        return on_boundary

    # Define fluid subdomain 
    class Fluid(SubDomain):
        def inside(self, x, on_boundary):
            return (between(x[0], (0.0, 4.0)) and between(x[1], (0.0, 1.0)))
    fluid = Fluid()
        
    # Define solid subdomain
    class Solid(SubDomain):
        def inside(self, x, on_boundary):
            return (between(x[0], (0.0, 4.0)) and between(x[1], (-1.0, 0.0)))
    solid = Solid()

    # Define mesh
    mesh = RectangleMesh(Point(0.0, -1.0), Point(4.0, 1.0), nx, ny)
    h = CellDiameter(mesh)

    # Mark subdomains
    subdomains = MeshFunction('size_t', mesh, mesh.topology().dim())
    fluid.mark(subdomains, 0)
    solid.mark(subdomains, 1)

    # Function space
    W = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, W * W)

    # Define boundary conditions
    u_0 = Constant(0.0)
    v_0 = Constant(0.0)
    bc_u = DirichletBC(V.sub(0), u_0, boundary)
    bc_v = DirichletBC(V.sub(1), v_0, boundary)
    bcs = [bc_u, bc_v]

    # Define right hand side
    f = Expression('exp(-10.0*(pow(x[0] - 0.5, 2) + pow(x[1] - \
                    0.5, 2)))*sign', sign = 1.0, degree = 1)
    N_external = int(1.0/(10.0*dt))

    # Define initial values
    u_n = interpolate(Constant(0.0), V.sub(0).collapse())
    v_n = interpolate(Constant(0.0), V.sub(1).collapse())

    # Define measure
    dx = Measure('dx', domain = mesh, subdomain_data = subdomains)
    
    # Define variational forms
    def a_f(u, v, phi, psi):
        return nu*dot(grad(v), grad(phi))*dx(0) \
             + dot(beta, grad(v))*phi*dx(0) \
             + alpha*h*h*dot(grad(u), grad(psi))*dx(0)
    def a_s(u, v, phi, psi):
        return zeta*dot(grad(u), grad(phi))*dx(1) \
             + delta*dot(grad(v), grad(phi))*dx(1) \
             - v*psi*dx(1) 
         
    # Create directory
    directory = os.getcwd()
    os.makedirs('coupled')
    os.chdir('coupled')
    
    # Create pvd files
    u_file_pvd = File('solution/displacement.pvd')
    v_file_pvd = File('solution/velocity.pvd')
    
    # Create arrays 
    U_array = []
    U_array.append(project(u_0, V.sub(0).collapse()))
    V_array = []
    V_array.append(project(v_0, V.sub(1).collapse()))
         
    # Create time loop 
    t = 0.0
    U = Function(V)
    switch = True
    count = 0
    for i in range(N):
        
        # Update current time
        print('Current time step: ', i + 1)
        t += dt
        
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
        
        # Define test and trial functions
        U = TrialFunction(V)
        (u, v) = split(U)
        Phi = TestFunction(V)
        (phi, psi) = split(Phi)
        
        # Define scheme
        A_f = v*phi*dx(0) + \
              0.5*dt*a_f(u, v, phi, psi)
        A_s = v*phi*dx(1) + \
              u*psi*dx(1) + \
              0.5*dt*a_s(u, v, phi, psi)
        A = A_f + A_s
        L_f = v_n*phi*dx(0) - \
              0.5*dt*a_f(u_n, v_n, phi, psi) + \
              dt*f*phi*dx(0) 
        L_s = v_n*phi*dx(1) + \
              u_n*psi*dx(1) - \
              0.5*dt*a_s(u_n, v_n, phi, psi)
        L = L_f + L_s
            
        # Compute solution
        U = Function(V)
        solve(A == L, U, bcs)
        (u, v) = U.split(U)
        
        # Save values
        U_array.append(u)
        V_array.append(v)
        
        # Save solution
        u.rename('Displacement', 'FullyCoupled')
        u_file_pvd << u
        v.rename('Velocity', 'FullyCoupled')
        v_file_pvd << v
        
        # Update previous solution
        u_n.assign(u)
        v_n.assign(v)
        
    os.chdir(directory)
        
    return [U_array, V_array]

    
    
    
        
    


