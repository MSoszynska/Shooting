from __future__ import print_function
from fenics import*
from parameters import*
from spaces import*
from fluid import fluid
from solid import solid
from coupling import dirichlet, neumann
import os
import numpy as np

print('Primal problem using relaxation')

parameters['allow_extrapolation'] = True

# Define initial interface values
u_f_n_i = Function(V_f.sub(0).collapse())
v_f_n_i = Function(V_f.sub(1).collapse())
u_s_n_i = Function(V_s.sub(0).collapse())
v_s_n_i = Function(V_s.sub(1).collapse())

# Define initial values for relaxation
u_s_old = Function(V_s.sub(0).collapse())
v_s_old = Function(V_s.sub(1).collapse())
u_s_new = Function(V_s.sub(0).collapse())
v_s_new = Function(V_s.sub(1).collapse())

# Define initial values for time loop
u_f_n = Function(V_f.sub(0).collapse())
v_f_n = Function(V_f.sub(1).collapse())
u_s_n = Function(V_s.sub(0).collapse())
v_s_n = Function(V_s.sub(1).collapse())

# Create arrays of empty functions
u_f_array = []
u_f_array.append(Function(V_f.sub(0).collapse()))
v_f_array = []
v_f_array.append(Function(V_f.sub(1).collapse()))
u_s_array = []
u_s_array.append(Function(V_s.sub(0).collapse()))
v_s_array = []
v_s_array.append(Function(V_s.sub(1).collapse()))

# Create directory
directory = os.getcwd()
os.makedirs('relaxation')
os.chdir('relaxation')

# Create pvd files
u_f_pvd = File('solutions/fluid/displacement.pvd')
v_f_pvd = File('solutions/fluid/velocity.pvd')
u_s_pvd = File('solutions/solid/displacement.pvd')
v_s_pvd = File('solutions/solid/velocity.pvd')

# Create time loop
for i in range(N):
    
    print('Current time step', i + 1)
    t = dt*(i + 1)
    save = False
    stop = False
    num_iter = 0
    
    # Preform decoupling:
    while (stop == False):
        
        num_iter += 1
        if (save == True):
            stop = True
        
        # Solve fluid problem
        Fluid = fluid(u_s_old, v_s_old, u_f_n, v_f_n, \
                      u_f_n_i, v_f_n_i, t)
        
        # Solve solid problem
        Solid = solid(Fluid[1], u_s_n, v_s_n, v_s_n_i)
        
        if save == True:
                    
            # Save values
            u_f_array.append(Fluid[0])
            v_f_array.append(Fluid[1])
            u_s_array.append(Solid[0])
            v_s_array.append(Solid[1])
                    
            # Save solution
            Fluid[0].rename('Displacement', 'Fluid')
            u_f_pvd << Fluid[0] 
            Fluid[1].rename('Velocity', 'Fluid')
            v_f_pvd << Fluid[1]
            Solid[0].rename('Displacement', 'Solid')
            u_s_pvd << Solid[0]
            Solid[1].rename('Velocity', 'Solid')
            v_s_pvd << Solid[1]  
        
        # Perform relaxataion
        u_s_new.assign(project(tau*Solid[0] + \
                      (1.0 - tau)*u_s_old, V_s.sub(0).collapse()))
        v_s_new.assign(project(tau*Solid[1] + \
                      (1.0 - tau)*v_s_old, V_s.sub(1).collapse()))
        if (stop == False):
            u_s_old.assign(u_s_new)
            v_s_old.assign(v_s_new)
        
        # Compute errors on the interface
        u_error = project(interpolate(Solid[0], V_i.sub(0).collapse()) - \
                  interpolate(Fluid[0], V_i.sub(0).collapse()), V_i.sub(0).collapse())
        print('Error on the interface of displacement: ', str(norm(u_error, 'L2')))
        v_error = project(interpolate(Solid[1], V_i.sub(1).collapse()) - \
                  interpolate(Fluid[1], V_i.sub(1).collapse()), V_i.sub(1).collapse())
        print('Error on the interface of velocity: ', str(norm(v_error, 'L2')))
        
        # Check stop conditions
        if (num_iter == maxiter_relax) and (save == False):
            print('Maximal number of iterations was reached.')
            print('In the next iteration solution will be saved.')
            save = True
        elif (norm(u_error, 'L2') < tol_relax) and (norm(v_error, 'L2') < tol_relax) \
                                               and (save == False):
            print('Algorithm converged successfully after ', \
                   str(num_iter), ' iterations.')
            print('In the next iteration solution will be saved.')
            save = True
            
    # Update solution
    u_f_n.assign(Fluid[0])
    v_f_n.assign(Fluid[1])
    u_s_n.assign(u_s_old)
    v_s_n.assign(v_s_old)
        
    # Update boundary conditions
    u_f_n_i.assign(dirichlet(u_s_old, V_s.sub(0).collapse(), \
            V_f.sub(0).collapse()))
    v_f_n_i.assign(dirichlet(v_s_old, V_s.sub(1).collapse(), \
            V_f.sub(1).collapse()))
    u_s_n_i.assign(neumann(Fluid[0], V_f.sub(0).collapse(), \
              V_s.sub(0).collapse()))
    v_s_n_i.assign(neumann(Fluid[1], V_f.sub(1).collapse(), \
              V_s.sub(1).collapse()))
            
os.chdir(directory)
            
            
            
            
