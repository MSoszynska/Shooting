from fenics import (Function, FunctionSpace, 
                    project, interpolate, norm)
from fluid_problem import fluid_problem
from solid_problem import solid_problem


# Define relaxation method
def relaxation(u, v, fluid, solid, interface, 
               param, t):
    
    num_iters = 0
    stop = False
    
    while not stop:
        
        num_iters += 1
        print('Current iteration of relaxation method: ', 
               num_iters)
        
        # Save old values
        u_s = Function(solid.V_u)
        u_s.assign(u.s)
        v_s = Function(solid.V_v)
        v_s.assign(v.s)
    
        # Perform one iteration
        fluid_problem(u, v, fluid, solid, interface, param, t)
        solid_problem(u, v, fluid, solid, interface, param)
        
        # Perform relaxation
        u.s.assign(project(param.tau*u.s 
                        + (1.0 - param.tau)*u_s, solid.V_u))
        v.s.assign(project(param.tau*v.s 
                        + (1.0 - param.tau)*v_s, solid.V_v))
        
        # Define errors on the interface
        u_error = interpolate(project(u_s - u.s, solid.V_u), 
                  interface.V_u)
        u_error_L2 = norm(u_error, 'L2')
        print('Error on the interface of displacement: ', u_error_L2)
        v_error = interpolate(project(v_s - v.s, solid.V_v), 
                  interface.V_v)
        v_error_L2 = norm(v_error, 'L2')
        print('Error on the interface of velocity: ', v_error_L2)
        
        # Check stop conditions
        if ((u_error_L2 < param.tol_relax) and 
           (v_error_L2 < param.tol_relax)):
                
            print('Algorithm converged successfully after ', 
                   num_iters, ' iterations.')
            stop = True
                
        elif (num_iters == param.maxiter_relax):
                
            print('Maximal number of iterations was reached.')
            stop = True
        
    return num_iters
