from fenics import (Function, FunctionSpace, 
                    project, interpolate, norm)
from fluid_problem import fluid_problem
from solid_problem import solid_problem


# Define relaxation method
def relaxation(u, v, fluid, solid, interface, 
               param, t):
    
    # Define initial values for relaxation method
    u_s = Function(solid.V_u)
    v_s = Function(solid.V_v)
    num_iters = 0
    stop = False
    
    while not stop:
        
        num_iters += 1
        print('Current iteration of relaxation method: ', 
               num_iters)
        
        # Save old values
        u_s.assign(u.s)
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
        u_error_linf = norm(u_error.vector(), 'linf')
        v_error = interpolate(project(v_s - v.s, solid.V_v), 
                  interface.V_v)
        v_error_linf = norm(v_error.vector(), 'linf')
        error_linf = max(u_error_linf, v_error_linf)
        if (num_iters == 1):
            error_0_linf = error_linf
        print('Absolute error on the interface: ', error_linf)
        print('Relative error on the interface: ', error_linf/error_0_linf)
        
        # Check stop conditions
        if ((error_linf < param.abs_tol_relax) or 
            (error_linf/error_0_linf < param.rel_tol_relax)):
                
            print('Algorithm converged successfully after ', 
                   num_iters, ' iterations.')
            stop = True
                
        elif (num_iters == param.maxiter_relax):
                
            print('Maximal number of iterations was reached.')
            stop = True
        
    return num_iters
