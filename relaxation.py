from fenics import (Function, FunctionSpace, 
                    project, interpolate, norm)
from fluid_problem import fluid_problem
from solid_problem import solid_problem


# Define relaxation method
def relaxation(u_f, v_f, u_s, v_s,
               fluid, solid, interface, param, t):
    
    # Define initial values for relaxation method
    u_s_new = Function(solid.V_split[0])
    v_s_new = Function(solid.V_split[1])
    num_iters = 0
    stop = False
    
    while not stop:
        
        num_iters += 1
        print('Current iteration of relaxation method: ', 
               num_iters)
        
        # Save old values
        u_s_new.assign(u_s.new)
        v_s_new.assign(v_s.new)
    
        # Perform one iteration
        fluid_problem(u_f, v_f, u_s, v_s, fluid, solid, param, t)
        solid_problem(u_f, v_f, u_s, v_s, fluid, solid, param)
        
        # Perform relaxation
        u_s.new.assign(project(param.tau*u_s.new
                    + (1.0 - param.tau)*u_s_new, solid.V_split[0]))
        v_s.new.assign(project(param.tau*v_s.new
                    + (1.0 - param.tau)*v_s_new, solid.V_split[1]))
        
        # Define errors on the interface
        u_error = interpolate(project(u_s_new - u_s.new, solid.V_split[0]),
                  interface.V_split[0])
        u_error_linf = norm(u_error.vector(), 'linf')
        v_error = interpolate(project(v_s_new - v_s.new, solid.V_split[1]),
                  interface.V_split[1])
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
