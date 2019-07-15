from fenics import Function, FunctionSpace, \
                   project, interpolate, norm 
from fluid_problem import fluid_problem
from solid_problem import solid_problem

# Define relaxation method
def relaxation(u, v, fluid, solid, interface, param, t):
    
    num_iters = 0
    stop = False
    
    while (stop == False):
        
        num_iters += 1
        print('Current iteration of relaxation method: ', \
               num_iters)
        
        # Save old values
        u_s = Function(solid.V.sub(0).collapse())
        u_s.assign(u.s)
        v_s = Function(solid.V.sub(1).collapse())
        v_s.assign(v.s)
    
        # Perform one iteration
        fluid_problem(u, v, fluid, solid, interface, param, t)
        solid_problem(u, v, fluid, solid, interface, param)
        
        # Perform relaxation
        u.s.assign(project(param.tau*u.s + \
                          (1.0 - param.tau)*u_s, solid.V.sub(0).collapse()))
        v.s.assign(project(param.tau*v.s + \
                          (1.0 - param.tau)*v_s, solid.V.sub(1).collapse()))
        
        # Define errors on the interface
        u_error = interpolate(project(u_s - u.s, solid.V.sub(0).collapse()), \
                  interface.V.sub(0).collapse())
        u_error_L2 = norm(u_error, 'L2')
        print('Error on the interface of displacement: ', u_error_L2)
        v_error = interpolate(project(v_s - v.s, solid.V.sub(1).collapse()), \
                  interface.V.sub(1).collapse())
        v_error_L2 = norm(v_error, 'L2')
        print('Error on the interface of velocity: ', v_error_L2)
        
        # Check stop conditions
        if (u_error_L2 < param.tol_relax) and \
           (v_error_L2 < param.tol_relax):
                
            print('Algorithm converged successfully after ', \
                   num_iters, ' iterations.')
            stop = True
                
        elif (num_iters == param.maxiter_relax):
                
            print('Maximal number of iterations was reached.')
            stop = True
        
    return
