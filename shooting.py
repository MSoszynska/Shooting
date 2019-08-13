import numpy as np
from fenics import (Function, FunctionSpace, 
                    interpolate, project, norm)
from fluid_problem import fluid_problem
from solid_problem import solid_problem
from scipy.sparse.linalg import LinearOperator, gmres

# Define shooting function
def S(u, v, fluid, solid, interface, param, t):
    
    # Save old values
    u_s = Function(solid.V_u)
    u_s.assign(u.s)
    v_s = Function(solid.V_v)
    v_s.assign(v.s)
    
    # Perform one iteration
    fluid_problem(u, v, fluid, solid, interface, param, t)
    solid_problem(u, v, fluid, solid, interface, param)
        
    # Define shooting function
    S_1 = interpolate(project(u_s - u.s, solid.V_u), interface.V_u)
    S_2 = interpolate(project(v_s - v.s, solid.V_v), interface.V_v)
        
    # Represent shooting function as an array
    S_1_array = S_1.vector().get_local()
    S_2_array = S_2.vector().get_local()
    S_array = np.concatenate((S_1_array, S_2_array), axis = None)
            
    return [S_array, norm(S_1, 'L2'), norm(S_2, 'L2')]
    
# Define linear operator for linear solver in shooting method
def shooting_newton(u, v, fluid, solid, interface, 
                    param, t, u_i, v_i, S_shooting):
    
    def shooting_gmres(D): 
            
        # Define empty functions on interface
        U_eps = Function(interface.V)
        (u_eps, v_eps) = U_eps.split(U_eps)
            
        # Split entrance vectors
        D_split = np.split(D, 2)
            
        # Set values of functions on interface
        u_eps.vector().set_local(u_i + param.eps*D_split[0])
        v_eps.vector().set_local(v_i + param.eps*D_split[1])
            
        # Interpolate functions on solid subdomain
        u_eps_s = interpolate(u_eps, solid.V_u)
        v_eps_s = interpolate(v_eps, solid.V_v)
        u.s.assign(u_eps_s)
        v.s.assign(v_eps_s)
                    
        # Compute shooting function
        S_eps = S(u, v, fluid, solid, interface, param, t)[0]
            
        return (S_eps - S_shooting)/param.eps
    
    return shooting_gmres

def shooting(u, v, fluid, solid, interface, 
             param, t):
        
    # Define initial values for Newton's method
    u_s = Function(solid.V_u)
    v_s = Function(solid.V_v)
    u_s.assign(u.s_n)
    v_s.assign(v.s_n)
    num_iters = 0
    num_linear = 0
    stop = False
        
    # Define Newton's method
    while not stop:
        
        num_iters += 1
        num_linear += 1
        print('Current iteration of Newton\'s method: ', num_iters)
        
        # Define right hand side
        u.s.assign(u_s)
        v.s.assign(v_s)
        S_shooting = S(u, v, fluid, solid, interface, param, t)
        print('Error on the interface of displacement: ', S_shooting[1])
        print('Error on the interface of velocity: ', S_shooting[2])
            
        # Define linear operator
        u_s_i = interpolate(u_s, interface.V_u)
        v_s_i = interpolate(v_s, interface.V_v)
        u_s_i_array = u_s_i.vector().get_local()
        v_s_i_array = v_s_i.vector().get_local()
        linear_operator_newton = shooting_newton(u, v, fluid, solid, 
                                                 interface, param, t, 
                                                 u_s_i_array, v_s_i_array, 
                                                 S_shooting[0])
        shooting_gmres = LinearOperator((2*param.nx + 2, 2*param.nx + 2), 
                                        matvec = linear_operator_newton)
            
        # Solve linear system
        num_iters_gmres = 0
        def callback(xk):
            
            nonlocal num_iters_gmres
            global res_norm_gmres
            num_iters_gmres += 1
            print('Current iteration of GMRES method: ', 
                   num_iters_gmres)
            res_norm_gmres = np.linalg.norm(xk)
            
        D, exit_code = gmres(shooting_gmres, -S_shooting[0], 
                             tol = param.tol_gmres, 
                             maxiter = param.maxiter_gmres, 
                             callback = callback)
        num_linear += num_iters_gmres
        if (exit_code == 0):
            
            print('GMRES method converged successfully after', 
                   num_iters_gmres, 'iterations.')
            
        else:
            
            print('GMRES method failed to converge.')
            print('Norm of residual: ', str(res_norm_gmres))
            
        # Advance solution
        D_split = np.split(D, 2) 
        u_s_i_array += D_split[0]
        v_s_i_array += D_split[1]
        u_s_i.vector().set_local(u_s_i_array)
        v_s_i.vector().set_local(v_s_i_array)
        u_s.assign(interpolate(u_s_i, solid.V_u))
        v_s.assign(interpolate(v_s_i, solid.V_v))
            
        # Check stop conditions
        if ((S_shooting[1] < param.tol_newton) and 
           (S_shooting[2] < param.tol_newton)):
               
            print('Newton\'s method converged successfully after', num_iters, 
                  'iterations and solving', num_linear, 'linear systems.')
            stop = True
            
        elif (num_iters == param.maxiter_newton):
            
            print('Newton\'s method failed to converge.')
            stop = True
                
    return num_linear
