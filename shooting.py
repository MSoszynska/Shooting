import numpy as np
from fenics import (Function, FunctionSpace, 
                    interpolate, project)
from solve_problem import solve_problem
from coupling import (solid_to_fluid,
                      fluid_to_solid)
from scipy.sparse.linalg import LinearOperator, gmres

# Define shooting function
def S(u_f, v_f, u_s, v_s, L_f_0, L_s_0,
      A_f, L_f, A_s, L_s, first_time_step,
      fluid, solid, interface, param, t):
    
    # Save old values
    u_s_new = Function(solid.V_split[0])
    u_s_new.assign(u_s.new)
    v_s_new = Function(solid.V_split[1])
    v_s_new.assign(v_s.new)
    
    # Perform one iteration
    solve_problem(u_f, v_f, u_s, v_s, fluid, solid,
                  solid_to_fluid, L_f_0, A_f, L_f,
                  first_time_step, param, t)
    solve_problem(u_s, v_s, u_f, v_f, solid, fluid,
                  fluid_to_solid, L_s_0, A_s, L_s,
                  first_time_step, param, t)
        
    # Define shooting function
    S_1 = interpolate(project(u_s_new - u_s.new, solid.V_split[0]),
                      interface.V_split[0])
    S_2 = interpolate(project(v_s_new - v_s.new, solid.V_split[1]),
                      interface.V_split[1])
        
    # Represent shooting function as an array
    S_1_array = S_1.vector().get_local()
    S_2_array = S_2.vector().get_local()
    S_array = np.concatenate((S_1_array, S_2_array), axis = None)
    
    return S_array
    
# Define linear operator for linear solver in shooting method
def shooting_newton(u_f, v_f, u_s, v_s,
                    L_f_0, L_s_0,
                    A_f, L_f, A_s, L_s,
                    first_time_step, fluid, solid, interface,
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
        u_eps_s = interpolate(u_eps, solid.V_split[0])
        v_eps_s = interpolate(v_eps, solid.V_split[1])
        u_s.new.assign(u_eps_s)
        v_s.new.assign(v_eps_s)
                    
        # Compute shooting function
        S_eps = S(u_f, v_f, u_s, v_s, L_f_0, L_s_0,
                  A_f, L_f, A_s, L_s, first_time_step,
                  fluid, solid, interface, param, t)
            
        return (S_eps - S_shooting) / param.eps
    
    return shooting_gmres

def shooting(u_f, v_f, u_s, v_s,
             L_f_0, L_s_0,
             A_f, L_f, A_s, L_s,
             first_time_step,
             fluid, solid, interface, param, t):
        
    # Define initial values for Newton's method
    u_s_new = Function(solid.V_split[0])
    v_s_new = Function(solid.V_split[1])
    u_s_new.assign(u_s.old)
    v_s_new.assign(v_s.old)
    num_iters = 0
    num_linear = 0
    stop = False
        
    # Define Newton's method
    while not stop:
        
        num_iters += 1
        num_linear += 1
        print(f'Current iteration of Newton\'s method: {num_iters}')
        
        # Define right hand side
        u_s.new.assign(u_s_new)
        v_s.new.assign(v_s_new)
        S_shooting = S(u_f, v_f, u_s, v_s,
                       L_f_0, L_s_0,
                       A_f, L_f, A_s, L_s,
                       first_time_step,
                       fluid, solid, interface, param, t)
        S_shooting_linf = np.max(np.abs(S_shooting))
        if num_iters == 1:

            S_0_linf = S_shooting_linf

        print(f'Absolute error on the interface: '
              f'{S_shooting_linf}')
        print(f'Relative error on the interface: '
              f'{S_shooting_linf / S_0_linf}')
            
        # Define linear operator
        u_s_i = interpolate(u_s_new, interface.V_split[0])
        v_s_i = interpolate(v_s_new, interface.V_split[1])
        u_s_i_array = u_s_i.vector().get_local()
        v_s_i_array = v_s_i.vector().get_local()
        linear_operator_newton = shooting_newton(u_f, v_f, u_s, v_s,
                                                 L_f_0, L_s_0,
                                                 A_f, L_f, A_s, L_s,
                                                 first_time_step,
                                                 fluid, solid, interface,
                                                 param, t, u_s_i_array,
                                                 v_s_i_array, S_shooting)
        shooting_gmres = LinearOperator((2 * param.nx + 2, 2 * param.nx + 2),
                                        matvec = linear_operator_newton)
            
        # Solve linear system
        num_iters_gmres = 0
        def callback(xk):
            
            nonlocal num_iters_gmres
            global res_norm_gmres
            num_iters_gmres += 1
            print(f'Current iteration of GMRES method: {num_iters_gmres}')
            res_norm_gmres = np.linalg.norm(xk)
            
        D, exit_code = gmres(shooting_gmres, - S_shooting,
                             tol = param.tol_gmres, 
                             maxiter = param.maxiter_gmres, 
                             callback = callback)
        num_linear += num_iters_gmres
        if exit_code == 0:
            
            print(f'GMRES method converged successfully after '
                  f'{num_iters_gmres} iterations')
            
        else:
            
            print('GMRES method failed to converge.')
            print(f'Norm of residual: {res_norm_gmres}')
            
        # Advance solution
        D_split = np.split(D, 2) 
        u_s_i_array += D_split[0]
        v_s_i_array += D_split[1]
        u_s_i.vector().set_local(u_s_i_array)
        v_s_i.vector().set_local(v_s_i_array)
        u_s_new.assign(interpolate(u_s_i, solid.V_split[0]))
        v_s_new.assign(interpolate(v_s_i, solid.V_split[1]))
            
        # Check stop conditions
        if (S_shooting_linf < param.abs_tol_newton or
                S_shooting_linf/S_0_linf < param.rel_tol_newton):
               
            print(f'Newton\'s method converged successfully after '
                  f'{num_iters} iterations and solving '
                  f'{num_linear} linear systems.')
            stop = True
            
        elif num_iters == param.maxiter_newton:
            
            print('Newton\'s method failed to converge.')
            stop = True

    u_f.iterations.append(num_linear)
    v_f.iterations.append(num_linear)
    u_s.iterations.append(num_linear)
    u_s.iterations.append(num_linear)
                
    return
