from fenics import Function
from solve_problem import solve_problem
from coupling import (solid_to_fluid,
                      fluid_to_solid)
from initial import Initial

def time_stepping(L_f_0, L_s_0,
                  A_f, L_f, A_s, L_s,
                  fluid, solid, interface,
                  param, decoupling, adjoint):

    # Initialize function objects
    if adjoint:

        u_name = 'adjoint_displacement'
        v_name = 'adjoint_velocity'

    else:

        u_name = 'primal_displacement'
        v_name = 'primal_velocity'

    u_f = Initial('fluid', u_name, fluid.V_split[0])
    v_f = Initial('fluid', v_name, fluid.V_split[1])
    u_s = Initial('solid', u_name, solid.V_split[0])
    v_s = Initial('solid', v_name, solid.V_split[1])

    # Create a table with numbers of iterations
    Num_iters = []

    # Save initial values for the primal problem
    if not adjoint:

        u_f.save(Function(fluid.V_split[0]))
        v_f.save(Function(fluid.V_split[1]))
        u_s.save(Function(solid.V_split[0]))
        v_s.save(Function(solid.V_split[1]))

    # Create time loop
    first_time_step = True
    for i in range(param.N):

        if not adjoint:

            print(f'Current time step {i + 1}')
            t = param.dt*(i + 1)

        else:

            print(f'Current time step {param.N - i}')
            t = param.dt*( param.N - i)

        # Perform decoupling
        Num_iters.append(decoupling(u_f, v_f, u_s, v_s,
                                    L_f_0, L_s_0,
                                    A_f, L_f, A_s, L_s,
                                    first_time_step,
                                    fluid, solid, interface, param, t))
        
        # Perform final iteration and save solutions
        solve_problem(u_f, v_f, u_s, v_s, fluid, solid,
                      solid_to_fluid, L_f_0, A_f, L_f,
                      first_time_step, param, t, save = True)
        solve_problem(u_s, v_s, u_f, v_f, solid, fluid,
                      fluid_to_solid, L_s_0, A_s, L_s,
                      first_time_step, param, t, save = True)

        first_time_step = False
                
        # Update solution
        u_f.old.assign(u_f.new)
        v_f.old.assign(v_f.new)
        u_s.old.assign(u_s.new)
        v_s.old.assign(v_s.new)
            
        # Update boundary conditions
        u_f.i_old.assign(u_f.i_new)
        v_f.i_old.assign(v_f.i_new)
        v_s.i_old.assign(v_s.i_new)

    # Save initial values for the adjoint problem
    if adjoint:

        u_f.save(Function(fluid.V_split[0]))
        v_f.save(Function(fluid.V_split[1]))
        u_s.save(Function(solid.V_split[0]))
        v_s.save(Function(solid.V_split[1]))
        
    return [u_f, v_f, u_s, v_s]
