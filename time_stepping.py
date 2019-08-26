from fluid_problem import fluid_problem
from solid_problem import solid_problem

def time_stepping(u_f, v_f, u_s, v_s,
                  fluid, solid, interface, param, decoupling):
    
    # Create a table with numbers of iterations
    Num_iters = []

    # Create time loop
    for i in range(param.N):
        
        print('Current time step', i + 1)
        t = param.dt*(i + 1)
        
        # Perform decoupling
        Num_iters.append(decoupling(u_f, v_f, u_s, v_s,
                                    fluid, solid, interface, param, t))
        
        # Perform final iteration and save solutions
        fluid_problem(u_f, v_f, u_s, v_s,
                      fluid, solid, param, t, save = True)
        solid_problem(u_f, v_f, u_s, v_s,
                      fluid, solid, param, save = True)
                
        # Update solution
        u_f.old.assign(u_f.new)
        v_f.old.assign(v_f.new)
        u_s.old.assign(u_s.new)
        v_s.old.assign(v_s.new)
            
        # Update boundary conditions
        u_f.i_old.assign(u_f.i_new)
        v_f.i_old.assign(v_f.i_new)
        v_s.i_old.assign(v_s.i_new)
        
    return Num_iters
