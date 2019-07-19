from fenics import Function, FunctionSpace
from coupling import fluid_to_solid, solid_to_fluid

def time_stepping(u, v, fluid, solid, interface, 
                  param, decoupling):
    
    # Create a table with numbers of iterations
    Num_iters = []

    # Create time loop
    for i in range(param.N):
        
        print('Current time step', i + 1)
        t = param.dt*(i + 1)
        
        # Perform decoupling
        Num_iters.append(decoupling(u, v, fluid, solid, 
                                    interface, param, t))
        
        # Save values
        u.attach()
        v.attach()
                
        # Update solution
        u.f_n.assign(u.f)
        v.f_n.assign(v.f)
        u.s_n.assign(u.s)
        v.s_n.assign(v.s)
            
        # Update boundary conditions
        u.f_n_i.assign(solid_to_fluid(u.s, fluid, solid, param, 0))
        v.f_n_i.assign(solid_to_fluid(v.s, fluid, solid, param, 1))
        u.s_n_i.assign(fluid_to_solid(u.f, fluid, solid, param, 0))
        v.s_n_i.assign(fluid_to_solid(v.f, fluid, solid, param, 1))
        
    return Num_iters
