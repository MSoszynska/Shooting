from fenics import Function, FunctionSpace
from coupling import flip

def time_stepping(u, v, fluid, solid, interface, \
                  param, decoupling):

    # Create time loop
    for i in range(param.N):
        
        print('Current time step', i + 1)
        t = param.dt*(i + 1)
        
        # Perform decoupling
        decoupling(u, v, fluid, solid, interface, param, t)
        
        # Save values
        u.attach()
        v.attach()
                
        # Update solution
        u.f_n.assign(u.f)
        v.f_n.assign(v.f)
        u.s_n.assign(u.s)
        v.s_n.assign(v.s)
            
        # Update boundary conditions
        u.f_n_i.assign(flip(u.s, \
                fluid.V.sub(0).collapse(), param))
        v.f_n_i.assign(flip(v.s, \
                fluid.V.sub(1).collapse(), param))
        u.s_n_i.assign(flip(u.f, \
                solid.V.sub(0).collapse(), param))
        v.s_n_i.assign(flip(v.f, \
                solid.V.sub(1).collapse(), param))
        
    return
    
            
            
            
            
            
