from fenics import Function, FunctionSpace, vertex_to_dof_map

# Copy two layers of dofs around the interface from fluid to solid
def fluid_to_solid(u, fluid, solid, param, i):
    
    u_v = u.vector()
    v2d_f = vertex_to_dof_map(fluid.V.sub(i).collapse())
    v = Function(solid.V.sub(i).collapse())
    v_v = v.vector()
    v2d_s = vertex_to_dof_map(solid.V.sub(i).collapse())
    N = param.nx + 1
    M = param.ny + 1
    for i in range(2):
        
        for j in range(N):
            
            v_v[v2d_s[(M - i - 1)*N + j]] = u_v[v2d_f[i*N + j]]
            
    return v

# Copy two layers of dofs around the interface from solid to fluid
def solid_to_fluid(u, fluid, solid, param, i):
    
    u_v = u.vector()
    v2d_s = vertex_to_dof_map(solid.V.sub(i).collapse())
    v = Function(fluid.V.sub(i).collapse())
    v_v = v.vector()
    v2d_f = vertex_to_dof_map(fluid.V.sub(i).collapse())
    N = param.nx + 1
    M = param.ny + 1
    for i in range(2):
        
        for j in range(N):
            
            v_v[v2d_f[i*N + j]] = u_v[v2d_s[(M - i - 1)*N + j]]
            
    return v
