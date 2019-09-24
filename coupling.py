from fenics import Function, FunctionSpace, vertex_to_dof_map

# Copy two layers of dofs around the interface from fluid to solid
def fluid_to_solid(u, solid, fluid, param, subspace_index):
    
    u_v = u.vector()
    v2d_f = vertex_to_dof_map(fluid.V_split[subspace_index])
    v = Function(solid.V_split[subspace_index])
    v_v = v.vector()
    v2d_s = vertex_to_dof_map(solid.V_split[subspace_index])
    N = param.nx + 1
    M = param.ny + 1
    for i in range(2):
        
        for j in range(N):
            
            v_v[v2d_s[(M - i - 1) * N + j]] = u_v[v2d_f[i * N + j]]
            
    return v

# Copy two layers of dofs around the interface from solid to fluid
def solid_to_fluid(u, fluid, solid, param, subspace_index):
    
    u_v = u.vector()
    v2d_s = vertex_to_dof_map(solid.V_split[subspace_index])
    v = Function(fluid.V_split[subspace_index])
    v_v = v.vector()
    v2d_f = vertex_to_dof_map(fluid.V_split[subspace_index])
    N = param.nx + 1
    M = param.ny + 1
    for i in range(2):
        
        for j in range(N):
            
            v_v[v2d_f[i * N + j]] = u_v[v2d_s[(M - i - 1) * N + j]]
            
    return v
