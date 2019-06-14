from __future__ import print_function
from fenics import*
from parameters import*

# Define symetric projections
def dirichlet(u, V, W):
    u_v = u.vector()
    v = Function(W)
    v_v = v.vector()
    v2d = vertex_to_dof_map(V)
    for i in range(nx + 1):
        v_v[v2d[i]] = u_v[v2d[(nx + 1)*ny + i]]
    return v

def neumann(u, V, W):
    u_v = u.vector()
    v = Function(W)
    v_v = v.vector()
    v2d = vertex_to_dof_map(V)
    for i in range(ny + 1):
        for j in range(nx + 1):
            k = i*(nx + 1) + j
            m = (ny - i)*(nx + 1) + j
            v_v[v2d[m]] = -u_v[v2d[k]]
    return v



