from fenics import FunctionSpace, Function
from parameters import*

def flip(u, V, param):
    u_v = u.vector()
    n = (param.nx + 1)*(param.ny + 1)
    v = Function(V)
    v_v = v.vector()
    for i in range(n):
        v_v[i] = u_v[n - i - 1]
    return v


