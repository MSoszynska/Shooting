from fenics import dot, grad

# Define variational form for fluid
def a_f(u_f, v_f, phi_f, psi_f, fluid, param):
    return (param.nu*dot(grad(v_f), grad(phi_f))*fluid.dx 
            + dot(param.beta, grad(v_f))*phi_f*fluid.dx 
            + dot(grad(u_f), grad(psi_f))*fluid.dx 
            - grad(u_f)[1]*psi_f*fluid.ds 
            - param.nu*grad(v_f)[1]*phi_f*fluid.ds 
            + param.gamma/fluid.h*u_f*psi_f*fluid.ds 
            + param.gamma/fluid.h*v_f*phi_f*fluid.ds)

# Define variational form for solid
def a_s(u_s, v_s, phi_s, psi_s, solid, param):
    return (param.zeta*dot(grad(u_s), grad(phi_s))*solid.dx 
            + param.delta*dot(grad(v_s), grad(phi_s))*solid.dx 
            - v_s*psi_s*solid.dx 
            + param.delta*grad(v_s)[1]*phi_s*solid.ds)
