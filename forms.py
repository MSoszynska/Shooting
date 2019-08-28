from fenics import dot, grad
from parameters import f

# Define variational form for fluid
def a_f(u_f, v_f, phi_f, psi_f, fluid, param):

    return (param.nu*dot(grad(v_f), grad(phi_f)) * fluid.dx
            + dot(param.beta, grad(v_f)) * phi_f * fluid.dx
            + dot(grad(u_f), grad(psi_f)) * fluid.dx
            - grad(u_f)[1] * psi_f * fluid.ds
            - param.nu * grad(v_f)[1] * phi_f * fluid.ds
            + param.gamma / fluid.h * u_f * psi_f * fluid.ds
            + param.gamma / fluid.h * v_f * phi_f * fluid.ds)

def A_f(u_f, v_f, phi_f, psi_f, fluid, param):

    return (v_f * phi_f * fluid.dx + 0.5 * param.dt / param.M
            * a_f(u_f, v_f, phi_f, psi_f, fluid, param))

def L_f(u_f_n, v_f_n, u_f_i, v_f_i, u_f_n_i, v_f_n_i,
        phi_f, psi_f, fluid, param, t):

    return (v_f_n * phi_f * fluid.dx
            - 0.5 * param.dt / param.M
            * a_f(u_f_n, v_f_n, phi_f, psi_f, fluid, param)
            + 0.5 * param.dt / param.M * param.gamma
            / fluid.h * u_f_i * psi_f * fluid.ds
            + 0.5 * param.dt / param.M * param.gamma
            / fluid.h * v_f_i * phi_f * fluid.ds
            + 0.5 * param.dt / param.M * param.gamma
            / fluid.h * u_f_n_i * psi_f * fluid.ds
            + 0.5 * param.dt / param.M * param.gamma
            / fluid.h * v_f_n_i * phi_f * fluid.ds
            + param.dt / param.M * f(t) * phi_f * fluid.dx)

# Define variational form for solid
def a_s(u_s, v_s, phi_s, psi_s, solid, param):

    return (param.zeta * dot(grad(u_s), grad(phi_s)) * solid.dx
            + param.delta * dot(grad(v_s), grad(phi_s)) * solid.dx
            - v_s * psi_s * solid.dx
            + param.delta * grad(v_s)[1] * phi_s * solid.ds)

def A_s(u_s, v_s, phi_s, psi_s, solid, param):

    return (v_s * phi_s * solid.dx + u_s * psi_s * solid.dx
            + 0.5 * param.dt / param.K
            * a_s(u_s, v_s, phi_s, psi_s, solid, param))

def L_s(u_s_n, v_s_n, u_s_i, v_s_i, u_s_n_i, v_s_n_i,
        phi_s, psi_s, solid, param, t):

    return (v_s_n * phi_s * solid.dx
            + u_s_n * psi_s * solid.dx
            - 0.5 * param.dt / param.K
            * a_s(u_s_n, v_s_n, phi_s, psi_s, solid, param)
            + 0.5 * param.dt / param.K * param.nu
            * dot(grad(v_s_i), solid.normal) * phi_s * solid.ds
            + 0.5 * param.dt / param.K * param.nu
            * dot(grad(v_s_n_i), solid.normal) * phi_s * solid.ds)
