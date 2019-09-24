from fenics import dot, grad, Expression
from parameters import f

# Define characteristic functions corresponding to chosen functionals
def char_func_J1(param):

    return Expression('2.0 <= x[0] && 0.0 <= x[1] ? J1 : 0.0',
                      J1 = param.J1, degree = 0)

def char_func_J2(param):

    return Expression('2.0 <= x[0] && x[1] <= 0.0 ? J2 : 0.0',
                      J2 = param.J2, degree = 0)

# Define variational forms of the fluid subproblem
def a_f(u_f, v_f, phi_f, psi_f, fluid, param):

    return (param.nu*dot(grad(v_f), grad(phi_f)) * fluid.dx
            + dot(param.beta, grad(v_f)) * phi_f * fluid.dx
            + dot(grad(u_f), grad(psi_f)) * fluid.dx
            - dot(grad(u_f), fluid.normal) * psi_f * fluid.ds(1)
            - param.nu * dot(grad(v_f), fluid.normal) * phi_f * fluid.ds(1)
            + param.gamma / fluid.h * u_f * psi_f * fluid.ds(1)
            + param.gamma / fluid.h * v_f * phi_f * fluid.ds(1))

def A_f(u_f, v_f, phi_f, psi_f, fluid, param):

    return (v_f * phi_f * fluid.dx + 0.5 * param.dt / param.M
            * a_f(u_f, v_f, phi_f, psi_f, fluid, param))

def L_f(u_f_n, v_f_n, u_f_i, v_f_i, u_f_n_i, v_f_n_i,
        phi_f, psi_f, fluid, param, t):

    return (v_f_n * phi_f * fluid.dx
            - 0.5 * param.dt / param.M
            * a_f(u_f_n, v_f_n, phi_f, psi_f, fluid, param)
            + 0.5 * param.dt / param.M * param.gamma
            / fluid.h * u_f_i * psi_f * fluid.ds(1)
            + 0.5 * param.dt / param.M * param.gamma
            / fluid.h * v_f_i * phi_f * fluid.ds(1)
            + 0.5 * param.dt / param.M * param.gamma
            / fluid.h * u_f_n_i * psi_f * fluid.ds(1)
            + 0.5 * param.dt / param.M * param.gamma
            / fluid.h * v_f_n_i * phi_f * fluid.ds(1)
            + param.dt / param.M * f(t) * phi_f * fluid.dx)

# Define adjoint variational forms of the fluid subproblem
def a_f_adjoint(z_f, y_f, xi_f, eta_f, fluid, param):

    return (param.nu * dot(grad(eta_f), grad(z_f)) * fluid.dx
           + dot(param.beta, grad(eta_f)) * z_f * fluid.dx
           + dot(grad(xi_f), grad(y_f)) * fluid.dx
           - dot(grad(xi_f), fluid.normal) * y_f * fluid.ds(1)
           - param.nu * dot(grad(eta_f), fluid.normal) * z_f * fluid.ds(1)
           + param.gamma / fluid.h * xi_f * y_f * fluid.ds(1)
           + param.gamma / fluid.h * eta_f * z_f * fluid.ds(1))

def L_f_adjoint_0(z_f_n, y_f_n, z_f_i, y_f_i, z_f_n_i, y_f_n_i,
                  xi_f, eta_f, fluid, param, t):

    return (0.5 * param.dt / param.M * param.nu
            * dot(grad(eta_f), fluid.normal) * z_f_i * fluid.ds(1)
            + 0.5 * param.dt / param.M
            * char_func_J1(param) * eta_f * fluid.dx)

def A_f_adjoint(z_f, y_f, xi_f, eta_f, fluid, param):

    return (eta_f * z_f * fluid.dx
           + 0.5 * param.dt / param.M
           * a_f_adjoint(z_f, y_f, xi_f, eta_f, fluid, param))

def L_f_adjoint(z_f_n, y_f_n, z_f_i, y_f_i, z_f_n_i, y_f_n_i,
                xi_f, eta_f, fluid, param, t):

    return (eta_f * z_f_n * fluid.dx
            - 0.5 * param.dt / param.M
            * a_f_adjoint(z_f_n, y_f_n, xi_f, eta_f, fluid, param)
            + 0.5 * param.dt / param.M * param.nu
            * dot(grad(eta_f), fluid.normal) * z_f_i * fluid.ds(1)
            + 0.5 * param.dt / param.M * param.nu
            * dot(grad(eta_f), fluid.normal) * z_f_n_i * fluid.ds(1)
            + param.dt / param.M * char_func_J1(param) * eta_f * fluid.dx)

# Define variational forms of the solid subproblem
def a_s(u_s, v_s, phi_s, psi_s, solid, param):

    return (param.zeta * dot(grad(u_s), grad(phi_s)) * solid.dx
            + param.delta * dot(grad(v_s), grad(phi_s)) * solid.dx
            - v_s * psi_s * solid.dx
            - param.delta * dot(grad(v_s), solid.normal) * phi_s * solid.ds(1))

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
            - 0.5 * param.dt / param.K * param.nu
            * dot(grad(v_s_i), solid.normal) * phi_s * solid.ds(1)
            - 0.5 * param.dt / param.K * param.nu
            * dot(grad(v_s_n_i), solid.normal) * phi_s * solid.ds(1))

# Define adjoint variational forms of the solid subproblem
def a_s_adjoint(z_s, y_s, xi_s, eta_s, solid, param):

    return (param.zeta * dot(grad(xi_s), grad(z_s)) * solid.dx
           + param.delta * dot(grad(eta_s), grad(z_s)) * solid.dx
           - eta_s * y_s * solid.dx
           - param.delta * dot(grad(eta_s), solid.normal) * z_s * solid.ds(1))

def L_s_adjoint_0(z_s_n, y_s_n, z_s_i, y_s_i, z_s_n_i, y_s_n_i,
                  xi_s, eta_s, solid, param, t):

    return (0.5 * param.dt / param.K * param.gamma
            / solid.h * xi_s * y_s_i * solid.ds(1)
            + 0.5 * param.dt / param.K * param.gamma
            / solid.h * eta_s * z_s_i * solid.ds(1)
            + 0.5 * param.dt / param.K * char_func_J2(param)
            * xi_s * solid.dx)

def A_s_adjoint(z_s, y_s, xi_s, eta_s, solid, param):

    return (eta_s * z_s * solid.dx
            + xi_s * y_s * solid.dx
            + 0.5 * param.dt / param.K
            * a_s_adjoint(z_s, y_s, xi_s, eta_s, solid, param))

def L_s_adjoint(z_s_n, y_s_n, z_s_i, y_s_i, z_s_n_i, y_s_n_i,
                xi_s, eta_s, solid, param, t):

    return (eta_s * z_s_n * solid.dx
            + xi_s * y_s_n * solid.dx
            - 0.5 * param.dt / param.K
            * a_s_adjoint(z_s_n, y_s_n, xi_s, eta_s, solid, param)
            + 0.5 * param.dt / param.K * param.gamma
            / solid.h * xi_s * y_s_i * solid.ds(1)
            + 0.5 * param.dt / param.K * param.gamma
            / solid.h * eta_s * z_s_i * solid.ds(1)
            + 0.5 * param.dt / param.K * param.gamma
            / solid.h * xi_s * y_s_n_i * solid.ds(1)
            + 0.5 * param.dt / param.K * param.gamma
            / solid.h * eta_s * z_s_n_i * solid.ds(1)
            + param.dt / param.K * char_func_J2(param) * xi_s * solid.dx)

