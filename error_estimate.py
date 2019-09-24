
Przejdź do treści
Korzystanie z usługi Gmail z czytnikami ekranu
Wątki
Używasz 3,85 GB (25%) z 15 GB
Zarządzaj
Warunki · Prywatność · Zasady programu
Ostatnia aktywność konta: 0 minut temu
Obecnie używane z 1 innej lokalizacji · Szczegóły

from fenics import project, assemble, dot, grad
from math import log, sqrt
from forms import (a_f, a_s,
                   a_f_adjoint, a_s_adjoint,
                   char_func_J1, char_func_J2)
from parameters import f
from coupling import (fluid_to_solid,
                      solid_to_fluid)

# Copy an array to the same space and in the same direction
def copy_array_right(U, space, param):

    return [U[n].copy(deepcopy = True)
            for n in range(param.N * space.N + 1)]

# Copy an array to the same space and in the opposite direction
def copy_array_left(Z, space, param):

    return [Z[param.N * space.N - 1 - n].copy(deepcopy = True)
            for n in range(param.N * space.N + 1)]

# Extrapolate an array to a different space in the same direction
def extrapolate_array_right(U, space, space_interface,
                            param, transfer_function,
                            subspace_index):
    W = []
    W.append(transfer_function(U[0], space,
                               space_interface,
                               param, subspace_index))
    for i in range(param.N):

        n = i + 1
        for j in range(space.N):

            m = j + 1
            W.append(project((space.N - m)
                             / space.N
                             * transfer_function(U[space_interface.N
                                                   * (n - 1)], space,
                                                 space_interface,
                                                 param, subspace_index)
                             + m / space.N
                             * transfer_function(U[space_interface.N
                                                   * n], space,
                                                 space_interface, param,
                                                 subspace_index),
                             space.V_split[subspace_index]))

    return W

# Extrapolate an array to a different space in the opposite direction
def extrapolate_array_left(Z, space, space_interface,
                           param, transfer_function,
                           subspace_index):
    W = []
    W.append(transfer_function(Z[space_interface.N * param.N], space,
                               space_interface,
                               param, subspace_index))
    for i in range(param.N):

        n = param.N - i - 1
        for j in range(space.N):

            m = j + 1
            W.append(project((space.N - m)
                             / space.N
                             * transfer_function(Z[space_interface.N
                                                   * (n + 1)], space,
                                                 space_interface,
                                                 param, subspace_index)
                             + m / space.N
                             * transfer_function(Z[space_interface.N
                                                   * n], space,
                                                 space_interface,
                                                 param, subspace_index),
                             space.V_split[subspace_index]))

    return W

# Define linear extrapolation
def linear_extrapolation(U, m, t, space, param):

    return ((U[m] - U[m - 1]) * space.N / param.dt * t
           + U[m - 1] * m - U[m] * (m - 1))

# Define reconstruction of the primal problem
def primal_reconstruction(U, m, t, space, param):

    a = ((U[m + 1] - 2 * U[m] + U[m - 1]) * space.N * space.N
         / (2.0 * param.dt * param.dt))
    b = ((-(2 * m - 1) * U[m + 1] + 4 * m * U[m]
          - (2 * m + 1) * U[m - 1]) * space.N / (2.0 * param.dt))
    c = (0.5 * ((m * m - m) * U[m + 1] + (-2 * m * m + 2)
                * U[m] + (m * m + m) * U[m - 1]))

    return a * t * t + b * t + c

def primal_derivative(U, m, t, space, param):

    a = (U[m + 1] - 2 * U[m] + U[m - 1]) * space.N * space.N \
        / (2.0 * param.dt * param.dt)
    b = (-(2 * m - 1) * U[m + 1] + 4 * m * U[m]
         - (2 * m + 1) * U[m - 1]) * space.N / (2.0 * param.dt)

    return 2.0 * a * t + b

# Define reconstruction of the adjoint problem
def t_bar(m, space, param):

    return 0.5 * (m * param.dt / space.N + (m - 1) * param.dt / space.N)

def adjoint_reconstruction(Z, m, t, space, param):

   if m == 0 or m == param.N * space.N:

       return linear_extrapolation(Z, m, t, space, param)

   else:

       return ((t - t_bar(m - 1, space, param))
           / (t_bar(m + 1, space, param)
              - t_bar(m - 1, space, param)) * Z[m + 1]
           + (t - t_bar(m + 1, space, param))
           / (t_bar(m - 1, space, param)
              - t_bar(m + 1, space, param)) * Z[m - 1])


# Define coefficients of 2-point Gaussian quadrature
def gauss(m, space, param):

    return [param.dt / (space.N * 2.0 * sqrt(3)) + t_bar(m, space, param),
            - param.dt / (space.N * 2.0 * sqrt(3)) + t_bar(m, space, param)]

# Compute goal functionals
def J_f(U_f, V_f, fluid, param):

    global_result = 0
    for i in range(param.N * param.M):

        m = i + 1
        result = 0
        print(f'Current contribution: {i}')
        gauss_1, gauss_2 = gauss(m, fluid, param)
        result += (0.5 * param.dt / param.M
                   * char_func_J1(param)
                   * linear_extrapolation(V_f, m, gauss_1, fluid, param)
                   * fluid.dx)
        result += (0.5 * param.dt / param.M
                   * char_func_J1(param)
                   * linear_extrapolation(V_f, m, gauss_2, fluid, param)
                   * fluid.dx)

        global_result += assemble(result)

    return global_result

def J_s(U_s, V_s, solid, param):

    global_result = 0
    for i in range(param.N * param.K):

        m = i + 1
        result = 0
        print(f'Current contribution: {i}')
        gauss_1, gauss_2 = gauss(m, solid, param)
        result += (0.5 * param.dt / param.K
                   * char_func_J2(param)
                   * linear_extrapolation(U_s, m, gauss_1, solid, param)
                   * solid.dx)
        result += (0.5 * param.dt / param.K
                   * char_func_J2(param)
                   * linear_extrapolation(U_s, m, gauss_2, solid, param)
                   * solid.dx)

        global_result += assemble(result)

    return global_result

# Compute primal residual of the fluid subproblem
def B_f(U_f, V_f, U_s, V_s, Z_f, Y_f, fluid, param):

    global_result = 0
    for i in range(param.N * param.M):

        m = i + 1
        result = 0
        print(f'Current contribution: {i}')
        gauss_1, gauss_2 = gauss(m, fluid, param)
        result += (0.5 * param.dt / param.M
                   * ((V_f[m] - V_f[m - 1])
                      * param.M / param.dt
                      * (adjoint_reconstruction(Z_f, m, gauss_1,
                                                fluid, param) - Z_f[m]))
                   * fluid.dx)
        result += (0.5 * param.dt / param.M
                   * ((V_f[m] - V_f[m - 1])
                      * param.M / param.dt
                      * (adjoint_reconstruction(Z_f, m, gauss_2,
                                                fluid, param) - Z_f[m]))
                   * fluid.dx)
        result += (0.5 * param.dt / param.M
                   * a_f(linear_extrapolation(U_f, m, gauss_1, fluid, param),
                         linear_extrapolation(V_f, m, gauss_1, fluid, param),
                         adjoint_reconstruction(Z_f, m, gauss_1, fluid, param)
                         - Z_f[m],
                         adjoint_reconstruction(Y_f, m, gauss_1, fluid, param)
                         - Y_f[m], fluid, param))
        result += (0.5 * param.dt / param.M
                   * a_f(linear_extrapolation(U_f, m, gauss_2, fluid, param),
                         linear_extrapolation(V_f, m, gauss_2, fluid, param),
                         adjoint_reconstruction(Z_f, m, gauss_2, fluid, param)
                         - Z_f[m],
                         adjoint_reconstruction(Y_f, m, gauss_2, fluid, param)
                         - Y_f[m], fluid, param))
        result -= (0.5 * param.dt / param.M
                   * param.gamma / fluid.h
                   * linear_extrapolation(U_s, m, gauss_1, fluid, param)
                   * (adjoint_reconstruction(Y_f, m, gauss_1,
                                             fluid, param) - Y_f[m])
                   * fluid.ds(1))
        result -= (0.5 * param.dt / param.M
                   * param.gamma / fluid.h
                   * linear_extrapolation(U_s, m, gauss_2, fluid, param)
                   * (adjoint_reconstruction(Y_f, m, gauss_2,
                                             fluid, param) - Y_f[m])
                   * fluid.ds(1))
        result -= (0.5 * param.dt / param.M
                   * param.gamma / fluid.h
                   * linear_extrapolation(V_s, m, gauss_1, fluid, param)
                   * (adjoint_reconstruction(Z_f, m, gauss_1,
                                             fluid, param) - Z_f[m])
                   * fluid.ds(1))
        result -= (0.5 * param.dt / param.M
                   * param.gamma / fluid.h
                   * linear_extrapolation(V_s, m, gauss_2, fluid, param)
                   * (adjoint_reconstruction(Z_f, m, gauss_2,
                                             fluid, param) - Z_f[m])
                   * fluid.ds(1))

        global_result += assemble(result)

    return global_result

def F_f(Z_f, Y_f, fluid, param):

    global_result = 0
    for i in range(param.N * param.M):

        m = i + 1
        result = 0
        print(f'Current contribution: {i}')
        gauss_1, gauss_2 = gauss(m, fluid, param)
        result += (0.5 * param.dt / param.M
                   * f(gauss_1)
                   * (adjoint_reconstruction(Z_f, m, gauss_1,
                                             fluid, param) - Z_f[m])
                   * fluid.dx)
        result += (0.5 * param.dt / param.M
                   * f(gauss_2)
                   * (adjoint_reconstruction(Z_f, m, gauss_2,
                                             fluid, param) - Z_f[m])
                   * fluid.dx)

        global_result += assemble(result)

    return global_result

# Compute primal residual of the solid subproblem
def B_s(U_s, V_s, U_f, V_f, Z_s, Y_s, solid, param):

    global_result = 0
    for i in range(param.N * param.K):

        m = i + 1
        result = 0
        print(f'Current contribution: {i}')
        gauss_1, gauss_2 = gauss(m, solid, param)
        result += (0.5 * param.dt / param.K
                   * ((V_s[m] - V_s[m - 1])
                      * param.K / param.dt
                      * (adjoint_reconstruction(Z_s, m, gauss_1,
                                                solid, param) - Z_s[m]))
                   * solid.dx)
        result += (0.5 * param.dt / param.K
                   * ((V_s[m] - V_s[m - 1])
                      * param.K / param.dt
                      * (adjoint_reconstruction(Z_s, m, gauss_2,
                                                solid, param) - Z_s[m]))
                   * solid.dx)
        result += (0.5 * param.dt / param.K
                   * ((U_s[m] - U_s[m - 1])
                      * param.K / param.dt
                      * (adjoint_reconstruction(Y_s, m, gauss_1,
                                                solid, param) - Y_s[m]))
                   * solid.dx)
        result += (0.5 * param.dt / param.K
                   * ((U_s[m] - U_s[m - 1])
                      * param.K / param.dt
                      * (adjoint_reconstruction(Y_s, m, gauss_2,
                                                solid, param) - Y_s[m]))
                   * solid.dx)
        result += (0.5 * param.dt / param.K
                   * a_s(linear_extrapolation(U_s, m, gauss_1, solid, param),
                         linear_extrapolation(V_s, m, gauss_1, solid, param),
                         adjoint_reconstruction(Z_s, m, gauss_1, solid, param)
                         - Z_s[m],
                         adjoint_reconstruction(Y_s, m, gauss_1, solid, param)
                         - Y_s[m], solid, param))
        result += (0.5 * param.dt / param.K
                   * a_s(linear_extrapolation(U_s, m, gauss_2, solid, param),
                         linear_extrapolation(V_s, m, gauss_2, solid, param),
                         adjoint_reconstruction(Z_s, m, gauss_2, solid, param)
                         - Z_s[m],
                         adjoint_reconstruction(Y_s, m, gauss_2, solid, param)
                         - Y_s[m], solid, param))
        result += (0.5 * param.dt / param.K * param.nu
                   * dot(grad(linear_extrapolation(V_f, m, gauss_1,
                                                   solid, param)),
                         solid.normal)
                   * (adjoint_reconstruction(Z_s, m, gauss_1,
                                             solid, param) - Z_s[m])
                   * solid.ds(1))
        result += (0.5 * param.dt / param.K * param.nu
                   * dot(grad(linear_extrapolation(V_f, m, gauss_2,
                                                   solid, param)),
                         solid.normal)
                   * (adjoint_reconstruction(Z_s, m, gauss_2,
                                             solid, param) - Z_s[m])
                   * solid.ds(1))

        global_result += assemble(result)

    return global_result

# Compute adjoint residual of the fluid subproblem
def B_f_adjoint(U_f, V_f, Z_f, Y_f, Z_s, Y_s, fluid, param):

    global_result = 0
    left = True
    for i in range(param.N * param.M):

        m = i + 1
        result = 0
        print(f'Current contribution: {i}')
        gauss_1, gauss_2 = gauss(m, fluid, param)
        if left:

            k = m
            left = False

        else:

            k = m - 1
            left = True

        result += (0.5 * param.dt / param.M
                   * ((primal_derivative(V_f, k, gauss_1, fluid, param)
                       - (V_f[m] - V_f[m - 1]) * param.M / param.dt)
                      * Z_f[m]) * fluid.dx)
        result += (0.5 * param.dt / param.M
                   * ((primal_derivative(V_f, k, gauss_2, fluid, param)
                       - (V_f[m] - V_f[m - 1]) * param.M / param.dt)
                      * Z_f[m]) * fluid.dx)
        result += (0.5 * param.dt / param.M
                   * a_f_adjoint(Z_f[m], Y_f[m],
                                 primal_reconstruction(U_f, k, gauss_1,
                                                       fluid, param)
                                 - linear_extrapolation(U_f, m, gauss_1,
                                                        fluid, param),
                                 primal_reconstruction(V_f, k, gauss_1,
                                                       fluid, param)
                                 - linear_extrapolation(V_f, m, gauss_1,
                                                        fluid, param),
                                 fluid, param))
        result += (0.5 * param.dt / param.M
                   * a_f_adjoint(Z_f[m], Y_f[m],
                                 primal_reconstruction(U_f, k, gauss_2,
                                                       fluid, param)
                                 - linear_extrapolation(U_f, m, gauss_2,
                                                        fluid, param),
                                 primal_reconstruction(V_f, k, gauss_2,
                                                       fluid, param)
                                 - linear_extrapolation(V_f, m, gauss_2,
                                                        fluid, param),
                                 fluid, param))
        result -= (0.5 * param.dt / param.M * param.nu
                   * dot(grad(primal_reconstruction(V_f, k, gauss_1,
                                                    fluid, param)
                              - linear_extrapolation(V_f, k, gauss_1,
                                                     fluid, param)),
                         fluid.normal) * Z_s[m] * fluid.ds(1))
        result -= (0.5 * param.dt / param.M * param.nu
                   * dot(grad(primal_reconstruction(V_f, k, gauss_2,
                                                    fluid, param)
                              - linear_extrapolation(V_f, k, gauss_2,
                                                     fluid, param)),
                         fluid.normal) * Z_s[m] * fluid.ds(1))

        global_result += assemble(result)

    return global_result

def J_f_adjoint(U_f, V_f, fluid, param):

    global_result = 0
    left = True
    for i in range(param.N * param.M):

        m = i + 1
        result = 0
        print(f'Current contribution: {i}')
        gauss_1, gauss_2 = gauss(m, fluid, param)
        if left:

            k = m
            left = False

        else:

            k = m - 1
            left = True

        result += (0.5 * param.dt / param.M
                   * char_func_J1(param)
                   * (primal_reconstruction(V_f, k, gauss_1, fluid, param)
                      - linear_extrapolation(V_f, m, gauss_1, fluid, param))
                   * fluid.dx)
        result += (0.5 * param.dt / param.M
                   * char_func_J1(param)
                   * (primal_reconstruction(V_f, k, gauss_2, fluid, param)
                      - linear_extrapolation(V_f, m, gauss_2, fluid, param))
                   * fluid.dx)

        global_result += assemble(result)

    return global_result

# Compute adjoint residual of the solid subproblem
def B_s_adjoint(U_s, V_s, Z_s, Y_s, Z_f, Y_f, solid, param):

    global_result = 0
    left = True
    for i in range(param.N * param.K):

        m = i + 1
        result = 0
        print(f'Current contribution: {i}')
        gauss_1, gauss_2 = gauss(m, solid, param)
        if left:

            k = m
            left = False

        else:

            k = m - 1
            left = True

        result += (0.5 * param.dt / param.K
                   * ((primal_derivative(V_s, k, gauss_1, solid, param)
                       - (V_s[m] - V_s[m - 1]) * param.K / param.dt)
                      * Z_s[m]) * solid.dx)
        result += (0.5 * param.dt / param.K
                   * ((primal_derivative(V_s, k, gauss_2, solid, param)
                       - (V_s[m] - V_s[m - 1]) * param.K / param.dt)
                      * Z_s[m]) * solid.dx)
        result += (0.5 * param.dt / param.K
                   * ((primal_derivative(U_s, k, gauss_1, solid, param)
                       - (U_s[m] - U_s[m - 1]) * param.K / param.dt)
                      * Y_s[m]) * solid.dx)
        result += (0.5 * param.dt / param.K
                   * ((primal_derivative(U_s, k, gauss_2, solid, param)
                       - (U_s[m] - U_s[m - 1]) * param.K / param.dt)
                      * Y_s[m]) * solid.dx)
        result += (0.5 * param.dt / param.K
                   * a_s_adjoint(Z_s[m], Y_s[m],
                                 primal_reconstruction(U_s, k, gauss_1,
                                                       solid, param)
                                 - linear_extrapolation(U_s, m, gauss_1,
                                                        solid, param),
                                 primal_reconstruction(V_s, k, gauss_1,
                                                       solid, param)
                                 - linear_extrapolation(V_s, m, gauss_1,
                                                        solid, param),
                                 solid, param))
        result += (0.5 * param.dt / param.K
                   * a_s_adjoint(Z_s[m], Y_s[m],
                                 primal_reconstruction(U_s, k, gauss_2,
                                                       solid, param)
                                 - linear_extrapolation(U_s, m, gauss_2,
                                                        solid, param),
                                 primal_reconstruction(V_s, k, gauss_2,
                                                       solid, param)
                                 - linear_extrapolation(V_s, m, gauss_2,
                                                        solid, param),
                                 solid, param))
        result -= (0.5 * param.dt / param.K
                   * param.gamma / solid.h
                   * (primal_reconstruction(U_s, k, gauss_1, solid, param)
                      - linear_extrapolation(U_s, m, gauss_1, solid, param))
                   * Y_f[m] * solid.ds(1))
        result -= (0.5 * param.dt / param.K
                   * param.gamma / solid.h
                   * (primal_reconstruction(U_s, k, gauss_2, solid, param)
                      - linear_extrapolation(U_s, m, gauss_2, solid, param))
                   * Y_f[m] * solid.ds(1))
        result -= (0.5 * param.dt / param.K
                   * param.gamma / solid.h
                   * (primal_reconstruction(V_s, k, gauss_1, solid, param)
                      - linear_extrapolation(V_s, m, gauss_1, solid, param))
                   * Z_f[m] * solid.ds(1))
        result -= (0.5 * param.dt / param.K
                   * param.gamma / solid.h
                   * (primal_reconstruction(V_s, k, gauss_2, solid, param)
                      - linear_extrapolation(V_s, m, gauss_2, solid, param))
                   * Z_f[m] * solid.ds(1))

        global_result += assemble(result)

    return global_result

def J_s_adjoint(U_s, V_s, solid, param):

    global_result = 0
    left = True
    for i in range(param.N * param.K):

        m = i + 1
        result = 0
        print(f'Current contribution: {i}')
        gauss_1, gauss_2 = gauss(m, solid, param)
        if left:

            k = m
            left = False

        else:

            k = m - 1
            left = True

        result += (0.5 * param.dt / param.K
                   * char_func_J2(param)
                   * (primal_reconstruction(U_s, k, gauss_1, solid, param)
                      - linear_extrapolation(U_s, m, gauss_1, solid, param))
                   * solid.dx)
        result += (0.5 * param.dt / param.K
                   * char_func_J2(param)
                   * (primal_reconstruction(U_s, k, gauss_2, solid, param)
                      - linear_extrapolation(U_s, m, gauss_2, solid, param))
                   * solid.dx)

        global_result += assemble(result)

    return global_result

def compute_residuals(u_f, v_f, u_s, v_s, z_f, y_f,
                      z_s, y_s, fluid, solid, param):

    # Load solutions of the fluid subproblem problem
    for i in range(param.M * param.N + 1):

        u_f.load(i)
        v_f.load(i)
        z_f.load(i)
        y_f.load(i)

    # Load solutions of the solid subproblem problem
    for i in range(param.K * param.N + 1):

        u_s.load(i)
        v_s.load(i)
        z_s.load(i)
        y_s.load(i)

    # Create text file
    residuals_txt = open('residuals.txt', 'a')

    # Prepare arrays of solutions
    U_f_f = copy_array_right(u_f.array, fluid, param)
    U_f_s = extrapolate_array_right(u_f.array, solid, fluid,
                                    param, fluid_to_solid, 0)
    V_f_f = copy_array_right(v_f.array, fluid, param)
    V_f_s = extrapolate_array_right(v_f.array, solid, fluid,
                                    param, fluid_to_solid, 1)
    U_s_s = copy_array_right(u_s.array, solid, param)
    U_s_f = extrapolate_array_right(u_s.array, fluid, solid,
                                    param, solid_to_fluid, 0)
    V_s_s = copy_array_right(v_s.array, solid, param)
    V_s_f = extrapolate_array_right(v_s.array, fluid, solid,
                                    param, solid_to_fluid, 1)
    Z_f_f = copy_array_left(z_f.array, fluid, param)
    Z_f_s = extrapolate_array_left(z_f.array, solid, fluid,
                                   param, fluid_to_solid, 0)
    Y_f_f = copy_array_left(y_f.array, fluid, param)
    Y_f_s = extrapolate_array_left(y_f.array, solid, fluid,
                                   param, fluid_to_solid, 1)
    Z_s_s = copy_array_left(z_s.array, solid, param)
    Z_s_f = extrapolate_array_left(z_s.array, fluid, solid,
                                   param, solid_to_fluid, 0)
    Y_s_s = copy_array_left(y_s.array, solid, param)
    Y_s_f = extrapolate_array_left(y_s.array, fluid, solid,
                                   param, solid_to_fluid, 1)

    # Compute residuals
    primal_fluid_lhs = F_f(Z_f_f, Y_f_f, fluid, param)
    primal_fluid_rhs = B_f(U_f_f, V_f_f, U_s_f, V_s_f, Z_f_f, Y_f_f, fluid, param)
    primal_residual_fluid = (0.5 * primal_fluid_lhs
                             - 0.5 * primal_fluid_rhs)
    print(f'Primal residual for the fluid subproblem: '
          f'{primal_residual_fluid}')
    residuals_txt.write(f'Primal residual for the fluid subproblem: '
                        f'{primal_residual_fluid} \r\n')
    primal_residual_solid = - 0.5 * B_s(U_s_s, V_s_s, U_f_s, V_f_s,
                                        Z_s_s, Y_s_s, solid, param)
    print(f'Primal residual for the solid subproblem: '
          f'{primal_residual_solid}')
    residuals_txt.write(f'Primal residual for the solid subproblem: '
                        f'{primal_residual_solid} \r\n')
    adjoint_fluid_lhs = J_f_adjoint(U_f_f, V_f_f, fluid, param)
    adjoint_fluid_rhs = B_f_adjoint(U_f_f, V_f_f, Z_f_f, Y_f_f, Z_s_f, Y_s_f, fluid, param)
    adjoint_residual_fluid = (0.5 * adjoint_fluid_lhs
                              - 0.5 * adjoint_fluid_rhs)
    print(f'Adjoint residual for the fluid subproblem: '
          f'{adjoint_residual_fluid}')
    residuals_txt.write(f'Adjoint residual for the fluid subproblem: '
                        f'{adjoint_residual_fluid} \r\n')
    adjoint_solid_lhs = J_s_adjoint(U_s_s, V_s_s, solid, param)
    adjoint_solid_rhs = B_s_adjoint(U_s_s, V_s_s, Z_s_s, Y_s_s, Z_f_s, Y_f_s, solid, param)
    adjoint_residual_solid = (0.5 * adjoint_solid_lhs
                              - 0.5 * adjoint_solid_rhs)
    print(f'Adjoint residual for the solid subproblem: '
          f'{adjoint_residual_solid}')
    residuals_txt.write(f'Adjoint residual for the solid subproblem: '
                        f'{adjoint_residual_solid} \r\n')

    # Compute goal functional
    if param.J1:

        goal_functional = J_f(U_f_f, V_f_f, fluid, param)

    else:

        goal_functional = J_s(U_s_s, V_s_s, solid, param)

    print(f'Value of goal functional: {goal_functional}')
    residuals_txt.write(f'Value of goal functional: {goal_functional} \r\n')

    residuals_txt.close()

    return [primal_residual_fluid,
            primal_residual_solid,
            adjoint_residual_fluid,
            adjoint_residual_solid,
            goal_functional]
