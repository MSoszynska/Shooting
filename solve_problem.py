Wątki
Używasz 3,85 GB (25%) z 15 GB
Zarządzaj
Warunki · Prywatność · Zasady programu
Ostatnia aktywność konta: 0 minut temu
Obecnie używane z 1 innej lokalizacji · Szczegóły

from fenics import (Function, FunctionSpace, project,
                    DirichletBC, Constant, TrialFunction, split,
                    TestFunction, solve)
from spaces import boundary

# Define a function solving a problem on a subdomain
def solve_problem(u, v, u_interface, v_interface,
                  space, space_interface, transfer_function,
                  L_0, A, L, first_time_step,
                  param, t, save = False):

    # Store old solutions
    u_n = Function(space.V_split[0])
    v_n = Function(space.V_split[1])
    u_n.assign(u.old)
    v_n.assign(v.old)

    # Store old interface values
    u_n_i = Function(space.V_split[0])
    v_n_i = Function(space.V_split[1])
    u_n_i.assign(u.i_old)
    v_n_i.assign(v.i_old)

    # Initialize new interface values
    u_i = Function(space.V_split[0])
    v_i = Function(space.V_split[1])

    # Define Dirichlet boundary conditions
    bc_u = DirichletBC(space.V.sub(0), Constant(0.0), boundary)
    bc_v = DirichletBC(space.V.sub(1), Constant(0.0), boundary)
    bcs = [bc_u, bc_v]

    # Compute fractional time-steps for fluid problem
    for n in range(space.N):

         # Extrapolate weak boundary conditions on the interface
         u_i.assign(project((space.N - n - 1.0)
                           / space.N
                           * u.i_old + (n + 1.0)
                           / space.N
                           * transfer_function(u_interface.new,
                                               space, space_interface,
                                               param, 0), space.V_split[0]))
         v_i.assign(project((space.N - n - 1.0)
                           / space.N
                           * v.i_old + (n + 1.0)
                           / space.N
                           * transfer_function(v_interface.new,
                                               space, space_interface,
                                               param, 1), space.V_split[1]))

         # Define trial and test functions
         U_new = TrialFunction(space.V)
         (u_new, v_new) = split(U_new)
         Phi = TestFunction(space.V)
         (phi, psi) = split(Phi)

         # Define scheme
         a = A(u_new, v_new, phi, psi, space, param)
         if first_time_step and n == 0:

             l = L_0(u_n, v_n, u_i, v_i, u_n_i, v_n_i,
                     phi, psi, space, param, t)

         else:

             l = L(u_n, v_n, u_i, v_i, u_n_i, v_n_i,
                   phi, psi, space, param, t)

         # Solve fluid problem
         U_new = Function(space.V)
         solve(a == l, U_new, bcs)
         (u_new, v_new) = U_new.split(U_new)

         # Save solutions
         if save:

             u.save(u_new)
             v.save(v_new)

         # Update fluid solution
         u_n.assign(u_new)
         v_n.assign(v_new)

         # Update boundary conditions
         u_n_i.assign(project(u_i, space.V_split[0]))
         v_n_i.assign(project(v_i, space.V_split[1]))

    # Save final values
    u.new.assign(u_new)
    v.new.assign(v_new)
    u.i_new.assign(u_i)
    v.i_new.assign(v_i)

    return
