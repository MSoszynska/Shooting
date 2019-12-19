from math import log

fluid_case = True
solid_case = False
uniform_equal = False
uniform_refined = False
adaptive = True

primal_residual_fluid = []
primal_residual_solid = []
adjoint_residual_fluid = []
adjoint_residual_solid = []
functional = []

if fluid_case and uniform_equal:
    primal_residual_fluid.append(2.1196691528455495e-06)
    primal_residual_fluid.append(7.468539511310226e-08)
    primal_residual_fluid.append(8.911081546902013e-09)
    primal_residual_fluid.append(2.1811509634202593e-09)

    primal_residual_solid.append(5.093083953543579e-13)
    primal_residual_solid.append(1.3827824252296961e-13)
    primal_residual_solid.append(3.608077931421517e-14)
    primal_residual_solid.append(8.776334513830535e-15)

    adjoint_residual_fluid.append(5.727920354599756e-06)
    adjoint_residual_fluid.append(1.2519319592704365e-07)
    adjoint_residual_fluid.append(1.47576454757463e-08)
    adjoint_residual_fluid.append(2.883640732255314e-09)

    adjoint_residual_solid.append(1.2947402029887962e-13)
    adjoint_residual_solid.append(-1.19762605285083e-14)
    adjoint_residual_solid.append(-1.0206052400360077e-15)
    adjoint_residual_solid.append(-3.3230454987013956e-16)


if fluid_case and uniform_refined:
    primal_residual_fluid.append(7.468491440583642e-08)
    primal_residual_fluid.append(8.911023989755842e-09)
    primal_residual_fluid.append(2.1811440988863484e-09)
    primal_residual_fluid.append(5.871586667236872e-10)

    primal_residual_solid.append(5.020745070605293e-13)
    primal_residual_solid.append(1.3503726426260264e-13)
    primal_residual_solid.append(3.48639287405699e-14)
    primal_residual_solid.append(8.597974664749264e-15)

    adjoint_residual_fluid.append(1.2519268767549449e-07)
    adjoint_residual_fluid.append(1.4757586128935083e-08)
    adjoint_residual_fluid.append(2.8836363869397465e-09)
    adjoint_residual_fluid.append(6.732171675128327e-10)

    adjoint_residual_solid.append(2.5950904717399063e-14)
    adjoint_residual_solid.append(-1.5872123289212083e-14)
    adjoint_residual_solid.append(-2.499311483893612e-15)
    adjoint_residual_solid.append(-5.398912894181489e-16)

if fluid_case and adaptive:
    primal_residual_fluid.append(-1.1217410909730264e-08)
    primal_residual_fluid.append(-4.1726175646372795e-09)
    primal_residual_fluid.append(-2.6576929081880357e-09)
    primal_residual_fluid.append(-8.064424930677801e-10)

    primal_residual_solid.append(5.052288042832065e-13)
    primal_residual_solid.append(4.847511001397291e-13)
    primal_residual_solid.append(4.706403619882383e-13)
    primal_residual_solid.append(4.64807223077798e-13)

    adjoint_residual_fluid.append(2.6009397791042288e-08)
    adjoint_residual_fluid.append(3.6688847491726673e-10)
    adjoint_residual_fluid.append(-2.0996660409471158e-09)
    adjoint_residual_fluid.append(-7.349314097298049e-10)

    adjoint_residual_solid.append(5.576085726718018e-13)
    adjoint_residual_solid.append(2.433376171021284e-13)
    adjoint_residual_solid.append(2.97416190390574e-13)
    adjoint_residual_solid.append(7.907447522174032e-13)

if fluid_case and uniform_equal:
    functional.append(0.0006002078985818325)
    functional.append(0.0006024784870824304)
    functional.append(0.0006026119921740284)
    functional.append(0.0006026315376227036)

if fluid_case and uniform_refined:
    functional.append(0.0006024784870958133)
    functional.append(0.0006026119921853403)
    functional.append(0.0006026315376239955)
    functional.append(0.0006026357100514537)

if fluid_case and adaptive:
    functional.append(0.0006024784870958133)
    functional.append(0.0006026119921853403)
    functional.append(0.0006026315376239955)
    functional.append(0.0006026357100514537)

# Perform extrapolation
print("Extrapolation")
J_exact = 0.0
J = functional
for i in range(len(J) - 2):

    print(f"Extrapolation of J{i + 1}, J{i + 2}, J{i + 3}")
    q = -log(abs((J[i + 1] - J[i + 2]) / (J[i] - J[i + 1]))) / log(2.0)
    print(f"Extrapolated order of convergence: {q}")
    C = pow(J[i] - J[i + 1], 2) / (J[i] - 2.0 * J[i + 1] + J[i + 2])
    print(f"Extrapolated constant: {C}")
    J_exact = (J[i] * J[i + 2] - J[i + 1] * J[i + 1]) / (
        J[i] - 2.0 * J[i + 1] + J[i + 2]
    )
    print(f"Extrapolated exact value of goal functional: {J_exact}")

# Compute effectiveness
print("Effectivity")
for i in range(len(J)):

    print(f"Effectivity of J{i + 1}")
    residual = (
        primal_residual_fluid[i]
        + primal_residual_solid[i]
        + adjoint_residual_fluid[i]
        + adjoint_residual_solid[i]
    )
    print(f"Overall residual: {residual}")
    effectivity = residual / (J_exact - J[i])
    print(f"Effectivity: {effectivity}")
