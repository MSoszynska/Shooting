from math import log

fluid_case = False
solid_case = True
uniform_equal = False
uniform_refined = False
adaptive = True

primal_residual_fluid = []
primal_residual_solid = []
adjoint_residual_fluid = []
adjoint_residual_solid = []
functional_extrapolation = []
functional = []

# if fluid_case:
#     functional_extrapolation.append(0.2076582826968302)
#     functional_extrapolation.append(0.2085227067564289)
#     functional_extrapolation.append(0.2086565845756113)
#     functional_extrapolation.append(0.208683263058678)
#
# if solid_case:
#     functional_extrapolation.append(0.02197030348452757)
#     functional_extrapolation.append(0.022119869111773962)
#     functional_extrapolation.append(0.02215146520085885)
#     functional_extrapolation.append(0.02215915123717612)
#
# if fluid_case and uniform_equal:
#     primal_residual_fluid.append(-0.00017077390004443098)
#     primal_residual_fluid.append(-4.5032110544606216e-05)
#     primal_residual_fluid.append(-1.1478404461708074e-05)
#     primal_residual_fluid.append(-2.9092799521418763e-06)
#
#     primal_residual_solid.append(0.0)
#     primal_residual_solid.append(0.0)
#     primal_residual_solid.append(0.0)
#     primal_residual_solid.append(0.0)
#
#     adjoint_residual_fluid.append(0.0005321824422173875)
#     adjoint_residual_fluid.append(0.00010569115645451979)
#     adjoint_residual_fluid.append(2.6646146586506326e-05)
#     adjoint_residual_fluid.append(6.667356736997568e-06)
#
#     adjoint_residual_solid.append(0.0)
#     adjoint_residual_solid.append(0.0)
#     adjoint_residual_solid.append(0.0)
#     adjoint_residual_solid.append(0.0)
#
#     functional.append(0.2076582826968302)
#     functional.append(0.2085227067564289)
#     functional.append(0.2086565845756113)
#     functional.append(0.208683263058678)
#
# if fluid_case and uniform_refined:
#     primal_residual_fluid.append(-4.7025318164528636e-05)
#     primal_residual_fluid.append(-1.1397145133545731e-05)
#     primal_residual_fluid.append(-2.8455108483795047e-06)
#     primal_residual_fluid.append(-7.182883423279761e-07)
#
#     primal_residual_solid.append(0.0)
#     primal_residual_solid.append(0.0)
#     primal_residual_solid.append(0.0)
#     primal_residual_solid.append(0.0)
#
#     adjoint_residual_fluid.append(0.00018832489438274486)
#     adjoint_residual_fluid.append(5.502162842672001e-05)
#     adjoint_residual_fluid.append(1.9217467498682743e-05)
#     adjoint_residual_fluid.append(7.773931098865304e-06)
#
#     adjoint_residual_solid.append(0.0)
#     adjoint_residual_solid.append(0.0)
#     adjoint_residual_solid.append(0.0)
#     adjoint_residual_solid.append(0.0)
#
#     functional.append(0.2111584815728643)
#     functional.append(0.2099366441210019)
#     functional.append(0.20929449431627836)
#     functional.append(0.2089886856007165)
#
# if fluid_case and adaptive:
#     primal_residual_fluid.append(-5.1580055019207165e-05)
#     primal_residual_fluid.append(-2.7324539992034087e-05)
#     primal_residual_fluid.append(-1.1891093033959517e-05)
#     primal_residual_fluid.append(-6.611309629801052e-06)
#
#     primal_residual_solid.append(0.0)
#     primal_residual_solid.append(0.0)
#     primal_residual_solid.append(0.0)
#     primal_residual_solid.append(0.0)
#
#     adjoint_residual_fluid.append(0.0001230778291733138)
#     adjoint_residual_fluid.append(0.00016932626646357852)
#     adjoint_residual_fluid.append(0.00010546370898358846)
#     adjoint_residual_fluid.append(9.795691479209844e-05)
#
#     adjoint_residual_solid.append(0.0)
#     adjoint_residual_solid.append(0.0)
#     adjoint_residual_solid.append(0.0)
#     adjoint_residual_solid.append(0.0)
#
#     functional.append(0.20972888721698954)
#     functional.append(0.2112334812701519)
#     functional.append(0.21235394821268205)
#     functional.append(0.21351054715281098)
#
# if solid_case and uniform_equal:
#     primal_residual_fluid.append(0.0)
#     primal_residual_fluid.append(0.0)
#     primal_residual_fluid.append(0.0)
#     primal_residual_fluid.append(0.0)
#
#     primal_residual_solid.append(-0.00027387545053833887)
#     primal_residual_solid.append(-6.737637490030645e-05)
#     primal_residual_solid.append(-1.6842889788959136e-05)
#     primal_residual_solid.append(-4.2136908079988265e-06)
#
#     adjoint_residual_fluid.append(0.0)
#     adjoint_residual_fluid.append(0.0)
#     adjoint_residual_fluid.append(0.0)
#     adjoint_residual_fluid.append(0.0)
#
#     adjoint_residual_solid.append(-7.027598307073391e-06)
#     adjoint_residual_solid.append(2.8658300246439015e-06)
#     adjoint_residual_solid.append(9.565198126932793e-07)
#     adjoint_residual_solid.append(2.5080866378758513e-07)
#
#     functional.append(0.02197030348452757)
#     functional.append(0.022119869111773962)
#     functional.append(0.02215146520085885)
#     functional.append(0.02215915123717612)
#
# if solid_case and uniform_refined:
#     primal_residual_fluid.append(0.0)
#     primal_residual_fluid.append(0.0)
#     primal_residual_fluid.append(0.0)
#     primal_residual_fluid.append(0.0)
#
#     primal_residual_solid.append(-6.908520275685951e-05)
#     primal_residual_solid.append(-1.6947992690730333e-05)
#     primal_residual_solid.append(-4.220284890742198e-06)
#     primal_residual_solid.append(-1.0541249305516516e-06)
#
#     adjoint_residual_fluid.append(0.0)
#     adjoint_residual_fluid.append(0.0)
#     adjoint_residual_fluid.append(0.0)
#     adjoint_residual_fluid.append(0.0)
#
#     adjoint_residual_solid.append(2.4144631243690688e-06)
#     adjoint_residual_solid.append(9.275523360560562e-07)
#     adjoint_residual_solid.append(2.489965719156664e-07)
#     adjoint_residual_solid.append(6.329785824388669e-08)
#
#     functional.append(0.022378731430627012)
#     functional.append(0.022216091052090423)
#     functional.append(0.022175291082608316)
#     functional.append(0.02216509461477297)
#
# if solid_case and adaptive:
#     primal_residual_fluid.append(0.0)
#     primal_residual_fluid.append(0.0)
#     primal_residual_fluid.append(0.0)
#     primal_residual_fluid.append(0.0)
#
#     primal_residual_solid.append(-6.571545859843094e-05)
#     primal_residual_solid.append(-1.59435276306466e-05)
#     primal_residual_solid.append(-4.9939861081049025e-06)
#     primal_residual_solid.append(-1.885288086357069e-06)
#
#     adjoint_residual_fluid.append(0.0)
#     adjoint_residual_fluid.append(0.0)
#     adjoint_residual_fluid.append(0.0)
#     adjoint_residual_fluid.append(0.0)
#
#     adjoint_residual_solid.append(4.886982528615656e-06)
#     adjoint_residual_solid.append(2.241548711517795e-06)
#     adjoint_residual_solid.append(2.6631356082680734e-07)
#     adjoint_residual_solid.append(2.4748401200026135e-07)
#
#     functional.append(0.02229894608184633)
#     functional.append(0.022374648544784884)
#     functional.append(0.022425958385764114)
#     functional.append(0.022463183826915187)

if fluid_case:
    functional_extrapolation.append(0.03190702346164951)
    functional_extrapolation.append(0.03194268700840532)
    functional_extrapolation.append(0.03194772394641445)
    functional_extrapolation.append(0.03194879651648573)

if solid_case:
    functional_extrapolation.append(0.0014687150123243614)
    functional_extrapolation.append(0.0014865546194304643)
    functional_extrapolation.append(0.0014906346036546987)
    functional_extrapolation.append(0.0014916334079003476)

if fluid_case and uniform_equal:
    print("Fluid non-refined case")
    primal_residual_fluid.append(-9.325024435650555e-06)
    primal_residual_fluid.append(-2.4530504207245808e-06)
    primal_residual_fluid.append(-6.298636940854178e-07)
    primal_residual_fluid.append(-1.5991010863895923e-07)

    primal_residual_solid.append(0.0)
    primal_residual_solid.append(0.0)
    primal_residual_solid.append(0.0)
    primal_residual_solid.append(0.0)

    adjoint_residual_fluid.append(3.45809813970592e-05)
    adjoint_residual_fluid.append(1.7595618770662394e-06)
    adjoint_residual_fluid.append(4.4148181246216983e-07)
    adjoint_residual_fluid.append(1.1033882733987005e-07)

    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)

    functional.append(0.03190702346164951)
    functional.append(0.03194268700840532)
    functional.append(0.03194772394641445)
    functional.append(0.03194879651648573)

if fluid_case and uniform_refined:
    print("Fluid refined case")
    primal_residual_fluid.append(-2.4146429321094778e-06)
    primal_residual_fluid.append(-6.231572871323567e-07)
    primal_residual_fluid.append(-1.5836810562320072e-07)
    primal_residual_fluid.append(-4.0030591032100545e-08)

    primal_residual_solid.append(0.0)
    primal_residual_solid.append(0.0)
    primal_residual_solid.append(0.0)
    primal_residual_solid.append(0.0)

    adjoint_residual_fluid.append(4.748852232679432e-06)
    adjoint_residual_fluid.append(1.6941538032675405e-06)
    adjoint_residual_fluid.append(7.039919315617237e-07)
    adjoint_residual_fluid.append(3.2247129642866505e-07)

    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)

    functional.append(0.0319342828246747)
    functional.append(0.03195382396428161)
    functional.append(0.03195280137279488)
    functional.append(0.031951247031023616)

if fluid_case and adaptive:
    print("Fluid adaptive case")
    primal_residual_fluid.append(-5.1580055019207165e-05)
    primal_residual_fluid.append(-2.7324539992034087e-05)
    primal_residual_fluid.append(-1.1891093033959517e-05)
    primal_residual_fluid.append(-6.611309629801052e-06)

    primal_residual_solid.append(0.0)
    primal_residual_solid.append(0.0)
    primal_residual_solid.append(0.0)
    primal_residual_solid.append(0.0)

    adjoint_residual_fluid.append(0.0001230778291733138)
    adjoint_residual_fluid.append(0.00016932626646357852)
    adjoint_residual_fluid.append(0.00010546370898358846)
    adjoint_residual_fluid.append(9.795691479209844e-05)

    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)

    functional.append(0.20972888721698954)
    functional.append(0.2112334812701519)
    functional.append(0.21235394821268205)
    functional.append(0.21351054715281098)

if solid_case and uniform_equal:
    print("Solid non-refined case")
    primal_residual_fluid.append(0.0)
    primal_residual_fluid.append(0.0)
    primal_residual_fluid.append(0.0)
    primal_residual_fluid.append(0.0)

    primal_residual_solid.append(-2.0910193912800207e-05)
    primal_residual_solid.append(-5.2879682198091325e-06)
    primal_residual_solid.append(-1.3213998921472268e-06)
    primal_residual_solid.append(-3.303005516699729e-07)

    adjoint_residual_fluid.append(0.0)
    adjoint_residual_fluid.append(0.0)
    adjoint_residual_fluid.append(0.0)
    adjoint_residual_fluid.append(0.0)

    adjoint_residual_solid.append(-2.201581499791791e-06)
    adjoint_residual_solid.append(-3.301883845358055e-07)
    adjoint_residual_solid.append(-6.404590785854852e-08)
    adjoint_residual_solid.append(-1.4861035249202046e-08)

    functional.append(0.0014687150123243614)
    functional.append(0.0014865546194304643)
    functional.append(0.0014906346036546987)
    functional.append(0.0014916334079003476)

if solid_case and uniform_refined:
    print("Solid refined case")
    primal_residual_fluid.append(0.0)
    primal_residual_fluid.append(0.0)
    primal_residual_fluid.append(0.0)
    primal_residual_fluid.append(0.0)

    primal_residual_solid.append(-5.329144880307317e-06)
    primal_residual_solid.append(-1.3239994412622998e-06)
    primal_residual_solid.append(-3.3046338142041495e-07)
    primal_residual_solid.append(-1.0541249305516516e-06)

    adjoint_residual_fluid.append(0.0)
    adjoint_residual_fluid.append(0.0)
    adjoint_residual_fluid.append(0.0)
    adjoint_residual_fluid.append(0.0)

    adjoint_residual_solid.append(-3.391450490213901e-07)
    adjoint_residual_solid.append(-6.463051804133927e-08)
    adjoint_residual_solid.append( -1.489782810581401e-08)
    adjoint_residual_solid.append(6.329785824388669e-08)

    functional.append(0.0014939675373538508)
    functional.append(0.001492490142438691)
    functional.append(0.0014920972331508092)
    functional.append(0.02216509461477297)

if solid_case and adaptive:
    print("Solid adaptive case")
    primal_residual_fluid.append(0.0)
    primal_residual_fluid.append(0.0)
    primal_residual_fluid.append(0.0)
    primal_residual_fluid.append(0.0)

    primal_residual_solid.append(-4.970860915152709e-06)
    primal_residual_solid.append(-1.1725794284172528e-06)
    primal_residual_solid.append(-3.2591666931943363e-07)
    primal_residual_solid.append(-9.443224159075162e-08)

    adjoint_residual_fluid.append(0.0)
    adjoint_residual_fluid.append(0.0)
    adjoint_residual_fluid.append(0.0)
    adjoint_residual_fluid.append(0.0)

    adjoint_residual_solid.append(-2.9880355539958364e-07)
    adjoint_residual_solid.append(-3.595170806763628e-08)
    adjoint_residual_solid.append(1.1094854183549938e-08)
    adjoint_residual_solid.append(1.2589765034455628e-08)

    functional.append(0.0014786546357436017)
    functional.append(0.0014945006558526963)
    functional.append(0.0014990625672142252)
    functional.append(0.00150066295138089)

# Perform extrapolation
print("Extrapolation")
J_exact = 0.0
J = functional_extrapolation
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
J = functional
for i in range(len(J)):

    print(f"Effectivity of J{i + 1}")
    residual = (
        primal_residual_fluid[i]
        + primal_residual_solid[i]
        + adjoint_residual_fluid[i]
        + adjoint_residual_solid[i]
    )
    print(f"Overall residual: {residual}")
    print(f"Extrapolated error: {J_exact - J[i]}")
    effectivity = residual / (J_exact - J[i])
    print(f"Effectivity: {effectivity}")
