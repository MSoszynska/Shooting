from fenics import dot, grad, Expression, project, Constant
from math import sqrt
from parameters import Parameters, external_force
from spaces import Space
from time_structure import MicroTimeStep

# Define characteristic functions corresponding to chosen functionals
def characteristic_function_fluid(param: Parameters):

    return Expression(
        "2.0 <= x[0] && 0.0 <= x[1] ? functional : 0.0",
        functional=param.GOAL_FUNCTIONAL_FLUID,
        degree=0,
    )

def characteristic_function_solid(param: Parameters):

    return Expression(
        "2.0 <= x[0] && x[1] <= 0.0 ? functional : 0.0",
        functional=param.GOAL_FUNCTIONAL_SOLID,
        degree=0,
    )


# Define coefficients of 2-point Gaussian quadrature
def gauss(microtimestep, microtimestep_before=None):

    t_new = microtimestep.point
    if microtimestep_before is None:
        t_old = microtimestep.before.point
        dt = microtimestep.before.dt
    else:
        t_old = microtimestep_before.point
        dt = microtimestep.point - microtimestep_before.point
    t_average = 0.5 * (t_old + t_new)
    return [
        dt / (2.0 * sqrt(3)) + t_average,
        -dt / (2.0 * sqrt(3)) + t_average,
    ]


# Define goal functional
def goal_functional(
    microtimestep_before: MicroTimeStep,
    microtimestep: MicroTimeStep,
    function_name,
):

    function_before = microtimestep_before.functions[function_name]
    time_before = microtimestep_before.point
    function = microtimestep.functions[function_name]
    time = microtimestep.point

    def linear_extrapolation(time_gauss):
        return (grad(function) - grad(function_before)) / (
            time - time_before
        ) * time_gauss + (
            grad(function_before) * time - grad(function) * time_before
        ) / (
            time - time_before
        )

    time_gauss_1, time_gauss_2 = gauss(microtimestep, microtimestep_before)

    return [
        0.5 * linear_extrapolation(time_gauss_1) * (-time_gauss_1 + time)
        + 0.5 * linear_extrapolation(time_gauss_2) * (-time_gauss_2 + time),
        0.5 * linear_extrapolation(time_gauss_1) * (time_gauss_1 - time_before)
        + 0.5
        * linear_extrapolation(time_gauss_2)
        * (time_gauss_2 - time_before),
    ]


# Define variational forms of the fluid subproblem
def form_fluid(
    displacement_fluid,
    velocity_fluid,
    phi_fluid,
    psi_fluid,
    fluid: Space,
    param: Parameters,
):

    return (
        param.NU * dot(grad(velocity_fluid), grad(phi_fluid)) * fluid.dx
    )


def bilinear_form_fluid(
    displacement_fluid,
    velocity_fluid,
    phi_fluid,
    psi_fluid,
    fluid: Space,
    param: Parameters,
    time_step,
):

    return (
        velocity_fluid * phi_fluid * fluid.dx
        + 0.5
        * time_step
        * form_fluid(
            displacement_fluid,
            velocity_fluid,
            phi_fluid,
            psi_fluid,
            fluid,
            param,
        )
    )


def functional_fluid(
    displacement_fluid_old,
    velocity_fluid_old,
    displacement_fluid_interface,
    velocity_fluid_interface,
    displacement_fluid_old_interface,
    velocity_fluid_old_interface,
    phi_fluid,
    psi_fluid,
    fluid: Space,
    param: Parameters,
    time,
    time_step,
):

    return (
        velocity_fluid_old * phi_fluid * fluid.dx
        - 0.5
        * time_step
        * form_fluid(
            displacement_fluid_old,
            velocity_fluid_old,
            phi_fluid,
            psi_fluid,
            fluid,
            param,
        )
        + time_step * external_force(time) * phi_fluid * fluid.dx
    )


# Define adjoint variational forms of the fluid subproblem
def form_fluid_adjoint(
    displacement_fluid_adjoint,
    velocity_fluid_adjoint,
    xi_fluid,
    eta_fluid,
    fluid: Space,
    param: Parameters,
):

    return (
        param.NU
        * dot(grad(eta_fluid), grad(displacement_fluid_adjoint))
        * fluid.dx
    )


def functional_fluid_adjoint_initial(
    displacement_fluid_adjoint_old,
    velocity_fluid_adjoint_old,
    displacement_fluid_adjoint_interface,
    velocity_fluid_adjoint_interface,
    displacement_fluid_adjoint_old_interface,
    velocity_fluid_adjoint_old_interface,
    xi_fluid,
    eta_fluid,
    fluid: Space,
    param: Parameters,
    time,
    time_step,
    microtimestep_before: MicroTimeStep,
    microtimestep: MicroTimeStep,
):

    return (
        + 2.0 * param.NU
        * characteristic_function_fluid(param)
        * dot(
            goal_functional(
                microtimestep_before, microtimestep, "primal_velocity"
            )[1],
            grad(eta_fluid),
        )
        * fluid.dx
    )


def bilinear_form_fluid_adjoint(
    displacement_fluid_adjoint,
    velocity_fluid_adjoint,
    xi_fluid,
    eta_fluid,
    fluid: Space,
    param: Parameters,
    time_step,
):

    return (
        eta_fluid * displacement_fluid_adjoint * fluid.dx
        + 0.5
        * time_step
        * form_fluid_adjoint(
            displacement_fluid_adjoint,
            velocity_fluid_adjoint,
            xi_fluid,
            eta_fluid,
            fluid,
            param,
        )
    )


def functional_fluid_adjoint(
    displacement_fluid_adjoint_old,
    velocity_fluid_adjoint_old,
    displacements_fluid_adjoint_interface,
    velocity_fluid_adjoint_interface,
    displacement_fluid_adjoint_old_interface,
    velocity_fluid_adjoint_old_interface,
    xi_fluid,
    eta_fluid,
    fluid: Space,
    param: Parameters,
    time,
    time_step,
    time_step_old,
    microtimestep_before: MicroTimeStep,
    microtimestep: MicroTimeStep,
    microtimestep_after: MicroTimeStep,
):

    return (
        eta_fluid * displacement_fluid_adjoint_old * fluid.dx
        - 0.5
        * time_step_old
        * form_fluid_adjoint(
            displacement_fluid_adjoint_old,
            velocity_fluid_adjoint_old,
            xi_fluid,
            eta_fluid,
            fluid,
            param,
        )
        + 2.0 * param.NU
        * characteristic_function_fluid(param)
        * dot(
            goal_functional(
                microtimestep_before, microtimestep, "primal_velocity"
            )[1],
            grad(eta_fluid),
        )
        * fluid.dx
        + 2.0 * param.NU
        * characteristic_function_fluid(param)
        * dot(
            goal_functional(
                microtimestep, microtimestep_after, "primal_velocity"
            )[0],
            grad(eta_fluid),
        )
        * fluid.dx
    )


# Define variational forms of the solid subproblem
def form_solid(
    displacement_solid,
    velocity_solid,
    phi_solid,
    psi_solid,
    solid: Space,
    param: Parameters,
):

    return (
        param.ZETA * dot(grad(displacement_solid), grad(phi_solid)) * solid.dx
        + param.DELTA * dot(grad(velocity_solid), grad(phi_solid)) * solid.dx
        - velocity_solid * psi_solid * solid.dx
        - param.DELTA
        * dot(grad(velocity_solid), solid.normal_vector)
        * phi_solid
        * solid.ds(1)
    )


def bilinear_form_solid(
    displacement_solid,
    velocity_solid,
    phi_solid,
    psi_solid,
    solid: Space,
    param: Parameters,
    time_step,
):

    return (
        velocity_solid * phi_solid * solid.dx
        + displacement_solid * psi_solid * solid.dx
        + 0.5
        * time_step
        * form_solid(
            displacement_solid,
            velocity_solid,
            phi_solid,
            psi_solid,
            solid,
            param,
        )
    )


def functional_solid(
    displacement_solid_old,
    velocity_solid_old,
    displacement_solid_interface,
    velocity_solid_interface,
    displacement_solid_old_interface,
    velocity_solid_old_interface,
    phi_solid,
    psi_solid,
    solid: Space,
    param: Parameters,
    time,
    time_step,
):

    return (
        velocity_solid_old * phi_solid * solid.dx
        + displacement_solid_old * psi_solid * solid.dx
        - 0.5
        * time_step
        * form_solid(
            displacement_solid_old,
            velocity_solid_old,
            phi_solid,
            psi_solid,
            solid,
            param,
        )
        - 0.5
        * time_step
        * param.NU
        * dot(grad(velocity_solid_interface), solid.normal_vector)
        * phi_solid
        * solid.ds(1)
        - 0.5
        * time_step
        * param.NU
        * dot(grad(velocity_solid_old_interface), solid.normal_vector)
        * phi_solid
        * solid.ds(1)
    )


# Define adjoint variational forms of the solid subproblem
def form_solid_adjoint(
    displacement_solid_adjoint,
    velocity_solid_adjoint,
    xi_solid,
    eta_solid,
    solid: Space,
    param: Parameters,
):

    return (
        param.ZETA
        * dot(grad(xi_solid), grad(displacement_solid_adjoint))
        * solid.dx
        + param.DELTA
        * dot(grad(eta_solid), grad(displacement_solid_adjoint))
        * solid.dx
        - eta_solid * velocity_solid_adjoint * solid.dx
        - param.DELTA
        * dot(grad(eta_solid), solid.normal_vector)
        * displacement_solid_adjoint
        * solid.ds(1)
    )


def functional_solid_adjoint_initial(
    displacement_solid_adjoint_old,
    velocity_solid_adjoint_old,
    displacement_solid_adjoint_interface,
    velocity_solid_adjoint_interface,
    displacement_solid_adjoint_old_interface,
    velocity_solid_adjoint_old_interface,
    xi_solid,
    eta_solid,
    solid: Space,
    param: Parameters,
    time,
    time_step,
    microtimestep_before: MicroTimeStep,
    microtimestep: MicroTimeStep,
):

    return (
        + 2.0 * param.ZETA
        * characteristic_function_solid(param)
        * dot(
            goal_functional(
                microtimestep_before, microtimestep, "primal_displacement"
            )[1],
            grad(xi_solid),
        )
        * solid.dx
    )


def bilinear_form_solid_adjoint(
    displacement_solid_adjoint,
    velocity_solid_adjoint,
    xi_solid,
    eta_solid,
    solid: Space,
    param: Parameters,
    time_step,
):

    return (
        eta_solid * displacement_solid_adjoint * solid.dx
        + xi_solid * velocity_solid_adjoint * solid.dx
        + 0.5
        * time_step
        * form_solid_adjoint(
            displacement_solid_adjoint,
            velocity_solid_adjoint,
            xi_solid,
            eta_solid,
            solid,
            param,
        )
    )


def functional_solid_adjoint(
    displacement_solid_adjoint_old,
    velocity_solid_adjoint_old,
    displacement_solid_adjoint_interface,
    velocity_solid_adjoint_interface,
    displacement_solid_adjoint_old_interface,
    velocity_solid_adjoint_old_interface,
    xi_solid,
    eta_solid,
    solid: Space,
    param: Parameters,
    time,
    time_step,
    time_step_old,
    microtimestep_before: MicroTimeStep,
    microtimestep: MicroTimeStep,
    microtimestep_after: MicroTimeStep,
):

    return (
        eta_solid * displacement_solid_adjoint_old * solid.dx
        + xi_solid * velocity_solid_adjoint_old * solid.dx
        - 0.5
        * time_step_old
        * form_solid_adjoint(
            displacement_solid_adjoint_old,
            velocity_solid_adjoint_old,
            xi_solid,
            eta_solid,
            solid,
            param,
        )
        + 2.0 * param.ZETA
        * characteristic_function_solid(param)
        * dot(
            goal_functional(
                microtimestep_before, microtimestep, "primal_displacement"
            )[1],
            grad(xi_solid),
        )
        * solid.dx
        + 2.0 * param.ZETA
        * characteristic_function_solid(param)
        * dot(
            goal_functional(
                microtimestep, microtimestep_after, "primal_displacement"
            )[0],
            grad(xi_solid),
        )
        * solid.dx
    )
