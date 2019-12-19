from fenics import Function, FunctionSpace, project, interpolate, norm
from solve_problem import solve_problem
from coupling import solid_to_fluid, fluid_to_solid
from parameters import Parameters
from spaces import Space
from time_structure import MacroTimeStep
from initial import Initial

# Define relaxation method
def relaxation(
    displacement_fluid: Initial,
    velocity_fluid: Initial,
    displacement_solid: Initial,
    velocity_solid: Initial,
    functional_fluid_initial,
    functional_solid_initial,
    bilinear_form_fluid,
    functional_fluid,
    bilinear_form_solid,
    functional_solid,
    first_time_step,
    fluid: Space,
    solid: Space,
    interface: Space,
    param: Parameters,
    fluid_macrotimestep: MacroTimeStep,
    solid_macrotimestep: MacroTimeStep,
    adjoint,
):

    # Define initial values for relaxation method
    displacement_solid_new = Function(solid.function_space_split[0])
    velocity_displacement_new = Function(solid.function_space_split[1])
    number_of_iterations = 0
    stop = False

    while not stop:

        number_of_iterations += 1
        print(
            f"Current iteration of relaxation method: {number_of_iterations}"
        )

        # Save old values
        displacement_solid_new.assign(displacement_solid.new)
        velocity_displacement_new.assign(velocity_solid.new)

        # Perform one iteration
        solve_problem(
            displacement_fluid,
            velocity_fluid,
            displacement_solid,
            velocity_solid,
            fluid,
            solid,
            solid_to_fluid,
            functional_fluid_initial,
            bilinear_form_fluid,
            functional_fluid,
            first_time_step,
            param,
            fluid_macrotimestep,
            adjoint,
        )
        solve_problem(
            displacement_solid,
            velocity_solid,
            displacement_fluid,
            velocity_fluid,
            solid,
            fluid,
            fluid_to_solid,
            functional_solid_initial,
            bilinear_form_solid,
            functional_solid,
            first_time_step,
            param,
            solid_macrotimestep,
            adjoint,
        )

        # Perform relaxation
        displacement_solid.new.assign(
            project(
                param.TAU * displacement_solid.new
                + (1.0 - param.TAU) * displacement_solid_new,
                solid.function_space_split[0],
            )
        )
        velocity_solid.new.assign(
            project(
                param.TAU * velocity_solid.new
                + (1.0 - param.TAU) * velocity_displacement_new,
                solid.function_space_split[1],
            )
        )

        # Define errors on the interface
        displacement_error = interpolate(
            project(
                displacement_solid_new - displacement_solid.new,
                solid.function_space_split[0],
            ),
            interface.function_space_split[0],
        )
        displacement_error_linf = norm(displacement_error.vector(), "linf")
        velocity_error = interpolate(
            project(
                velocity_displacement_new - velocity_solid.new,
                solid.function_space_split[1],
            ),
            interface.function_space_split[1],
        )
        velocity_error_linf = norm(velocity_error.vector(), "linf")
        error_linf = max(displacement_error_linf, velocity_error_linf)
        if number_of_iterations == 1:

            error_initial_linf = error_linf

        print(f"Absolute error on the interface: {error_linf}")
        print(
            f"Relative error on the interface: {error_linf / error_initial_linf}"
        )

        # Check stop conditions
        if (
            error_linf < param.ABSOLUTE_TOLERANCE_RELAXATION
            or error_linf / error_initial_linf
            < param.RELATIVE_TOLERANCE_RELAXATION
        ):

            print(
                f"Algorithm converged successfully after "
                f"{number_of_iterations} iterations"
            )
            stop = True

        elif number_of_iterations == param.MAX_ITERATIONS_RELAXATION:

            print("Maximal number of iterations was reached.")
            stop = True
            number_of_iterations = -1

    displacement_fluid.iterations.append(number_of_iterations)
    velocity_fluid.iterations.append(number_of_iterations)
    displacement_solid.iterations.append(number_of_iterations)
    displacement_solid.iterations.append(number_of_iterations)

    return
