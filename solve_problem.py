from fenics import (
    Function,
    FunctionSpace,
    project,
    DirichletBC,
    Constant,
    TrialFunction,
    split,
    TestFunction,
    solve,
    assemble,
    dot,
    grad,
    rhs,
    lhs,
    Expression
)
from spaces import Space
from parameters import Parameters
from time_structure import MacroTimeStep
from initial import Initial

# Define a function solving a problem on a subdomain
def solve_problem(
    displacement: Initial,
    velocity: Initial,
    given_displacement: Initial,
    given_velocity: Initial,
    space: Space,
    space_interface: Space,
    transfer_function,
    functional_initial,
    bilinear_form,
    functional,
    first_time_step,
    param: Parameters,
    macrotimestep: MacroTimeStep,
    adjoint,
    save=False,
):

    # Store old solutions
    displacement_old = Function(space.function_space_split[0])
    velocity_old = Function(space.function_space_split[1])
    displacement_old.assign(displacement.old)
    velocity_old.assign(velocity.old)

    # Store old interface values
    displacement_old_interface = Function(space.function_space_split[0])
    velocity_old_interface = Function(space.function_space_split[1])
    displacement_old_interface.assign(displacement.interface_old)
    velocity_old_interface.assign(velocity.interface_old)

    # Initialize new interface values
    displacement_interface = Function(space.function_space_split[0])
    velocity_interface = Function(space.function_space_split[1])

    # Define time pointers
    if adjoint:

        microtimestep = macrotimestep.tail.before

    else:

        microtimestep = macrotimestep.head
    microtimestep_before = None
    microtimestep_after = None

    # Compute macro time-step size
    size = macrotimestep.size - 1
    for m in range(size):

        # Extrapolate weak boundary conditions on the interface
        if adjoint:

            extrapolation_proportion = (
                microtimestep.point - macrotimestep.head.point
            ) / macrotimestep.dt
            time_step = microtimestep.dt
            microtimestep_adjoint = microtimestep.after
            microtimestep_adjoint_before = microtimestep
            if m == 0 and macrotimestep.after is None:
                time_step_old = microtimestep.dt
                microtimestep_adjoint_after = microtimestep_adjoint
            elif m == 0:
                time_step_old = macrotimestep.microtimestep_after.before.dt
                microtimestep_adjoint_after = macrotimestep.microtimestep_after
            else:
                time_step_old = microtimestep.after.dt
                microtimestep_adjoint_after = microtimestep_adjoint.after

        else:

            extrapolation_proportion = (
                macrotimestep.tail.point - microtimestep.after.point
            ) / macrotimestep.dt
            time_step = microtimestep.dt
            time_step_old = microtimestep.dt

        displacement_interface.assign(
            project(
                extrapolation_proportion * displacement.interface_old
                + (1.0 - extrapolation_proportion)
                * transfer_function(
                    given_displacement.new, space, space_interface, param, 0
                ),
                space.function_space_split[0],
            )
        )
        velocity_interface.assign(
            project(
                extrapolation_proportion * velocity.interface_old
                + (1.0 - extrapolation_proportion)
                * transfer_function(
                    given_velocity.new, space, space_interface, param, 1
                ),
                space.function_space_split[1],
            )
        )

        if space.name == "fluid" and not adjoint:

            velocity_new = TrialFunction(space.function_space_split[1])
            phi = TestFunction(space.function_space_split[1])
            displacement_new = Function(space.function_space_split[0])
            psi = Function(space.function_space_split[0])

        elif space.name == "fluid" and adjoint:

            displacement_new = TrialFunction(space.function_space_split[0])
            psi = TestFunction(space.function_space_split[0])
            velocity_new = Function(space.function_space_split[1])
            phi = Function(space.function_space_split[1])

        else:

            trial_function = TrialFunction(space.function_space)
            (displacement_new, velocity_new) = split(trial_function)
            test_function = TestFunction(space.function_space)
            (phi, psi) = split(test_function)

        # Define scheme
        time = microtimestep.point
        left_hand_side = bilinear_form(
            displacement_new, velocity_new, phi, psi, space, param, time_step
        )
        if adjoint and first_time_step and m == 0:

            right_hand_side = functional_initial(
                displacement_old,
                velocity_old,
                displacement_interface,
                velocity_interface,
                displacement_old_interface,
                velocity_old_interface,
                phi,
                psi,
                space,
                param,
                time,
                time_step,
                microtimestep_adjoint_before,
                microtimestep_adjoint,
            )

        elif adjoint:

            right_hand_side = functional(
                displacement_old,
                velocity_old,
                displacement_interface,
                velocity_interface,
                displacement_old_interface,
                velocity_old_interface,
                phi,
                psi,
                space,
                param,
                time,
                time_step,
                time_step_old,
                microtimestep_adjoint_before,
                microtimestep_adjoint,
                microtimestep_adjoint_after,
            )

        else:

            right_hand_side = functional(
                displacement_old,
                velocity_old,
                displacement_interface,
                velocity_interface,
                displacement_old_interface,
                velocity_old_interface,
                phi,
                psi,
                space,
                param,
                time,
                time_step,
            )

        if space.name == "fluid" and not adjoint:

            velocity_new = Function(space.function_space_split[1])
            equation = left_hand_side - right_hand_side
            A = lhs(equation)
            L = rhs(equation)
            solve(A == L, velocity_new, space.boundaries)

        elif space.name == "fluid" and adjoint:

            displacement_new = Function(space.function_space_split[0])
            equation = left_hand_side - right_hand_side
            A = lhs(equation)
            L = rhs(equation)
            solve(A == L, displacement_new, space.boundaries)

        else:

            trial_function = Function(space.function_space)
            solve(
                left_hand_side == right_hand_side,
                trial_function,
                space.boundaries,
            )
            (displacement_new, velocity_new) = trial_function.split(trial_function)

        # # Define scheme
        # time = microtimestep.point
        # left_hand_side = bilinear_form(
        #     displacement_new, velocity_new, displacement_new, velocity_new, space, param, time_step
        # )
        # if adjoint and first_time_step and m == 0:
        #
        #     right_hand_side = functional_initial(
        #         displacement_old,
        #         velocity_old,
        #         displacement_interface,
        #         velocity_interface,
        #         displacement_old_interface,
        #         velocity_old_interface,
        #         displacement_new,
        #         velocity_new,
        #         space,
        #         param,
        #         time,
        #         time_step,
        #         microtimestep_adjoint_before,
        #         microtimestep_adjoint,
        #     )
        #
        # elif adjoint:
        #
        #     right_hand_side = functional(
        #         displacement_old,
        #         velocity_old,
        #         displacement_interface,
        #         velocity_interface,
        #         displacement_old_interface,
        #         velocity_old_interface,
        #         displacement_new,
        #         velocity_new,
        #         space,
        #         param,
        #         time,
        #         time_step,
        #         time_step_old,
        #         microtimestep_adjoint_before,
        #         microtimestep_adjoint,
        #         microtimestep_adjoint_after,
        #     )
        #
        # else:
        #
        #     right_hand_side = functional(
        #         displacement_old,
        #         velocity_old,
        #         displacement_interface,
        #         velocity_interface,
        #         displacement_old_interface,
        #         velocity_old_interface,
        #         displacement_new,
        #         velocity_new,
        #         space,
        #         param,
        #         time,
        #         time_step,
        #     )
        # print(assemble(left_hand_side - right_hand_side))

        # Monitor the interface
        if param.MONITOR:
            print(f"Normal derivative of the old solution of the {space_interface.name} problem on the interface")
            print(assemble(dot(grad(given_velocity.old), space_interface.normal_vector) * space_interface.ds(1)))
            print(f"Old solution of the {space_interface.name} problem on the interface")
            print(assemble(given_velocity.old * space_interface.ds(1)))
            print(f"Normal derivative of the old solution of the {space.name} problem on the interface")
            print(assemble(dot(grad(velocity.old), space.normal_vector) * space.ds(1)))
            print(f"Old solution of the {space.name} problem on the interface")
            print(assemble(velocity_old * space.ds(1)))
            print(f"Normal derivative of the solution of the {space_interface.name} problem on the interface")
            print(assemble(dot(grad(given_velocity.new), space_interface.normal_vector) * space_interface.ds(1)))
            print(f"Solution of the {space_interface.name} problem on the interface")
            print(assemble(given_velocity.new * space_interface.ds(1)))
            print(f"Normal derivative of the solution of the {space.name} problem on the interface")
            print(assemble(dot(grad(velocity_new), space.normal_vector) * space.ds(1)))
            print(f"Old solution of the {space.name} problem on the interface")
            print(assemble(velocity_new * space.ds(1)))

        # Save solutions
        if save:

            displacement.save(displacement_new)
            velocity.save(velocity_new)

        # Update solution
        displacement_old.assign(displacement_new)
        velocity_old.assign(velocity_new)

        # Update boundary conditions
        displacement_old_interface.assign(
            project(displacement_interface, space.function_space_split[0])
        )
        velocity_old_interface.assign(
            project(velocity_interface, space.function_space_split[1])
        )

        # Advance timeline
        if adjoint:

            microtimestep = microtimestep.before

        else:

            microtimestep = microtimestep.after

    # Save final values
    displacement.new.assign(displacement_new)
    velocity.new.assign(velocity_new)
    displacement.interface_new.assign(displacement_interface)
    velocity.interface_new.assign(velocity_interface)

    return
