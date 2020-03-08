from fenics import Function, DirichletBC, Constant
from solve_problem import solve_problem
from coupling import solid_to_fluid, fluid_to_solid
from initial import Initial
from spaces import (
    Space,
    boundary_up,
    boundary_left,
    boundary_right,
    boundary_down,
    boundary_between)
from parameters import Parameters
from time_structure import TimeLine


def time_stepping(
    functional_fluid_initial,
    functional_solid_initial,
    bilinear_form_fluid,
    functional_fluid,
    bilinear_form_solid,
    functional_solid,
    fluid: Space,
    solid: Space,
    interface: Space,
    param: Parameters,
    decoupling,
    fluid_timeline: TimeLine,
    solid_timeline: TimeLine,
    adjoint,
):

    # Initialize function objects
    if adjoint:

        displacement_name = "adjoint_displacement"
        velocity_name = "adjoint_velocity"

    else:

        displacement_name = "primal_displacement"
        velocity_name = "primal_velocity"

    displacement_fluid = Initial(
        "fluid", displacement_name, fluid.function_space_split[0]
    )
    velocity_fluid = Initial(
        "fluid", velocity_name, fluid.function_space_split[1]
    )
    displacement_solid = Initial(
        "solid", displacement_name, solid.function_space_split[0]
    )
    velocity_solid = Initial(
        "solid", velocity_name, solid.function_space_split[1]
    )

    # Define Dirichlet boundary conditions
    if not adjoint:
        fluid.boundaries = [
            DirichletBC(fluid.function_space_split[1], Constant(0.0), boundary_between),
            DirichletBC(fluid.function_space_split[1], Constant(0.0), boundary_up),
            DirichletBC(fluid.function_space_split[1], Constant(0.0), boundary_left),
            DirichletBC(fluid.function_space_split[1], Constant(0.0), boundary_right)
        ]
        solid.boundaries = [
            DirichletBC(solid.function_space.sub(0), Constant(0.0), boundary_left),
            DirichletBC(solid.function_space.sub(1), Constant(0.0), boundary_left),
            DirichletBC(solid.function_space.sub(0), Constant(0.0), boundary_right),
            DirichletBC(solid.function_space.sub(1), Constant(0.0), boundary_right),
            DirichletBC(solid.function_space.sub(0), Constant(0.0), boundary_down),
            DirichletBC(solid.function_space.sub(1), Constant(0.0), boundary_down)
        ]
    elif adjoint and param.GOAL_FUNCTIONAL_FLUID:
        fluid.boundaries = [
            DirichletBC(fluid.function_space_split[0], Constant(0.0), boundary_between),
            DirichletBC(fluid.function_space_split[0], Constant(0.0), boundary_up),
            DirichletBC(fluid.function_space_split[0], Constant(0.0), boundary_left),
            DirichletBC(fluid.function_space_split[0], Constant(0.0), boundary_right),
        ]
        solid.boundaries = [
            DirichletBC(solid.function_space.sub(0), Constant(0.0), boundary_left),
            DirichletBC(solid.function_space.sub(1), Constant(0.0), boundary_left),
            DirichletBC(solid.function_space.sub(0), Constant(0.0), boundary_right),
            DirichletBC(solid.function_space.sub(1), Constant(0.0), boundary_right),
            DirichletBC(solid.function_space.sub(0), Constant(0.0), boundary_down),
            DirichletBC(solid.function_space.sub(1), Constant(0.0), boundary_down)
        ]
    else:
        fluid.boundaries = [
            DirichletBC(fluid.function_space_split[0], Constant(0.0), boundary_up),
            DirichletBC(fluid.function_space_split[0], Constant(0.0), boundary_left),
            DirichletBC(fluid.function_space_split[0], Constant(0.0), boundary_right),
        ]
        solid.boundaries = [
            DirichletBC(solid.function_space.sub(0), Constant(0.0), boundary_between),
            DirichletBC(solid.function_space.sub(0), Constant(0.0), boundary_left),
            DirichletBC(solid.function_space.sub(1), Constant(0.0), boundary_left),
            DirichletBC(solid.function_space.sub(0), Constant(0.0), boundary_right),
            DirichletBC(solid.function_space.sub(1), Constant(0.0), boundary_right),
            DirichletBC(solid.function_space.sub(0), Constant(0.0), boundary_down),
            DirichletBC(solid.function_space.sub(1), Constant(0.0), boundary_down)
        ]

    # Save initial values for the primal problem
    if not adjoint:

        displacement_fluid.save(Function(fluid.function_space_split[0]))
        velocity_fluid.save(Function(fluid.function_space_split[1]))
        displacement_solid.save(Function(solid.function_space_split[0]))
        velocity_solid.save(Function(solid.function_space_split[1]))

    # Define time pointers
    if adjoint:

        fluid_macrotimestep = fluid_timeline.tail
        solid_macrotimestep = solid_timeline.tail

    else:

        fluid_macrotimestep = fluid_timeline.head
        solid_macrotimestep = solid_timeline.head

    # Create time loop
    size = fluid_timeline.size
    first_time_step = True
    counter = 0
    for n in range(size):

        if adjoint:

            print(f"Current macro time-step {size - counter}")

        else:

            print(f"Current macro time-step {counter + 1}")

        # Perform decoupling
        decoupling(
            displacement_fluid,
            velocity_fluid,
            displacement_solid,
            velocity_solid,
            functional_fluid_initial,
            functional_solid_initial,
            bilinear_form_fluid,
            functional_fluid,
            bilinear_form_solid,
            functional_solid,
            first_time_step,
            fluid,
            solid,
            interface,
            param,
            fluid_macrotimestep,
            solid_macrotimestep,
            adjoint,
        )

        # Define Dirichlet boundary conditions
        if not adjoint:
            boundary_velocity = solid_to_fluid(velocity_solid.new, fluid, solid, param, 1)
            fluid.boundaries = [
                DirichletBC(fluid.function_space_split[1], boundary_velocity, boundary_between),
                DirichletBC(fluid.function_space_split[1], Constant(0.0), boundary_up),
                DirichletBC(fluid.function_space_split[1], Constant(0.0), boundary_left),
                DirichletBC(fluid.function_space_split[1], Constant(0.0), boundary_right)
            ]

        # Solve fluid problem
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
            save=True,
        )

        # Solve solid problem
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
            save=True,
        )

        first_time_step = False

        # Update solution
        displacement_fluid.old.assign(displacement_fluid.new)
        velocity_fluid.old.assign(velocity_fluid.new)
        displacement_solid.old.assign(displacement_solid.new)
        velocity_solid.old.assign(velocity_solid.new)

        # Update boundary conditions
        displacement_fluid.interface_old.assign(
            displacement_fluid.interface_new
        )
        velocity_fluid.interface_old.assign(velocity_fluid.interface_new)
        velocity_solid.interface_old.assign(velocity_solid.interface_new)

        # Advance timeline
        if adjoint:

            fluid_macrotimestep = fluid_macrotimestep.before
            solid_macrotimestep = solid_macrotimestep.before

        else:

            fluid_macrotimestep = fluid_macrotimestep.after
            solid_macrotimestep = solid_macrotimestep.after
        counter += 1

    # Save initial values for the adjoint problem
    if adjoint:

        displacement_fluid.save(Function(fluid.function_space_split[0]))
        velocity_fluid.save(Function(fluid.function_space_split[1]))
        displacement_solid.save(Function(solid.function_space_split[0]))
        velocity_solid.save(Function(solid.function_space_split[1]))

    # Check convergence
    failed = 0
    for i in range(len(displacement_fluid.iterations)):
        failed += min(0, displacement_fluid.iterations[i])
    if failed < 0:
        print('The decoupling method failed at some point')

    return


