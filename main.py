import os

from fenics import (parameters, RectangleMesh, 
                    Point, BoundaryMesh, SubMesh,
                    HDF5File, MPI)
from parameters import Parameters
from spaces import Inner_boundary, Space
from initial import Initial
from time_stepping import time_stepping
from forms import (A_f, L_f, A_s, L_s,
                   L_f_adjoint_0, L_s_adjoint_0,
                   A_f_adjoint, L_f_adjoint,
                   A_s_adjoint, L_s_adjoint)
from relaxation import relaxation
from shooting import shooting
from error_estimate import compute_residuals

parameters['allow_extrapolation'] = True
param = Parameters()

# Create meshes
mesh_f = RectangleMesh(Point(0.0, 0.0), 
                       Point(4.0, 1.0), param.nx, param.ny, 
                       diagonal = 'right')
mesh_s = RectangleMesh(Point(0.0, -1.0), 
                       Point(4.0, 0.0), param.nx, param.ny, 
                       diagonal = 'left')
boundary_mesh = BoundaryMesh(mesh_f, 'exterior')
inner_boundary = Inner_boundary()
mesh_i = SubMesh(boundary_mesh, inner_boundary)

# Create function spaces
fluid = Space(mesh_f, param.M)
solid = Space(mesh_s, param.K)
interface = Space(mesh_i)

# Create directory
try:

    os.makedirs(str(param.N))

except FileExistsError:

    pass
os.chdir(str(param.N))

# Set deoupling method
if param.relaxation:

    decoupling = relaxation

else:

    decoupling = shooting

# Perform time-stepping of the primal problem
adjoint = False
u_f, v_f, u_s, v_s = time_stepping(L_f, L_s, A_f, L_f, A_s, L_s,
                                   fluid, solid, interface, param,
                                   decoupling, adjoint)

# Perform time-stepping of the adjoint problem
adjoint = True
z_f, y_f, z_s, y_s = time_stepping(L_f_adjoint_0, L_s_adjoint_0,
                                   A_f_adjoint, L_f_adjoint,
                                   A_s_adjoint, L_s_adjoint,
                                   fluid, solid, interface, param,
                                   decoupling, adjoint)

# Compute residuals
residuals = compute_residuals(u_f, v_f, u_s, v_s, z_f, y_f,
                              z_s, y_s, fluid, solid, param)

