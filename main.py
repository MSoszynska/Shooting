import os

from fenics import (parameters, RectangleMesh, 
                    Point, BoundaryMesh, SubMesh)
from parameters import Parameters
from spaces import Inner_boundary, Space, boundary
from initial import Initial
from time_stepping import time_stepping
from relaxation import relaxation
from shooting import shooting

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
fluid = Space(mesh_f)
solid = Space(mesh_s)
interface = Space(mesh_i)

# Create directory
directory = os.getcwd()
os.makedirs('decoupling')
os.chdir('decoupling')

# Initialise solution
u = Initial(0, 'displacement', fluid.V, solid.V)
v = Initial(1, 'velocity', fluid.V, solid.V)

# Perform time-stepping with relaxation
Num_iters = time_stepping(u, v, fluid, solid, interface, 
                          param, shooting)

# Save solution
for i in range(param.N + 1):
    
    u.f_array[i].rename('Displacement', 'Fluid')
    u.f_pvd << u.f_array[i] 
    v.f_array[i].rename('Velocity', 'Fluid')
    v.f_pvd << v.f_array[i]
    u.s_array[i].rename('Displacement', 'Solid')
    u.s_pvd << u.s_array[i]
    v.s_array[i].rename('Velocity', 'Solid')
    v.s_pvd << v.s_array[i]  
    
# Print number of iterations
for n in Num_iters:
    
    print('Number of solved linear systems: ', n)

os.chdir(directory)
