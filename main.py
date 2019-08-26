import os

from fenics import (parameters, RectangleMesh, 
                    Point, BoundaryMesh, SubMesh)
from parameters import Parameters
from spaces import Inner_boundary, Space
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
u_f = Initial('fluid', 'displacement', fluid.V_split[0])
v_f = Initial('fluid', 'velocity', fluid.V_split[1])
u_s = Initial('solid', 'displacement', solid.V_split[0])
v_s = Initial('solid', 'velocity', solid.V_split[1])

# Perform time-stepping with relaxation
Num_iters = time_stepping(u_f, v_f, u_s, v_s,
                          fluid, solid, interface, param, relaxation)

# Save solutions in pvd format
for i in range(param.M*param.N + 1):
    
    u_f.array[i].rename('Displacement', 'Fluid')
    u_f.pvd << u_f.array[i]
    v_f.array[i].rename('Velocity', 'Fluid')
    v_f.pvd << v_f.array[i]
    
for i in range(param.K*param.N + 1):
    
    u_s.array[i].rename('Displacement', 'Solid')
    u_s.pvd << u_s.array[i]
    v_s.array[i].rename('Velocity', 'Solid')
    v_s.pvd << v_s.array[i]
    
# Print number of iterations
for n in Num_iters:
    
    print('Number of solved linear systems: ', n)

os.chdir(directory)
