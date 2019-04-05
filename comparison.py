from __future__ import print_function
from fenics import*
from parameters import*
from coupled import coupled
from primal_shooting import primal_shooting
from primal_relaxation import primal_relaxation
import os
import numpy as np

# Set option to allow extrapolation
parameters['allow_extrapolation'] = True

# Define interface
class Interface(SubDomain):
    def inside(self, x, on_boundary):            
        return near(x[1], 0.0)
interface = Interface()

# Create meshes
mesh_f = RectangleMesh(Point(0.0, 0.0), Point(4.0, 1.0), nx, ny)
mesh_s = RectangleMesh(Point(0.0, -1.0), Point(4.0, 0.0), nx, ny)
boundarymesh = BoundaryMesh(mesh_s, 'exterior')
mesh_i = SubMesh(boundarymesh, interface)

# Define measures
dx_f = Measure('dx', domain = mesh_f)
dx_s = Measure('dx', domain = mesh_s)
dx_i = Measure('dx', domain = mesh_i)

# Define function spaces
W_f = FiniteElement('Lagrange', mesh_f.ufl_cell(), 1)
V_f = FunctionSpace(mesh_f, W_f * W_f)
W_s = FiniteElement('Lagrange', mesh_s.ufl_cell(), 1)
V_s = FunctionSpace(mesh_s, W_s * W_s)
W_i = FiniteElement('Lagrange', mesh_i.ufl_cell(), 1)
V_i = FunctionSpace(mesh_i, W_i * W_i)

# Create directory
comparison_directory = os.getcwd()
os.makedirs('comparison')
os.chdir('comparison')

# Solve coupled problem
coupled = coupled()
U_array = coupled[0]
V_array = coupled[1]

# Solve primal problem using relaxation 
primal_relaxation = primal_relaxation()
U_f_array_relaxation = primal_relaxation[0]
V_f_array_relaxation = primal_relaxation[1]
U_s_array_relaxation = primal_relaxation[2]
V_s_array_relaxation = primal_relaxation[3]

# Solve primal problem using shooting
primal_shooting = primal_shooting()
U_f_array_shooting = primal_shooting[0]
V_f_array_shooting = primal_shooting[1]
U_s_array_shooting = primal_shooting[2]
V_s_array_shooting = primal_shooting[3]

# Print length of problems
print('Length of a coupled problem: ', str(len(U_array)), ', ', str(len(V_array))  )
print('Length of a relaxation problem: ', str(len(U_f_array_relaxation)), ', ', \
                                          str(len(V_f_array_relaxation)), ', ', \
                                          str(len(U_s_array_relaxation)), ', ', \
                                          str(len(V_s_array_relaxation)))
print('Length of a shooting problem: ', str(len(U_f_array_shooting)), ', ', \
                                        str(len(V_f_array_shooting)), ', ', \
                                        str(len(U_s_array_shooting)), ', ', \
                                        str(len(V_s_array_shooting)))

# Comparison of shooting method with fully coupled problem
print('Comparison of shooting method with fully coupled problem')
shooting_vs_coupled_directory = os.getcwd()
os.makedirs('shooting_vs_coupled')
os.chdir('shooting_vs_coupled')

# Create pvd files
u_f_1_pvd = File('solutions/fluid/displacement.pvd')
v_f_1_pvd = File('solutions/fluid/velocity.pvd')
u_s_1_pvd = File('solutions/solid/displacement.pvd')
v_s_1_pvd = File('solutions/solid/velocity.pvd')

# Create empty functions 
u_f_1 = Function(V_f.sub(0).collapse())
v_f_1 = Function(V_f.sub(1).collapse())
u_s_1 = Function(V_s.sub(0).collapse())
v_s_1 = Function(V_s.sub(1).collapse())

for i in range(N):
    
    print('Current time step: ', i + 1)
    
    # Extrapolate solutions
    u_f_1.assign(project(interpolate(U_f_array_shooting[i], V_f.sub(0).collapse()) - \
                 interpolate(U_array[i], V_f.sub(0).collapse()), V_f.sub(0).collapse()))
    v_f_1.assign(project(interpolate(V_f_array_shooting[i], V_f.sub(1).collapse()) - \
                 interpolate(V_array[i], V_f.sub(1).collapse()), V_f.sub(1).collapse()))
    u_s_1.assign(project(interpolate(U_s_array_shooting[i], V_s.sub(0).collapse()) - \
                 interpolate(U_array[i], V_s.sub(0).collapse()), V_s.sub(0).collapse()))
    v_s_1.assign(project(interpolate(V_s_array_shooting[i], V_s.sub(0).collapse()) - \
                 interpolate(V_array[i], V_s.sub(1).collapse()), V_s.sub(1).collapse()))
    
    # Save errors
    u_f_1.rename('Displacement', 'Fluid')
    u_f_1_pvd << u_f_1 
    v_f_1.rename('Velocity', 'Fluid')
    v_f_1_pvd << v_f_1
    u_s_1.rename('Displacement', 'Solid')
    u_s_1_pvd << u_s_1
    v_s_1.rename('Velocity', 'Solid')
    v_s_1_pvd << v_s_1  
      
    # Compute L2 norms
    print('Errors in L2 norm')
    u_f_1_L2 = sqrt(abs(assemble(u_f_1**2*dx_f)))
    print('Displacement L2 error for the fluid subdomain: ', str(u_f_1_L2))
    v_f_1_L2 = sqrt(abs(assemble(v_f_1**2*dx_f)))
    print('Velocity L2 error for the fluid subdomain: ', str(v_f_1_L2))
    u_s_1_L2 = sqrt(abs(assemble(u_s_1**2*dx_s)))
    print('Displacement L2 error for the solid subdomain: ', str(u_s_1_L2))
    v_s_1_L2 = sqrt(abs(assemble(v_s_1**2*dx_s)))
    print('Velocity L2 error for the solid subdomain: ', str(v_s_1_L2))
    L2_error_1 = u_f_1_L2 + v_f_1_L2 + u_s_1_L2 + v_s_1_L2
    print('Overall L2 error: ', str(L2_error_1))
    L2_error_1_txt = open('L2_error.txt', 'a')
    L2_error_1_txt.write(str(i) + ' ' + str(L2_error_1) + '\r\n')
    L2_error_1_txt.close()
    
    # Compute maximum norm
    print('Error in maximum norm')
    u_f_1_max = np.max(np.abs(u_f_1.vector().get_local()))
    print('Displacement max error for the fluid subdomain: ', str(u_f_1_max))
    v_f_1_max = np.max(np.abs(v_f_1.vector().get_local()))
    print('Velocity max error for the fluid subdomain: ', str(v_f_1_max))
    u_s_1_max = np.max(np.abs(u_s_1.vector().get_local()))
    print('Displacement max error for the solid subdomain: ', str(u_s_1_max))
    v_s_1_max = np.max(np.abs(v_s_1.vector().get_local()))
    print('Velocity max error for the solid subdomain: ', str(v_s_1_max))
    max_error_1 = u_f_1_max + v_f_1_max + u_s_1_max + v_s_1_max
    print('Overall max error: ', str(max_error_1))
    max_error_1_txt = open('max_error.txt', 'a')
    max_error_1_txt.write(str(i) + ' ' + str(max_error_1) + '\r\n')
    max_error_1_txt.close()
    
os.chdir(shooting_vs_coupled_directory)
    
# Comparison of shooting method with relaxation method
print('Comparison of shooting method with relaxation method')
shooting_vs_relaxation_directory = os.getcwd()
os.makedirs('shooting_vs_relaxation')
os.chdir('shooting_vs_relaxation')

# Create pvd files
u_f_2_pvd = File('solutions/fluid/displacement.pvd')
v_f_2_pvd = File('solutions/fluid/velocity.pvd')
u_s_2_pvd = File('solutions/solid/displacement.pvd')
v_s_2_pvd = File('solutions/solid/velocity.pvd')

# Create empty functions 
u_f_2 = Function(V_f.sub(0).collapse())
v_f_2 = Function(V_f.sub(1).collapse())
u_s_2 = Function(V_s.sub(0).collapse())
v_s_2 = Function(V_s.sub(1).collapse())

for i in range(N):
    
    # Extrapolate solutions
    u_f_2.assign(project(interpolate(U_f_array_shooting[i], V_f.sub(0).collapse()) - \
                 interpolate(U_f_array_relaxation[i], V_f.sub(0).collapse()), V_f.sub(0).collapse()))
    v_f_2.assign(project(interpolate(V_f_array_shooting[i], V_f.sub(1).collapse()) - \
                 interpolate(V_f_array_relaxation[i], V_f.sub(1).collapse()), V_f.sub(1).collapse()))
    u_s_2.assign(project(interpolate(U_s_array_shooting[i], V_s.sub(0).collapse()) - \
                 interpolate(U_s_array_relaxation[i], V_s.sub(0).collapse()), V_s.sub(0).collapse()))
    v_s_2.assign(project(interpolate(V_s_array_shooting[i], V_s.sub(0).collapse()) - \
                 interpolate(V_s_array_relaxation[i], V_s.sub(1).collapse()), V_s.sub(1).collapse()))    
    
    # Save errors
    u_f_2.rename('Displacement', 'Fluid')
    u_f_2_pvd << u_f_2 
    v_f_2.rename('Velocity', 'Fluid')
    v_f_2_pvd << v_f_2
    u_s_2.rename('Displacement', 'Solid')
    u_s_2_pvd << u_s_2
    v_s_2.rename('Velocity', 'Solid')
    v_s_2_pvd << v_s_2  
    
    # Compute L2 norms
    print('Errors in L2 norm')
    u_f_2_L2 = sqrt(abs(assemble(u_f_2**2*dx_f)))
    print('Displacement L2 error for the fluid subdomain: ', str(u_f_2_L2))
    v_f_2_L2 = sqrt(abs(assemble(v_f_2**2*dx_f)))
    print('Velocity L2 error for the fluid subdomain: ', str(v_f_2_L2))
    u_s_2_L2 = sqrt(abs(assemble(u_s_2**2*dx_s)))
    print('Displacement L2 error for the solid subdomain: ', str(u_s_2_L2))
    v_s_2_L2 = sqrt(abs(assemble(v_s_2**2*dx_s)))
    print('Velocity L2 error for the solid subdomain: ', str(v_s_2_L2))
    L2_error_2 = u_f_2_L2 + v_f_2_L2 + u_s_2_L2 + v_s_2_L2
    print('Overall L2 error: ', str(L2_error_2))
    L2_error_2_txt = open('L2_error.txt', 'a')
    L2_error_2_txt.write(str(i) + ' ' + str(L2_error_2) + '\r\n')
    L2_error_2_txt.close()
    
    # Compute maximum norm
    print('Error in maximum norm')
    u_f_2_max = np.max(np.abs(u_f_2.vector().get_local()))
    print('Displacement max error for the fluid subdomain: ', str(u_f_2_max))
    v_f_2_max = np.max(np.abs(v_f_2.vector().get_local()))
    print('Velocity max error for the fluid subdomain: ', str(v_f_2_max))
    u_s_2_max = np.max(np.abs(u_s_2.vector().get_local()))
    print('Displacement max error for the solid subdomain: ', str(u_s_2_max))
    v_s_2_max = np.max(np.abs(v_s_2.vector().get_local()))
    print('Velocity max error for the solid subdomain: ', str(v_s_2_max))
    max_error_2 = u_f_2_max + v_f_2_max + u_s_2_max + v_s_2_max
    print('Overall max error: ', str(max_error_2))
    max_error_2_txt = open('max_error.txt', 'a')
    max_error_2_txt.write(str(i) + ' ' + str(max_error_2) + '\r\n')
    max_error_2_txt.close()
    
    
os.chdir(shooting_vs_relaxation_directory)   

# Comparison of relaxation method with fully coupled problem
print('Comparison of relaxation method with fully coupled problem')
relaxation_vs_coupled_directory = os.getcwd()
os.makedirs('relaxation_vs_coupled')
os.chdir('relaxation_vs_coupled')

# Create pvd files
u_f_3_pvd = File('solutions/fluid/displacement.pvd')
v_f_3_pvd = File('solutions/fluid/velocity.pvd')
u_s_3_pvd = File('solutions/solid/displacement.pvd')
v_s_3_pvd = File('solutions/solid/velocity.pvd')

# Create empty functions 
u_f_3 = Function(V_f.sub(0).collapse())
v_f_3 = Function(V_f.sub(1).collapse())
u_s_3 = Function(V_s.sub(0).collapse())
v_s_3 = Function(V_s.sub(1).collapse())

for i in range(N):
    
    print('Current time step: ', i + 1)
    
    # Extrapolate solutions
    u_f_3.assign(project(interpolate(U_f_array_relaxation[i], V_f.sub(0).collapse()) - \
                 interpolate(U_array[i], V_f.sub(0).collapse()), V_f.sub(0).collapse()))
    v_f_3.assign(project(interpolate(V_f_array_relaxation[i], V_f.sub(1).collapse()) - \
                 interpolate(V_array[i], V_f.sub(1).collapse()), V_f.sub(1).collapse()))
    u_s_3.assign(project(interpolate(U_s_array_relaxation[i], V_s.sub(0).collapse()) - \
                 interpolate(U_array[i], V_s.sub(0).collapse()), V_s.sub(0).collapse()))
    v_s_3.assign(project(interpolate(V_s_array_relaxation[i], V_s.sub(0).collapse()) - \
                 interpolate(V_array[i], V_s.sub(1).collapse()), V_s.sub(1).collapse()))
    
    # Save errors
    u_f_3.rename('Displacement', 'Fluid')
    u_f_3_pvd << u_f_3 
    v_f_3.rename('Velocity', 'Fluid')
    v_f_3_pvd << v_f_3
    u_s_3.rename('Displacement', 'Solid')
    u_s_3_pvd << u_s_3
    v_s_3.rename('Velocity', 'Solid')
    v_s_3_pvd << v_s_3 
      
    # Compute L2 norms
    print('Errors in L2 norm')
    u_f_3_L2 = sqrt(abs(assemble(u_f_3**2*dx_f)))
    print('Displacement L2 error for the fluid subdomain: ', str(u_f_3_L2))
    v_f_3_L2 = sqrt(abs(assemble(v_f_3**2*dx_f)))
    print('Velocity L2 error for the fluid subdomain: ', str(v_f_3_L2))
    u_s_3_L2 = sqrt(abs(assemble(u_s_3**2*dx_s)))
    print('Displacement L2 error for the solid subdomain: ', str(u_s_3_L2))
    v_s_3_L2 = sqrt(abs(assemble(v_s_3**2*dx_s)))
    print('Velocity L2 error for the solid subdomain: ', str(v_s_3_L2))
    L2_error_3 = u_f_3_L2 + v_f_3_L2 + u_s_3_L2 + v_s_3_L2
    print('Overall L2 error: ', str(L2_error_3))
    L2_error_3_txt = open('L2_error.txt', 'a')
    L2_error_3_txt.write(str(i) + ' ' + str(L2_error_3) + '\r\n')
    L2_error_3_txt.close()
    
    # Compute maximum norm
    print('Error in maximum norm')
    u_f_3_max = np.max(np.abs(u_f_3.vector().get_local()))
    print('Displacement max error for the fluid subdomain: ', str(u_f_3_max))
    v_f_3_max = np.max(np.abs(v_f_3.vector().get_local()))
    print('Velocity max error for the fluid subdomain: ', str(v_f_3_max))
    u_s_3_max = np.max(np.abs(u_s_3.vector().get_local()))
    print('Displacement max error for the solid subdomain: ', str(u_s_3_max))
    v_s_3_max = np.max(np.abs(v_s_3.vector().get_local()))
    print('Velocity max error for the solid subdomain: ', str(v_s_3_max))
    max_error_3 = u_f_3_max + v_f_3_max + u_s_3_max + v_s_3_max
    print('Overall max error: ', str(max_error_3))
    max_error_3_txt = open('max_error.txt', 'a')
    max_error_3_txt.write(str(i) + ' ' + str(max_error_3) + '\r\n')
    max_error_3_txt.close()
    
os.chdir(relaxation_vs_coupled_directory)

os.chdir(comparison_directory)
