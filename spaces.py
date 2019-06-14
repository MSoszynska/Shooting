from __future__ import print_function
from fenics import*
from parameters import*

# Define boundary
def boundary(x, on_boundary):
    return on_boundary and (not near(x[1], 0.0))

# Define interface
class Interface(SubDomain):
    def inside(self, x, on_boundary):            
        return near(x[1], 0.0)
interface = Interface()

# Create meshes
mesh_f = RectangleMesh(Point(0.0, 0.0), Point(4.0, 1.0), nx, ny)
mesh_s = RectangleMesh(Point(0.0, -1.0), Point(4.0, 0.0), nx, ny)
h_f = CellDiameter(mesh_f)
h_s = CellDiameter(mesh_s)
n_f = FacetNormal(mesh_f)
n_s = FacetNormal(mesh_s)
boundarymesh = BoundaryMesh(mesh_s, 'exterior')
mesh_i = SubMesh(boundarymesh, interface)

# Define measures
dx_f = Measure('dx', domain = mesh_f)
ds_f = Measure('ds', domain = mesh_f)
dx_s = Measure('dx', domain = mesh_s)
ds_s = Measure('ds', domain = mesh_s)
dx_i = Measure('dx', domain = mesh_i)

# Define function space
W_f = FiniteElement('Lagrange', mesh_f.ufl_cell(), 1)
V_f = FunctionSpace(mesh_f, W_f * W_f)
W_s = FiniteElement('Lagrange', mesh_s.ufl_cell(), 1)
V_s = FunctionSpace(mesh_s, W_s * W_s)
W_i = FiniteElement('Lagrange', mesh_i.ufl_cell(), 1)
V_i = FunctionSpace(mesh_i, W_i * W_i)
