from fenics import (near, SubDomain, CellDiameter, 
                    FacetNormal, Measure, FiniteElement, 
                    FunctionSpace)

# Define boundary
def boundary(x, on_boundary):
    return on_boundary and not near(x[1], 0.0)

# Define interface
class Inner_boundary(SubDomain):
    def inside(self, x, on_boundary):            
        return near(x[1], 0.0)

# Store space attributes
class Space:
    def __init__(self, mesh):
        
        # Define mesh parameters
        self.mesh = mesh
        self.h = CellDiameter(mesh)
        self.n = FacetNormal(mesh)
        
        # Define measures
        self.dx = Measure('dx', domain = mesh)
        self.ds = Measure('ds', domain = mesh)
        
        # Define function space
        W = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
        self.V = FunctionSpace(mesh, W * W)
        self.V_u = self.V.sub(0).collapse()
        self.V_v = self.V.sub(1).collapse()
