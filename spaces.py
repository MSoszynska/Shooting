from fenics import (near, SubDomain, CellDiameter, 
                    FacetNormal, Measure, MeshFunction,
                    FiniteElement, FunctionSpace)

# Define boundary
def boundary(x, on_boundary):

    return on_boundary and not near(x[1], 0.0)

# Define interface
class Inner_boundary(SubDomain):

    def inside(self, x, on_boundary):

        return near(x[1], 0.0)

# Store space attributes
class Space:

    def __init__(self, mesh, N = 0):
        
        # Define mesh parameters
        self.mesh = mesh
        self.h = CellDiameter(mesh)
        self.normal = FacetNormal(mesh)
        
        # Define measures
        inner_boundary = Inner_boundary()
        sub_domains = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
        sub_domains.set_all(0)
        inner_boundary.mark(sub_domains, 1)
        self.dx = Measure('dx', domain = mesh)
        self.ds = Measure('ds', domain = mesh, subdomain_data = sub_domains)
        
        # Define function spaces
        W = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
        self.V = FunctionSpace(mesh, W * W)
        self.V_split = [self.V.sub(0).collapse(),
                        self.V.sub(1).collapse()]

        # Define number of fractional time-steps
        self.N = N

