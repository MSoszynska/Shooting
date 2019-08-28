from fenics import Function, FunctionSpace, File

# Define initialization of a function
class Initial:
    def __init__(self, subdomain_name, space_name, V):
        
        # Define initial interface values
        self.i_new = Function(V)
        self.i_old = Function(V)

        # Define initial values for time loop
        self.new = Function(V)
        self.old = Function(V)
        
        # Create arrays of empty functions
        self.array = []
        self.array.append(Function(V))
        
        # Create pvd files
        self.pvd = File('solutions/' + subdomain_name +
                        '/' + space_name + '.pvd')
        
        # Remember space
        self.V = V
