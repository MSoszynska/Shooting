from fenics import (Function, FunctionSpace, File,
                    HDF5File, MPI)

# Define initialization of a function
class Initial:

    def __init__(self, space_name, variable_name, V):
        
        # Define initial interface values
        self.i_new = Function(V)
        self.i_old = Function(V)

        # Define initial values for time loop
        self.new = Function(V)
        self.old = Function(V)
        
        # Create arrays of empty functions and iterations
        self.array = []
        self.iterations = []
        
        # Create pvd files
        self.pvd = File(f'solutions/{space_name}/pvd/{variable_name}.pvd')
        
        # Remember space
        self.V = V

        # Remember space and variable names
        self.space_name = space_name
        self.variable_name = variable_name

        # Create HDF5 counter
        self.HDF5_counter = 0

    def save(self, u):

        # Save solution in pvd format
        u.rename(self.variable_name, self.space_name)
        self.pvd << u

        # Save solution in HDF5 format
        uFile = HDF5File(MPI.comm_world,
                         f'solutions/{self.space_name}'
                         f'/HDF5/{self.variable_name}_'
                         f'{self.HDF5_counter}.h5', 'w')
        uFile.write(u,
                    f'solutions/{self.space_name}'
                    f'/HDF5/{self.variable_name}_{self.HDF5_counter}')
        uFile.close()
        self.HDF5_counter += 1

        return

    def load(self, m):

        # Load solution in HDF5 format
        u = Function(self.V)
        uFile = HDF5File(MPI.comm_world,
                         f'solutions/{self.space_name}'
                         f'/HDF5/{self.variable_name}_{m}.h5', 'r')
        uFile.read(u,
                   f'solutions/{self.space_name}'
                   f'/HDF5/{self.variable_name}_{m}')
        uFile.close()
        self.array.append(u)




