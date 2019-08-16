import os

from fenics import Function, FunctionSpace, File


# Define initialization of a function
class Initial:
    def __init__(self, i, name, V_f, V_s):
        
        # Define initial interface values
        self.f_n_i = Function(V_f.sub(i).collapse())
        self.s_n_i = Function(V_s.sub(i).collapse())

        # Define initial values for time loop
        self.f = Function(V_f.sub(i).collapse())
        self.s = Function(V_f.sub(i).collapse())
        self.f_n = Function(V_f.sub(i).collapse())
        self.s_n = Function(V_s.sub(i).collapse())
        
        # Create arrays of empty functions
        self.f_array = []
        self.f_array.append(Function(V_f.sub(i).collapse()))
        self.s_array = []
        self.s_array.append(Function(V_s.sub(i).collapse()))
        
        # Create pvd files
        self.f_pvd = File('solutions/fluid/' + name + '.pvd')
        self.s_pvd = File('solutions/solid/' + name + '.pvd')
        
        # Remember arguments
        self.i = i
        self.name = name
        self.V_f = V_f
        self.V_s = V_s
    
