import CFx.io
import numpy as np
x = CFx.io.read_input_file('test/ONR_testbed/Input_Files/A11_unstructured.txt')
print(x)


t = np.zeros((150,1))
numpoints=150
vals_out = np.zeros((numpoints,len(x["QoI"])))
print(t.shape)
vals_out[:,0] = t[:,0]
