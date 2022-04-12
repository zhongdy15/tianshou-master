from discrete import *
import numpy as np
v1 = np.array([[1],[2],[3]])
v2 = np.array([[2],[2],[2]])
A_v1 = cal_A(v1)
A_v2 = cal_A(v2)
r = correlation_dist(v1,v2)
print(r)

