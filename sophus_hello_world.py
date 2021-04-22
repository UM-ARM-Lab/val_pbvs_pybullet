import numpy as np
import sophus as sp
from scipy.spatial.transform import Rotation as R
r = R.from_matrix([[0, -1, 0],
                   [1, 0, 0],
                   [0, 0, 1]])
print(type(r.as_quat()))
print(r.as_quat())
sp.SO3()
'''
SO3([[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]])
'''
T = sp.SE3(np.eye(3), np.arange(3))
print(type(T.translation()))
R =  sp.SO3([[0, 1, 0],
             [0, 0, 1],
             [1, 0, 0]])
R1 = sp.SO3([[0, 1, 0],
             [0, 0, 1],
             [1, 0, 0]])

print((R*R1).matrix())
print(type(R*R1))
print(type((T*T).matrix()))
#print(np.array(R*R1.matrix()).shape)