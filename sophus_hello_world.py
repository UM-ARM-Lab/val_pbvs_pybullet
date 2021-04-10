import numpy as np
import sophus as sp

sp.SO3()
'''
SO3([[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]])
'''
sp.SE3(np.eye(3), np.ones(3))
R = sp.SO3()
R1 = sp.SO3([[0, 1, 0],
             [0, 0, 1],
             [1, 0, 0]])
print(np.array(R*R1.matrix()).shape)