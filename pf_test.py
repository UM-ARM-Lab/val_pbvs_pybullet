import pybullet as p
import time
import pybullet_data
import numpy as np
import sophus as sp
import matplotlib.pyplot as plt
from pybullet_utils import *
from visual_servo_pybullet import vs

def main():
    val = vs()

    # Set val initial pose
    # high cond number
    # init_pos = [[0.2]] * len(val.left_arm_joint_indices)
    init_pos2 = []
    for joint in val.left_arm_joints:
        init_pos2.append([val.home[joint]])
    init_pos = [[0.3]] * len(val.left_arm_joint_indices)
    p.resetJointStatesMultiDof(val.robot, jointIndices=val.left_arm_joint_indices, targetValues=init_pos2)
    time.sleep(0.5)

    val.pf_init(print_init_poses=True)
    time.sleep(5)

    p.disconnect()


if __name__ == "__main__":
    # execute only if run as a script
    main()
