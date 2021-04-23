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

    # Define and draw goal pose
    cur_pos, cur_rot = get_pose(val.robot, val.left_tool)
    goal_pos = tuple(np.asarray(np.array(cur_pos) + np.array([0.05, 0.05, -0.1])))
    goal_rot = p.getQuaternionSlerp(cur_rot, (0, 0, 0, 1), 0.4) # weird thing happens if it is 0.7
    draw_pose(cur_pos, cur_rot)
    draw_pose(goal_pos, goal_rot)

    '''
    for i in range(15):
    print(i)
    time.sleep(1)
    '''

    val.pbvs("left", goal_pos, goal_rot,
             kv=1.0,
             kw=0.8,
             eps_pos=0.005,
             eps_rot=0.05,
             plot_pose=True,
             perturb_jacobian=False,
             perturb_Jac_joint_mu=0.2,
             perturb_Jac_joint_sigma=0.2,
             perturb_orientation=False,
             mu_R=0.3,
             sigma_R=0.3,
             plot_result=True)
    p.disconnect()


if __name__ == "__main__":
    # execute only if run as a script
    main()
