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
    init_pos = []
    for joint in val.left_arm_joints:
        init_pos.append([val.home[joint]])
    p.resetJointStatesMultiDof(val.robot, jointIndices=val.left_arm_joint_indices, targetValues=init_pos)
    time.sleep(0.5)

    # joint velocity controller test - linear velocity
    print("joint velocity controller test - linear velocity")
    val.cart_vel_linear_test(t=1.5, v=0.05)

    # joint velocity controller test - angular velocity
    print("joint velocity controller test - angular velocity")
    val.cart_vel_angular_test(t=1.5, w=0.25)

    # Define and draw goal pose
    cur_pos, cur_rot = get_pose(val.robot, val.left_tool)
    goal_pos_global = tuple(np.asarray(np.array(cur_pos) + np.array([0.05, 0.05, -0.1])))
    goal_rot_global = p.getQuaternionSlerp(cur_rot, (0, 0, 0, 1), 0.4) # weird thing happens if it is 0.7
    draw_pose(cur_pos, cur_rot)
    draw_pose(goal_pos_global, goal_rot_global)
    # transform goal_pose to camera frame
    goal_pos, goal_rot = SE32_trans_rot(val.Trc.inverse() * trans_rot2SE3(goal_pos_global, goal_rot_global))

    # Position Based Visual Servoing with no perturbation
    print("Position Based Visual Servoing with no perturbation")
    val.pbvs("left", goal_pos, goal_rot,
             kv=1.0,
             kw=0.8,
             eps_pos=0.005,
             eps_rot=0.05,
             plot_pose=True,
             perturb_jacobian=False,
             perturb_orientation=False,
             plot_result=True)

    # reset to home pose
    p.resetJointStatesMultiDof(val.robot, jointIndices=val.left_arm_joint_indices, targetValues=init_pos)
    draw_pose(cur_pos, cur_rot)
    draw_pose(goal_pos_global, goal_rot_global)

    # Position Based Visual Servoing with perturbation in Jacobian
    print("Position Based Visual Servoing with perturbation in Jacobian")
    val.pbvs("left", goal_pos, goal_rot,
             kv=1.0,
             kw=0.8,
             eps_pos=0.005,
             eps_rot=0.05,
             plot_pose=True,
             perturb_jacobian=True,
             perturb_Jac_joint_mu=0.2,
             perturb_Jac_joint_sigma=0.2,
             perturb_orientation=False,
             mu_R=0.3,
             sigma_R=0.3,
             plot_result=True)

    # reset to home pose
    p.resetJointStatesMultiDof(val.robot, jointIndices=val.left_arm_joint_indices, targetValues=init_pos)
    draw_pose(cur_pos, cur_rot)
    draw_pose(goal_pos_global, goal_rot_global)

    # execute Particle-Filter based PBVS, using default PBVS parameters which can be changed
    val.pbvs_pf("left", goal_pos, goal_rot)

    p.disconnect()


if __name__ == "__main__":
    # execute only if run as a script
    main()
