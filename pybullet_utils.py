import pybullet as p
import numpy as np
import sophus as sp
from scipy.spatial.transform import Rotation as R

# this mp4 recording requires ffmpeg installed
# mp4log = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,"humanoid.mp4")

def get_motor_joint_states(robot):
    joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
    joint_infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
    joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
    joint_names = [info[1] for info in joint_infos if info[3] > -1]
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]
    joint_torques = [state[3] for state in joint_states]
    return joint_positions, joint_velocities, joint_torques


def get_joint_states(robot):
    joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]
    joint_torques = [state[3] for state in joint_states]
    return joint_positions, joint_velocities, joint_torques


def draw_cross(pos, length=0.05, width=5):
    p.addUserDebugLine(np.asarray(np.array(pos) - np.array([length, 0, 0])),
                       np.asarray(np.array(pos) + np.array([length, 0, 0])),
                       lineColorRGB=[1.0, 0, 0],
                       lineWidth=width)
    p.addUserDebugLine(np.asarray(np.array(pos) - np.array([0, length, 0])),
                       np.asarray(np.array(pos) + np.array([0, length, 0])),
                       lineColorRGB=[0, 1.0, 0],
                       lineWidth=width)
    p.addUserDebugLine(np.asarray(np.array(pos) - np.array([0, 0, length])),
                       np.asarray(np.array(pos) + np.array([0, 0, length])),
                       lineColorRGB=[0, 0, 1.0],
                       lineWidth=width)


def draw_pose(trans, rot, uids=None, width=5, axis_len=0.1):
    unique_ids = []
    coords = np.array(p.getMatrixFromQuaternion(rot)).reshape(3, 3) * axis_len + np.array(trans).reshape(3, 1)
    colors = np.eye(3)
    if uids is None:
        for i in range(3):
            unique_ids.append(p.addUserDebugLine(trans,
                                                 np.asarray(coords[:, i]),
                                                 lineColorRGB=np.asarray(colors[:, i]),
                                                 lineWidth=width))
        return unique_ids
    else:
        for i in range(3):
            unique_ids.append(p.addUserDebugLine(trans,
                                                 np.asarray(coords[:, i]),
                                                 lineColorRGB=np.asarray(colors[:, i]),
                                                 lineWidth=width,
                                                 replaceItemUniqueId=uids[i]))


def erase_pos(line_ids):
    for line_id in line_ids:
        p.removeUserDebugItem(line_id)


def get_pos(robot, link):
    """
    :param robot: body unique id of robot
    :param link: linkID
    :return: Cartesian position of center of mass of link
    """
    result = p.getLinkState(robot,
                            link,
                            computeLinkVelocity=1,
                            computeForwardKinematics=1)
    link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
    return link_trn


def get_pose(robot, link):
    """
    :param robot: body unique id of robot
    :param link: linkID
    :return: Cartesian position of center of mass of link
             Cartesian rotation of center of mass of link in the form of se(3)
             i.e. angle-axis representation, axis vector scaled by angle
    """
    result = p.getLinkState(robot,
                            link,
                            computeLinkVelocity=1,
                            computeForwardKinematics=1)
    link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
    '''
    axis_angle = p.getAxisAngleFromQuaternion(link_rot)
    print(axis_angle)
    '''
    return link_trn, link_rot


def get_jacobian(robot, tool, loc_pos=[0.0] * 3, perturb=False, mu=0.1, sigma=0.1):
    """
    returns two 3 by 20 full jacobian, translational, and rotational velocity respectively
    for val, 20 actuated joints are ordered as follows:
            2 Torso joints:     [joint 56, joint 57]
            7 left arm joints:  [joint 41 - joint 47]
            2 left grippers  :  [leftgripper & leftgripper2]
            7 right arm joints: [joint 1 - joint 7]
            2 right grippers  : [rightgripper & rightgripper2]
    robot: body unique id of robot
    tool: linkID. The Jacobian is calculated w.r.t the CoM of linkID by default
    loc_pos: the point on the specified link to compute the jacobian for, in
            link local coordinates around its center of mass.
    """
    pos, vel, torq = get_motor_joint_states(robot)
    zero_vec = [0.0] * len(pos)

    if perturb:
        err = np.random.normal(mu, sigma, len(pos))
        jac_t, jac_r = p.calculateJacobian(robot, tool, loc_pos, list(pos + err), zero_vec, zero_vec)
        return np.array(jac_t), np.array(jac_r)

    jac_t, jac_r = p.calculateJacobian(robot, tool, loc_pos, pos, zero_vec, zero_vec)
    return np.array(jac_t), np.array(jac_r)


def get_arm_jacobian(robot, arm, tool, loc_pos=[0.0] * 3, perturb=False, mu=0.1, sigma=0.1):
    """
    return 6 by 7 jacobian of the 7 dof left or right arm
    robot: body unique id of robot
    arm: "left" or "right"
    tool: linkID. The Jacobian is calculated w.r.t the CoM of linkID by default
    loc_pos: the point on the specified link to compute the jacobian for, in
            link local coordinates around its center of mass.
    """
    jac_t, jac_r = get_jacobian(robot, tool, loc_pos, perturb=perturb, mu=mu, sigma=sigma)
    if arm == "left":
        return np.vstack((jac_t[:, 2:9], jac_r[:, 2:9]))
    elif arm == "right":
        return np.vstack((jac_t[:, 11:18], jac_r[:, 11:18]))
    else:
        print('''arm incorrect, input "left" or "right" ''')


def get_true_depth(depth_img, zNear, zFar):
    z_n = 2.0 * depth_img - 1.0
    return 2.0 * zNear * zFar / (zFar + zNear - z_n * (zFar - zNear))


def camera_test():
    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=45.0,
        aspect=1.0,
        nearVal=0.1,
        farVal=3.1)
    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=[0, 0, 3],
        cameraTargetPosition=[0, 0, 0],
        cameraUpVector=[1, 1, 0])
    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=224,
        height=224,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix)
    rgb_img = np.array(rgbImg)[:, :, :3]
    depth_img = np.array(depthImg)
    '''
    plt.imshow(rgb_img)
    plt.show()
    plt.figure()
    plt.imshow(depth_img)
    plt.show()
    plt.imshow(get_true_depth(depth_img, 0.1, 3.1))
    plt.show()
    '''
    return rgb_img, depth_img

def quat2se3(quat):
    axis, angle = p.getAxisAngleFromQuaternion(quat)
    return np.array(axis) * angle

def trans_rot2SE3(trans, rot):
    rotm = np.array(p.getMatrixFromQuaternion(rot)).reshape(3, 3)
    return sp.SE3(rotm, np.array(trans))

def SE32_trans_rot(pose):
    r = R.from_matrix(pose.rotationMatrix())
    return list(pose.translation()), list(r.as_quat())