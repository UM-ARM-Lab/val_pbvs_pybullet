import pybullet as p
import time
import pybullet_data
import numpy as np
import sophus as sp
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


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


def draw_pose(trans, rot, uids=None, relength=0.05, width=5, axis_len=0.1):
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


# return the
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


def get_arm_jacobian(robot, arm, tool, loc_pos=[0.0] * 3, perturb=False):
    """
    return 6 by 7 jacobian of the 7 dof left or right arm
    robot: body unique id of robot
    arm: "left" or "right"
    tool: linkID. The Jacobian is calculated w.r.t the CoM of linkID by default
    loc_pos: the point on the specified link to compute the jacobian for, in
            link local coordinates around its center of mass.
    """
    jac_t, jac_r = get_jacobian(robot, tool, loc_pos, perturb=perturb)
    if arm == "left":
        return np.vstack((jac_t[:, 2:9], jac_r[:, 2:9]))
    elif arm == "right":
        return np.vstack((jac_t[:, 11:18], jac_r[:, 11:18]))
    else:
        print('''arm incorrect, input "left" or "right" ''')


def quat2se3(quat):
    axis, angle = p.getAxisAngleFromQuaternion(quat)
    return np.array(axis) * angle


class vs():
    def __init__(self):
        ##################### initialize ##################
        ###vars to init
        self.left_arm_joint_indices = []
        self.right_arm_joint_indices = []
        self.jdict = {}
        self.left_tool = 0
        self.right_tool = 0
        ###
        physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetDebugVisualizerCamera(cameraDistance=1.4, cameraYaw=216,
                                     cameraPitch=-49.6, cameraTargetPosition=[0, 0, 0])
        planeId = p.loadURDF("plane.urdf")
        startPos = [0, 0, 0.2]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot = p.loadURDF("hdt_michigan_description/urdf/hdt_michigan_generated.urdf", startPos, startOrientation)

        # set jdict, maps from joint name to joint index
        for i in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, i)
            jname = info[1].decode("ascii")
            self.jdict[jname] = i

        # set left/right_arm_joint_indices
        for i in range(1, 8):
            self.left_arm_joint_indices.append(self.jdict["joint4" + str(i)])
            self.right_arm_joint_indices.append(self.jdict["joint" + str(i)])

        # index of the link at the left/right gripper tip when gripper is closed
        self.left_tool = self.jdict["left_tool_joint"]
        self.right_tool = self.jdict["right_tool_joint"]
        ##################### finish initialize ##################

    def cartesian_vel_control(self,
                              arm,
                              twist,
                              duration,
                              r=100,
                              v_threshold=0.15,
                              omega_threshold=0.6,
                              joint_vel_threshold=1.5,
                              cond_threshold=100000,
                              show_cond=False,
                              perturb_jacobian=False):
        """
        Cartesian end-effector controller
        requires:
        group_name: 'right_arm' or 'left_arm'
        twist: [vx,vy,vz,wx,wy,wz], in global frame
        duration: execution time in sec
        r: control bandwidth (loop rate)
        v_threshold: end-effector max linear velocity magnitude
        omega_threshold: end-effector max angular velocity magnitude
        """
        if arm == "right":
            tool = self.right_tool
            arm_joint_indices = self.right_arm_joint_indices
        elif arm == "left":
            tool = self.left_tool
            arm_joint_indices = self.left_arm_joint_indices
        else:
            print('''arm incorrect, input "left" or "right" ''')
            return

        # cartesian velocity safety check
        if np.linalg.norm(np.array(twist[:3])) > v_threshold:
            print("linear velocity greater than threshold {} !".format(v_threshold))
            print("Current velocity: {}".format(np.linalg.norm(np.array(twist[:3]))))
            return

        if np.linalg.norm(np.array(twist[3:])) > omega_threshold:
            print("angular velocity greater than threshold {} !".format(omega_threshold))
            print("Current angular velocity: {}".format(np.linalg.norm(np.array(twist[3:]))))
            return

        # control loop
        for i in range(int(r * duration)):
            # calculate joint_vels
            J = get_arm_jacobian(self.robot, arm, tool, loc_pos=[0.0] * 3,
                                 perturb=perturb_jacobian)  # get current jacobian
            # calculate desired joint velocity (by multiplying jacobian pseudo-inverse), redundant -> min energy path
            joint_vels = list(np.linalg.pinv(np.array(J)).dot(np.array(twist)).reshape(-1))

            # joint velocity safety check
            if max(joint_vels) > joint_vel_threshold:
                print("Highest joint velocity is {:.2f}, greater than {:.2f}, stopping!".format(max(joint_vels),
                                                                                                joint_vel_threshold))
                break

            # condition number of JJ', used for safety check
            cond = np.linalg.cond(J.dot(J.T))
            cond2 = np.linalg.cond(J[:3, :].dot(J[:3, :].T))
            # print(J.dot(J.T))
            if show_cond:
                print("Conditional number of JJ' {:.2f}".format(cond))
                print("Conditional number of JJ' cart {:.2f}".format(cond2))
            if cond > cond_threshold:
                print("Large conditional number! {:.2f}".format(cond))
                break

            p.setJointMotorControlArray(self.robot, arm_joint_indices, controlMode=p.VELOCITY_CONTROL,
                                        targetVelocities=joint_vels)
            p.stepSimulation()
            time.sleep(1. / r)

    def pbvs(self,
             arm,
             goal_pos,
             goal_rot,
             kv=0.9,
             kw=0.5,
             eps_pos=0.005,
             eps_rot=0.05,
             vs_rate=20,
             joint_vel_threshold=1.5,
             cond_threshold=10000,
             plot_pose=False,
             camera_update=False,
             perturb_jacobian=False,
             perturb_orientation=False,
             mu_R=0.4,
             sigma_R=0.4):

        if arm == "right":
            tool = self.right_tool
            arm_joint_indices = self.right_arm_joint_indices
        elif arm == "left":
            tool = self.left_tool
            arm_joint_indices = self.left_arm_joint_indices
        else:
            print('''arm incorrect, input "left" or "right" ''')
            return
        cur_pos, cur_rot = get_pose(self.robot, tool)
        pos_plot_id = draw_pose(cur_pos, cur_rot)
        while True:
            if camera_update:
                camera_test()
            cur_pos, cur_rot = get_pose(self.robot, tool)
            if plot_pose:
                draw_pose(cur_pos, cur_rot, uids=pos_plot_id)
            cur_pos_inv, cur_rot_inv = p.invertTransform(cur_pos, cur_rot)
            # Pose of goal in camera frame
            pos_cg, rot_cg = p.multiplyTransforms(cur_pos_inv, cur_rot_inv, goal_pos, goal_rot)
            # Evaluate current translational and rotational error
            err_pos = np.linalg.norm(pos_cg)
            err_rot = np.linalg.norm(p.getAxisAngleFromQuaternion(rot_cg)[1])
            if err_pos < eps_pos and err_rot < eps_rot:
                break
            else:
                print("Error to goal, position: {:2f}, orientation: {:2f}".format(err_pos, err_rot))
            Rsc = np.array(p.getMatrixFromQuaternion(cur_rot)).reshape(3, 3)

            # Perturb Rsc in SO(3) by a random variable in tangent space so(3)
            if perturb_orientation:
                dR = sp.SO3.exp(np.random.normal(mu_R, sigma_R, 3))
                Rsc = Rsc.dot(dR.matrix())
            twist_local = np.hstack((np.array(pos_cg) * kv, np.array(quat2se3(rot_cg)) * kw)).reshape(6, 1)
            local2global = np.block([[Rsc, np.zeros((3, 3))],
                                     [np.zeros((3, 3)), Rsc]])
            twist_global = local2global.dot(twist_local)
            self.cartesian_vel_control(arm, np.asarray(twist_global), 1 / vs_rate,
                                       show_cond=False,
                                       perturb_jacobian=perturb_jacobian)
        print("PBVS goal achieved!")


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


def cart_vel_linear_test(val, t=1.5, v=0.05):
    val.cartesian_vel_control('left', [-v, 0, 0, 0, 0, 0], t)
    time.sleep(1)
    val.cartesian_vel_control('left', [v, 0, 0, 0, 0, 0], t)
    time.sleep(1)
    val.cartesian_vel_control('left', [0, v, 0, 0, 0, 0], t)
    time.sleep(1)
    val.cartesian_vel_control('left', [0, -v, 0, 0, 0, 0], t)
    time.sleep(1)
    val.cartesian_vel_control('left', [0, 0, v, 0, 0, 0], t)
    time.sleep(1)
    val.cartesian_vel_control('left', [0, 0, -v, 0, 0, 0], t)
    time.sleep(2)


def cart_vel_angular_test(val, t2=1.5, w=0.25):
    val.cartesian_vel_control('left', [0, 0, 0, w, 0, 0], t2)
    time.sleep(1)
    val.cartesian_vel_control('left', [0, 0, 0, -w, 0, 0], t2)
    time.sleep(1)
    val.cartesian_vel_control('left', [0, 0, 0, 0, w, 0], t2)
    time.sleep(1)
    val.cartesian_vel_control('left', [0, 0, 0, 0, -w, 0], t2)
    time.sleep(1)
    val.cartesian_vel_control('left', [0, 0, 0, 0, 0, w], t2)
    time.sleep(1)
    val.cartesian_vel_control('left', [0, 0, 0, 0, 0, -w], t2)
    time.sleep(2)


def main():
    val = vs()

    # Set val initial pose
    # high cond number
    # init_pos = [[0.2]] * len(val.left_arm_joint_indices)
    init_pos = [[0.2]] * len(val.left_arm_joint_indices)
    p.resetJointStatesMultiDof(val.robot, jointIndices=val.left_arm_joint_indices, targetValues=init_pos)
    time.sleep(0.5)

    # Optional cartesian velocity controller test
    cart_vel_linear_test(val)
    cart_vel_angular_test(val)

    # Get Jacobian testing script
    jac_trn, jac_rot = get_jacobian(val.robot, val.left_tool)
    jac = get_arm_jacobian(val.robot, "left", val.left_tool)

    # Get pose testing script
    pos, rot = get_pose(val.robot, val.left_tool)
    print("axis angle result: {}".format(rot))

    # Define and draw goal pose
    cur_pos, cur_rot = get_pose(val.robot, val.left_tool)
    goal_pos = tuple(np.asarray(np.array(cur_pos) + np.array([0.05, -0.05, 0.07])))
    goal_rot = p.getQuaternionSlerp(cur_rot, (0, 0, 0, 1), 0.3)
    draw_pose(cur_pos, cur_rot)
    draw_pose(goal_pos, goal_rot)

    # camera test
    camera_test()

    val.pbvs("left", goal_pos, goal_rot, plot_pose=True, perturb_jacobian=True, perturb_orientation=False)
    time.sleep(100)
    p.disconnect()


if __name__ == "__main__":
    # execute only if run as a script
    main()
