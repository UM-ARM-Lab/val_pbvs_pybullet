import pybullet as p
import time
import pybullet_data
import numpy as np
import sophus as sp
import matplotlib.pyplot as plt
from pybullet_utils import *


class vs():
    def __init__(self,
                 perturb_Jac_joint_mu=0.1,
                 perturb_Jac_joint_sigma=0.1,
                 mu_R=0.4,
                 sigma_R=0.4,
                 perturb_jacobian=False,
                 perturb_orientation=False,
                 ):
        ##################### initialize ##################
        ###vars to init
        # Initialize variables
        self.perturb_Jac_joint_mu = perturb_Jac_joint_mu
        self.perturb_Jac_joint_sigma = perturb_Jac_joint_sigma
        self.perturb_R_mu = mu_R
        self.perturb_R_sigma = sigma_R
        self.perturb_jacobian = perturb_jacobian
        self.perturb_orientation = perturb_orientation

        self.left_arm_joint_indices = []
        self.right_arm_joint_indices = []
        self.jdict = {}
        self.left_tool = 0
        self.right_tool = 0
        # set joint names
        self.right_arm_joints = [
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6',
            'joint7',
        ]
        self.left_arm_joints = [
            'joint41',
            'joint42',
            'joint43',
            'joint44',
            'joint45',
            'joint46',
            'joint47',
        ]
        self.home = {
            'joint56': -1.55,
            'joint57': 0.077,
            'joint41': 0.25829190015792847,
            'joint42': 0.05810129642486572,
            'joint43': -0.5179260969161987,
            'joint44': 0.29261577129364014,
            'joint45': 1.4885820150375366,
            'joint46': 0.9461115002632141,
            'joint47': -3.719825267791748,
            'joint1': 0.5208024382591248,
            'joint2': -0.030105292797088623,
            'joint3': 0.42895248532295227,
            'joint4': -0.08494678139686584,
            'joint5': 6.152984619140625 - 6.28,
            'joint6': 0.6138027906417847,
            'joint7': -1.5069904327392578
        }
        self.z_near = 0.1
        self.z_far = 3.1
        self.projectionMatrix = p.computeProjectionMatrixFOV(
            fov=65.0,
            aspect=1.0,
            nearVal=self.z_near,
            farVal=self.z_far)

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
                              damped=True,
                              l_damped=0.01,
                              v_threshold=0.15,
                              omega_threshold=0.6,
                              joint_vel_threshold=1.5,
                              cond_threshold=100000,
                              show_cond=False,
                              perturb_jacobian=False
                              ):
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
        self.perturb_jacobian = perturb_jacobian
        # Initialize tool and arm_joint_indices
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
                                 perturb=self.perturb_jacobian, mu=self.perturb_Jac_joint_mu,
                                 sigma=self.perturb_Jac_joint_sigma)  # get current jacobian

            # calculate desired joint velocity (by multiplying jacobian pseudo-inverse), redundant -> min energy path
            if damped:
                # damped least-squares to make it invertible when near-singular
                J = np.array(J)
                joint_vels = list(np.linalg.inv(J.T.dot(J) + l_damped ** 2 * np.eye(J.shape[1])).dot(J.T).dot(
                    np.array(twist)).reshape(-1))
            else:
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

    # eps_pos = 0.005 eps_rpt = 0.05
    def pbvs(self,
             arm,
             goal_pos,
             goal_rot,
             kv=1.3,
             kw=0.8,
             eps_pos=0.005,
             eps_rot=0.05,
             vs_rate=20,
             plot_pose=False,
             camera_update=False,
             perturb_jacobian=False,
             perturb_Jac_joint_mu=0.1,
             perturb_Jac_joint_sigma=0.1,
             perturb_orientation=False,
             mu_R=0.4,
             sigma_R=0.4,
             plot_result=False):

        # Initialize variables
        self.perturb_Jac_joint_mu = perturb_Jac_joint_mu
        self.perturb_Jac_joint_sigma = perturb_Jac_joint_sigma
        self.perturb_R_mu = mu_R
        self.perturb_R_sigma = sigma_R
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

        # record end effector pose and time stamp at each iteration
        if plot_result:
            start = time.time()
            times = []
            eef_pos = []
            eef_rot = []

        ##### Control loop starts #####
        while True:
            if camera_update:
                camera_test()
            cur_pos, cur_rot = get_pose(self.robot, tool)
            if plot_pose:
                draw_pose(cur_pos, cur_rot, uids=pos_plot_id)
            if plot_result:
                eef_pos.append(cur_pos)
                eef_rot.append(cur_rot)
                times.append(time.time() - start)
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
                dR = sp.SO3.exp(np.random.normal(self.perturb_R_mu, self.perturb_R_sigma, 3))
                Rsc = Rsc.dot(dR.matrix())

            twist_local = np.hstack((np.array(pos_cg) * kv, np.array(quat2se3(rot_cg)) * kw)).reshape(6, 1)
            local2global = np.block([[Rsc, np.zeros((3, 3))],
                                     [np.zeros((3, 3)), Rsc]])
            twist_global = local2global.dot(twist_local)
            self.cartesian_vel_control(arm, np.asarray(twist_global), 1 / vs_rate,
                                       show_cond=False,
                                       perturb_jacobian=perturb_jacobian)
        print("PBVS goal achieved!")
        if plot_result:
            eef_pos = np.array(eef_pos)
            eef_rot_rpy = np.array([p.getEulerFromQuaternion(quat) for quat in eef_rot])

            fig, axs = plt.subplots(3, 2)

            sub_titles = [['x', 'roll'], ['y', 'pitch'], ['z', 'yaw']]
            fig.suptitle("Position Based Visual Servo End Effector Pose - time plot")
            for i in range(3):
                axs[i, 0].plot(times, eef_pos[:, i] * 100)
                axs[i, 0].plot(times, goal_pos[i] * np.ones(len(times)) * 100)
                axs[i, 0].legend([sub_titles[i][0], 'goal'])
                axs[i, 0].set_xlabel('Time(s)')
                axs[i, 0].set_ylabel('cm')
                # axs[i, 0].set_title(sub_titles[i][0])
            goal_rpy = p.getEulerFromQuaternion(goal_rot)
            print("rpy final error: ")
            # [-0.0056446   0.02230498  0.0133852 ]
            # [-0.0152125   0.03453246  0.01255567]
            # [-0.01669663  0.03247189  0.02033215]

            print(eef_rot_rpy[-1, :] - goal_rpy)
            for i in range(3):
                axs[i, 1].plot(times, eef_rot_rpy[:, i] * 180 / np.pi)
                axs[i, 1].plot(times, goal_rpy[i] * np.ones(len(times)) * 180 / np.pi)
                axs[i, 1].legend([sub_titles[i][1], 'goal'])
                axs[i, 1].set_xlabel('Time(s)')
                axs[i, 1].set_ylabel('deg')
                # axs[i, 1].set_title(sub_titles[i][1])

            '''
            plt.subplot(2, 1, 2)
            plt.plot(times, eef_rot_rpy)
            '''
            for ax in axs.flat:
                ax.set(xlabel='time')

            plt.show()

    def cart_vel_linear_test(sself, t=1.5, v=0.05):
        sself.cartesian_vel_control('left', [-v, 0, 0, 0, 0, 0], t)
        time.sleep(1)
        sself.cartesian_vel_control('left', [v, 0, 0, 0, 0, 0], t)
        time.sleep(1)
        sself.cartesian_vel_control('left', [0, v, 0, 0, 0, 0], t)
        time.sleep(1)
        sself.cartesian_vel_control('left', [0, -v, 0, 0, 0, 0], t)
        time.sleep(1)
        sself.cartesian_vel_control('left', [0, 0, v, 0, 0, 0], t)
        time.sleep(1)
        sself.cartesian_vel_control('left', [0, 0, -v, 0, 0, 0], t)
        time.sleep(2)

    def cart_vel_angular_test(self, t2=1.5, w=0.25):
        self.cartesian_vel_control('left', [0, 0, 0, w, 0, 0], t2)
        time.sleep(1)
        self.cartesian_vel_control('left', [0, 0, 0, -w, 0, 0], t2)
        time.sleep(1)
        self.cartesian_vel_control('left', [0, 0, 0, 0, w, 0], t2)
        time.sleep(1)
        self.cartesian_vel_control('left', [0, 0, 0, 0, -w, 0], t2)
        time.sleep(1)
        self.cartesian_vel_control('left', [0, 0, 0, 0, 0, w], t2)
        time.sleep(1)
        self.cartesian_vel_control('left', [0, 0, 0, 0, 0, -w], t2)
        time.sleep(2)

    def get_image(self, imsize=(214, 214), plot_cam_pose = True):
        # Get "camera link pose", but currently it coincides with torso
        cam_trans, cam_rot = get_pose(self.robot, self.jdict["realsense_joint"])
        cam_rotm = np.array(p.getMatrixFromQuaternion(cam_rot)).reshape(3, 3)
        # Double check whether it is true if torso joints move
        cam_trans = (cam_rotm.dot(np.array([0.035, 0.032, 0.521])) + np.array(cam_trans)).tolist()
        if plot_cam_pose:
            draw_pose(cam_trans, cam_rot)
        # Calculate extrinsic matrix
        # target position is camera frame y axis tip in global frame
        # up vector is the z axis of camera frame
        viewMatrix = p.computeViewMatrix(
            cameraEyePosition=cam_trans,
            cameraTargetPosition=(cam_rotm.dot(cam_rotm[1, :]) + np.array(cam_trans)).tolist(),
            cameraUpVector=cam_rotm[2, :].tolist())
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=imsize[0],
            height=imsize[1],
            viewMatrix=viewMatrix,
            projectionMatrix=self.projectionMatrix,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
        # depth image transformed to true depth, each pixel in segImg is the linkID
        return rgbImg[:, :, :3], get_true_depth(depthImg, self.z_near, self.z_far), ((segImg >> 24) - 1)