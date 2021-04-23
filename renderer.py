import pybullet as p
import sophus as sp
import matplotlib.pyplot as plt
from pybullet_utils import *
import time


class renderer:
    def __init__(self):
        self.physicsClient = p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version
        startPos = [0, 0, -0.181]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot = p.loadURDF("hdt_michigan_description/urdf/left_gripper_no_housing.urdf", startPos,
                                startOrientation, physicsClientId=self.physicsClient)

        # set jdict, maps from joint name to joint index
        self.jdict = {}
        for i in range(p.getNumJoints(self.robot, physicsClientId=self.physicsClient)):
            info = p.getJointInfo(self.robot, i, physicsClientId=self.physicsClient)
            jname = info[1].decode("ascii")
            self.jdict[jname] = i
        print(self.jdict)
        self.joint_ids = [self.jdict["leftgripper"], self.jdict["leftgripper2"]]

        # camera stuff
        self.z_near = 0.1
        self.z_far = 3.1
        self.projectionMatrix = p.computeProjectionMatrixFOV(
            fov=65.0,
            aspect=1.0,
            nearVal=self.z_near,
            farVal=self.z_far)

    def render_img(self, left_gripper_joint, left_gripper2_joint, Tce, plot_cam_pose=False, imsize=(214, 214),
                   plot_img=False, show_time_taken = False):
        if show_time_taken:
            start = time.time()
        p.resetJointStatesMultiDof(self.robot, jointIndices=self.joint_ids,
                                   targetValues=[[left_gripper_joint], [left_gripper2_joint]],
                                   physicsClientId=self.physicsClient)
        Tec = Tce.inverse()
        cam_trans, cam_rot = SE32_trans_rot(Tec)
        cam_rotm = Tec.rotationMatrix()
        if plot_cam_pose:
            draw_pose(cam_trans, cam_rot)
        # Calculate extrinsic matrix
        # target position is camera frame y axis tip in global frame
        # up vector is the z axis of camera frame
        viewMatrix = p.computeViewMatrix(
            cameraEyePosition=cam_trans,
            cameraTargetPosition=(cam_rotm.dot(np.array([0, 1, 0]).T) + np.array(cam_trans)).tolist(),
            cameraUpVector=cam_rotm[:, 2].tolist())
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=imsize[0],
            height=imsize[1],
            viewMatrix=viewMatrix,
            projectionMatrix=self.projectionMatrix,
            flags=p.ER_NO_SEGMENTATION_MASK,
            physicsClientId=self.physicsClient)

        if show_time_taken:
            print("time taken for rendering: {}".format(time.time() - start))
        if plot_img:
            plt.imshow(rgbImg[:, :, :3])
            plt.show()
        return rgbImg[:, :, :3]


def main():
    rd = renderer()
    Tce = sp.SE3([[0, 1, 0, 0],
                  [-1, 0, 0, 0.3],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    rgb = rd.render_img(0.2, 0.2, Tce, plot_img=True)

    p.disconnect(rd.physicsClient)


if __name__ == "__main__":
    # execute only if run as a script
    main()
