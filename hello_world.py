import pybullet as p
import time
import pybullet_data
import numpy as np


# this mp4 recording requires ffmpeg installed
# mp4log = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,"humanoid.mp4")

def get_motor_joint_states(robot):
    joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
    joint_infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
    joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
    joint_names = [info[1] for info in joint_infos if info[3] > -1]
    print(joint_names, len(joint_names))
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


def draw_cross(pos, length=0.05, width=5, color=[1.0, 0, 0]):
    p.addUserDebugLine(np.asarray(np.array(pos) - np.array([length, 0, 0])),
                       np.asarray(np.array(pos) + np.array([length, 0, 0])),
                       lineColorRGB=color,
                       lineWidth=width)
    p.addUserDebugLine(np.asarray(np.array(pos) - np.array([0, length, 0])),
                       np.asarray(np.array(pos) + np.array([0, length, 0])),
                       lineColorRGB=color,
                       lineWidth=width)


def get_pos(robot, link):
    """
    :param robot: body unique id of robot
    :param link: linkID
    :return: Cartesian position of center of mass of link
    """
    result = p.getLinkState(val,
                            left_tool,
                            computeLinkVelocity=1,
                            computeForwardKinematics=1)
    link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
    return link_trn


# return the
def get_jacobian(robot, tool, loc_pos=[0.0] * 3):
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
    jac_t, jac_r = p.calculateJacobian(robot, tool, loc_pos, pos, zero_vec, zero_vec)
    return np.array(jac_t), np.array(jac_r)


def get_arm_jacobian(robot, arm, tool, loc_pos=[0.0] * 3):
    """
    return 6 by 7 jacobian of the 7 dof left or right arm
    robot: body unique id of robot
    arm: "left" or "right"
    tool: linkID. The Jacobian is calculated w.r.t the CoM of linkID by default
    loc_pos: the point on the specified link to compute the jacobian for, in
            link local coordinates around its center of mass.
    """
    jac_t, jac_r = get_jacobian(robot, tool, loc_pos)
    if arm == "left":
        return jac_t[:, 2:9], jac_r[:, 2:9]
    elif arm == "right":
        return jac_t[:, 11:18], jac_r[:, 11:18]
    else:
        print('''arm incorrect, input "left" or "right" ''')

##################### initialize ##################
###vars to init
left_arm_joint_indices = []
right_arm_joint_indices = []
jdict = {}
left_tool = 0
right_tool = 0
###
physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setGravity(0, 0, -9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")
startPos = [0, 0, 0.2]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
val = p.loadURDF("hdt_michigan_description/urdf/hdt_michigan_generated.urdf", startPos, startOrientation)

# set jdict, maps from joint name to joint index
for i in range(p.getNumJoints(val)):
    info = p.getJointInfo(val, i)
    jname = info[1].decode("ascii")
    jdict[jname] = i
print(jdict)

# set left/right_arm_joint_indices
for i in range(1, 8):
    left_arm_joint_indices.append(jdict["joint4" + str(i)])
    right_arm_joint_indices.append(jdict["joint" + str(i)])

# index of the link at the left/right gripper tip when gripper is closed
left_tool = jdict["left_tool_joint"]
right_tool = jdict["right_tool_joint"]
##################### finish initialize ##################

pos = get_pos(val, left_tool)
draw_cross(pos)

init_pos = [[-0.2]] * len(left_arm_joint_indices)
# p.resetJointStatesMultiDof(val, jointIndices=left_arm_joint_indices, targetValues=init_pos)
time.sleep(0.5)

pos = get_pos(val, left_tool)
draw_cross(pos)

# p.addUserDebugLine([0.5,0.5,0.5], [0.62,0.52,0.52], lineColorRGB = [1.0, 0, 0], lifeTime = 4)
jac_trn, jac_rot = get_jacobian(val, left_tool)
print(jac_trn)
print(jac_rot)
jac_trn, jac_rot = get_arm_jacobian(val, "left", left_tool)
print(jac_trn)
print(jac_rot)

for i in range(1000):
    targetVelocities = [0.05] * len(left_arm_joint_indices)
    p.setJointMotorControlArray(val, left_arm_joint_indices, controlMode=p.VELOCITY_CONTROL,
                                targetVelocities=targetVelocities)
    p.stepSimulation()
    # print(p.getJointStates(val, left_arm_joint_indices))
    time.sleep(1. / 240.)

pos = get_pos(val, left_tool)
draw_cross(pos)

time.sleep(100)
p.disconnect()
