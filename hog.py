import matplotlib.pyplot as plt
import numpy as np

from skimage.feature import hog
from skimage import data, exposure
from visual_servo_pybullet import vs
from pybullet_utils import *
import time

def hog_test(image):
    start = time.process_time()
    #start = time.time()
    fd = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), multichannel=True)
    print(time.process_time() - start)
    #print(f'Time: {time.time() - start}')
    print(fd.shape)
    # print(hog_image.shape)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), multichannel=True, visualize = True)
    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()


def main():
    val = vs()
    # set to eye-before-hand initial pose
    init_pos2 = []
    for joint in val.left_arm_joints:
        init_pos2.append([val.home[joint]])
    p.resetJointStatesMultiDof(val.robot, jointIndices=val.left_arm_joint_indices, targetValues=init_pos2)

    # image = data.astronaut()
    rgb, depth, seg = val.get_image()
    # mask of two left grippers
    mask = np.logical_and(seg != (val.jdict["leftgripper"]), seg != (val.jdict["leftgripper2"]))
    rgb[mask] = 255 # set to white
    plt.imshow(rgb)
    plt.show()
    plt.imshow(depth)
    plt.show()
    hog_test(rgb)

if __name__ == "__main__":
    # execute only if run as a script
    main()
