# Pybullet Simulation for markerless Position-Based Visual Servoing (PBVS) of a robot arm
### **Author:** [Haoran Mike Cheng](https://www.linkedin.com/in/hrcheng/)
### **Email:** hrcheng@umich.edu
### **References:** [Project Report](https://drive.google.com/file/d/1RTxMGtyoRdckZbu1weEqGqMV1YU7K-99/view?usp=sharing)
### **video:** [link](https://drive.google.com/file/d/1CJzujuzSmLbED6PKnHZAKvR8Pr7eE_Hm/view?usp=sharing)
### ![Sample Image](https://drive.google.com/uc?export=view&id=1SN8ObK_XjMm3gslmQTcdCA_5doAEI9a0)

## Abstract
Light-weight service robot arms and humanoids usually have inaccurate kinematics model as opposed to industrial robots. Visual servoing tackles this problem by continuously measuring the robot and target state and creating a feedback signal from camera images. In this project, a **particle filter-based Position-Based Visual Servoing (PBVS)** method is implemented in **PyBullet** simulation. As opposed to methods that attach markers to the end-effector for pose estimate, our method uses particle filter, which **renders image at the pose of each particle**, and uses **HOG likelihood** with the actual image as the measurement update, without the need of attaching a marker. The result in simulation shows that the end-effector can reach the target pose within the error threshold. 

## Prerequisite
### **Linux**
The code is developed and tested on Ubuntu 20.04, other versions might also work with the following libraries installed. Numpy, Scipy, Matplotlib are required.

### **PyBullet**
Can use pip to install (this repo uses PyBullet 3.1.0), no issue encountered.[Official Documentation](https://pybullet.org/wordpress/)
 
### **sophuspy**
sophuspy library is used to perform Lie theory related calculations. 
Directly installation using pip has know issues 
Please use following steps to install **pybind11** locally, and then install **sophuspy**
[Installation step references](https://github.com/craigstar/SophusPy/issues/3)
```
git clone https://github.com/pybind/pybind11.git
cd pybind11
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install/
make -j8
make install
export pybind11_DIR= **local directory to pybind11**
eg: /home/mike/Pybullet/lib/pybind11/install/share/cmake/pybind11/
pip3 install --user sophuspy
```
### **scikit-image**
```
pip install scikit-image
```
## Examples 
Please run `demo_script.py` for sample codes to run the velocity controller, PBVS with added noise, and Particle-filter based PBVS

