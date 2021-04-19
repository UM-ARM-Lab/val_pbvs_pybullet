from multiprocessing import Process, Value
import time
def comp_likelihood(img, shared_pose, shared_pf_fin):
    for i in range(10):
        print("multiprocessing calculation {} of 10".format(i+1))
        time.sleep(0.1)
    shared_pose.value = img
    shared_pf_fin.value = 1
    print("assigned shared_pf_fin_val, end of multiprocessing")
class v():
    def __init__(self):
        self.x = 0
        self.shared_pf_fin = Value('i', 0)
    def cart_vel_ctl(self, vel_cmd):
        motion = 0
        while self.shared_pf_fin.value == 0:
            self.x += vel_cmd
            motion += vel_cmd
            print("cart, x = {}".format(self.x))
            time.sleep(0.1)
        return motion

    def pbvs(self):
        cur_pos = 0
        cur_img = 0
        while 10 - self.x > 0.1:
            cur_img += 1
            vel_cmd = 0.008*(10.0 - cur_pos)
            self.shared_pf_fin.value = 0
            shared_pose = Value('d', 0.0)
            print("start multiprocessing")
            p_pf = Process(target=comp_likelihood, args=(cur_img, shared_pose, self.shared_pf_fin))
            p_pf.start()
            print("start vel controller")
            motion = self.cart_vel_ctl(vel_cmd)
            print("finish vel controller")
            p_pf.join()
            cur_pos = (shared_pose.value + self.x)/2
            print("cur_pos {}".format(cur_pos))
vs = v()
vs.pbvs()