import pickle
import numpy as np
import math
import matplotlib.pyplot as plt

#get the data I need,position on x-y koordinate,and the total velocity at each time step
def load_data_files(filepath,tra_num):
    with open(filepath,'rb') as fp:
        data = pickle.load(fp)
    curve_data = np.array(data[tra_num])
    cd = np.array(curve_data[1])
    return cd

def sample_time(pos_x,pos_y,vel_t):
    sum = 0
    for i in range(len(pos_x)-1):

        dis_x = pos_x[i+1] - pos_x[i]
        dis_y = pos_y[i+1] - pos_y[i]
        dis_t = math.sqrt(dis_x**2 + dis_y**2)
        vel = vel_t[i]
        sum = sum + dis_t/vel
    dt = sum / (len(pos_x) - 1)

    return dt

#Use the akman model(bicycle model),generate the trajectory with the information of acceleration and steering angle
def akman(acc,angle,tranum,num):
    cd = load_data_files('allAmsterdamerRing.pickle', tranum)
    pos_xa = cd[:, 2]
    x_init = pos_xa[num]
    pos_ya = cd[:, 3]
    y_init = pos_ya[num]
    vel_xa = cd[:, 5]
    vel_xinit = vel_xa[num]
    vel_ya = cd[:, 6]
    vel_yinit= vel_ya[num]
    pfai_init = math.atan((pos_ya[num+1]-pos_ya[num])/(pos_xa[num+1]-pos_xa[num]))
    beta_init = math.atan(vel_yinit/vel_xinit) - pfai_init
    vel_init = math.sqrt(vel_xinit **2 + vel_yinit**2)
    beta = []
    beta.append(beta_init)
    pre_x = []
    pre_x.append(x_init)
    pre_y = []
    pre_y.append(y_init)
    pfai = []
    pfai.append(pfai_init)
    vel = []
    vel.append(vel_init)
    for i in range(len(acc)):
        beta.append(math.atan(lr * math.tan(angle[i]) / l))
        pre_x.append(pre_x[i] + vel[i] * math.cos(pfai[i] + beta[i]) * dt)
        pre_y.append(pos_y[i] + vel[i] * math.sin(pfai[i] + beta[i]) * dt)
        pfai.append(pfai[i] + vel[i] * math.sin(beta[i]) * dt / lr)
        vel.append(vel[i] + acc[i] * dt)
    return beta,pre_x,pre_y,pfai,vel

def geo2(pos_x,pos_y,angle,num):

    for i in range(num, len(pos_x) - num):
        # compute the radius of curvature first
        k1 = (pos_y[i] - pos_y[i - num]) / (pos_x[i] - pos_x[i - num])
        k1_1 = -1 / k1
        x1 = (pos_x[i] + pos_x[i - num]) / 2
        y1 = (pos_y[i] + pos_y[i - num]) / 2

        k2 = (pos_y[i + num] - pos_y[i]) / (pos_x[i + num] - pos_x[i])
        k2_1 = -1 / k2
        x2 = (pos_x[i + num] + pos_x[i]) / 2
        y2 = (pos_y[i + num] + pos_y[i]) / 2

        x_1 = (k1_1 * x1 - k2_1 * x2 - y1 + y2) / (k1_1 - k2_1)
        y_1 = (-k2_1 * k1_1 * x1 + y1 * k2_1 + k1_1 * k2_1 * x2 - k1_1 * y2) / (k2_1 - k1_1)

        r = math.sqrt((x_1 - pos_x[i]) ** 2 + (y_1 - pos_y[i]) ** 2)

        # use the akman model to compute the steering angle
        tan_delta = math.sqrt(l * l / abs(r ** 2 - lr ** 2))
        delta = math.atan(tan_delta)
        angle.append(delta)
    return angle


def acc_computation(pos_x, pos_y, dt,num):
    for i in range(num, len(pos_x) - num):
        dx1 = pos_x[i] - pos_x[i -1]
        dy1 = pos_y[i] - pos_y[i -1]
        d1 = math.sqrt(dx1 ** 2 + dy1 ** 2)
        v1 = d1 / dt

        dx2 = pos_x[i + 1] - pos_x[i]
        dy2 = pos_y[i + 1] - pos_y[i]
        d2 = math.sqrt(dx2 ** 2 + dy2 ** 2)
        v2 = d2 / dt

        acc.append((v2 - v1) / dt)
    return acc

def generate_time(pos_x):
    time_step = []
    for i in range(num,len(pos_x) - num):
        time_step.append(i * dt)
    return time_step


def figure_plot(acc, angle, time_step):
    plt.figure()

    plt.subplot(2, 1, 1)
    plt.scatter(time_step, angle, s=1)
    plt.xlabel('time')
    plt.ylabel('steering_angle')

    plt.subplot(2, 1, 2)
    plt.scatter(time_step, acc, s=1)
    plt.xlabel('time')
    plt.ylabel('acceleration')

    plt.show()

def figure_plot2(pos_x,pos_y):
    plt.figure(figsize=(7, 7))
    plt.xlim((-50, 50))
    plt.ylim((-50, 50))
    plt.scatter(pos_x,pos_y,s=1)

    plt.show()

if __name__ == '__main__':
      lr = 1.25
      l = 2 * lr

      cd = load_data_files('allAmsterdamerRing.pickle',2)

      pos_x = cd[:,2]
      pos_y = cd[:,3]
      vel_t = cd[:,7]

      angle = []
      acc =[]
      num = 5

      dt = sample_time(pos_x,pos_y,vel_t)

      angle = geo2(pos_x,pos_y,angle,num)

      acc = acc_computation(pos_x,pos_y,dt,num)

      time_step = generate_time(pos_x)

      beta,pos_x,pos_y,pfai,vel = akman(acc,angle,tranum=2,num=num)

      figure_plot2(pos_x,pos_y)