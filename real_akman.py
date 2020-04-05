import pickle
import numpy as np
import math
import matplotlib.pyplot as plt

#get the data I need,position on x-y koordinate,and the total velocity at each time step
def load_data_files(filepath,tra_num):
    with open(filepath,'rb') as fp:
        data = pickle.load(fp)
    curve_data = data[tra_num]
    cd = curve_data[1]
    return cd

#computation of delta_t
def sample_time(pos_x,pos_y,vel_t):
    sum = 0
    for i in range(len(pos_x)-1):

        dis_x = pos_x[i+1] - pos_x[i]
        dis_y = pos_y[i+1] - pos_y[i]
        dis_t = math.sqrt(dis_x**2 + dis_y**2)
        vel = vel_t[i]
        if vel != 0:
          sum = sum + dis_t/vel
        else:
          sum = sum + 0.2
    dt = sum / (len(pos_x) - 1)

    return dt


#Use the akman model(bicycle model),generate the trajectory with the information of acceleration and steering angle
def akman(vel,angle,tranum,num):
    cd = load_data_files('allAmsterdamerRing.pickle', tranum)

    pos_xa = np.array(cd['pos_x'])
    x_init = pos_xa[num]

    pos_ya = np.array(cd['pos_y'])
    y_init = pos_ya[num]

    vel_xa = np.array(cd['vel_x'])
    vel_xinit = vel_xa[num]

    vel_ya = np.array(cd['vel_y'])
    vel_yinit = vel_ya[num]

    x_zeichen = pos_xa[num+3] - pos_xa[num]
    y_zeichen = pos_ya[num+3] - pos_ya[num]

    if x_zeichen > 0 :
        pfai_init = math.atan(y_zeichen/x_zeichen)
    else:
        pfai_init = math.atan(y_zeichen/x_zeichen) + math.pi

    if vel_xinit != 0:
        beta_init = math.atan(vel_yinit/vel_xinit) - pfai_init
    else:
        beta_init = 0

    beta = []
    beta.append(beta_init)
    pre_x = []
    pre_x.append(x_init)
    pre_y = []
    pre_y.append(y_init)
    pfai = []
    pfai.append(pfai_init)


    for i in range(len(vel)):


        pre_x.append(pre_x[i] + vel[i] * math.cos(pfai[i] + beta[i]) * dt)
        pre_y.append(pre_y[i] + vel[i] * math.sin(pfai[i] + beta[i]) * dt)

        beta.append(math.atan(lr * math.tan(angle[i]) / l))
        pfai.append(pfai[i] + vel[i] * math.sin(beta[i]) * dt / lr)

    return pre_x,pre_y


#computation of steering angle
def geo2(pos_x,pos_y):

    num = 0
    dist = 0
    while (dist < dis):
        num = num + 1
        dist = math.sqrt((pos_x[num] - pos_x[0])**2 + (pos_y[num] - pos_y[0])**2)

    num_1 = -1
    dist_1 = 0
    while (dist_1 < dis):
        num_1 = num_1 - 1
        dist_1 = math.sqrt((pos_x[num_1] - pos_x[-1]) ** 2 + (pos_y[num_1] - pos_y[-1]) ** 2)
    num_1 = len(pos_x) + num_1
    for i in range(num,num_1+1):

       nub = i
       diss = 0
       while (diss < dis):
           nub = nub - 1
           diss = math.sqrt((pos_x[nub] - pos_x[i]) ** 2 + (pos_y[nub] - pos_y[i]) ** 2)

       nub_1 = i
       diss_1 = 0
       while (diss_1 < dis):
           nub_1 = nub_1 + 1
           diss_1 = math.sqrt((pos_x[nub_1] - pos_x[i]) ** 2 + (pos_y[nub_1] - pos_y[i]) ** 2)
       k1 = (pos_y[i] - pos_y[nub]) / (pos_x[i] - pos_x[nub])
       k1_1 = -1 / k1
       x1 = (pos_x[i] + pos_x[nub]) / 2
       y1 = (pos_y[i] + pos_y[nub]) / 2

       k2 = (pos_y[nub_1] - pos_y[i]) / (pos_x[nub_1] - pos_x[i])
       k2_1 = -1 / k2
       x2 = (pos_x[nub_1] + pos_x[i]) / 2
       y2 = (pos_y[nub_1] + pos_y[i]) / 2

       x_1 = (k1_1 * x1 - k2_1 * x2 - y1 + y2) / (k1_1 - k2_1)
       y_1 = (-k2_1 * k1_1 * x1 + y1 * k2_1 + k1_1 * k2_1 * x2 - k1_1 * y2) / (k2_1 - k1_1)

       r = math.sqrt((x_1 - pos_x[i]) ** 2 + (y_1 - pos_y[i]) ** 2)

       #distinguish between right and left
       v1_x = pos_y[i] - pos_y[nub]
       v1_y = pos_x[nub] - pos_x[i]  # Normal Vektor von anfangliche Geschwindigkeit
       v2_x = pos_x[nub_1] - pos_x[i]
       v2_y = pos_y[nub_1] - pos_y[i]
       if (v1_x * v2_x + v1_y * v2_y)  > 0:
           vorzeichen = -1
       else:
           vorzeichen = 1

       # use the akman model to compute the steering angle
       tan_delta = math.sqrt(l * l / abs(r ** 2 - lr ** 2))
       delta = math.atan(tan_delta) * vorzeichen
       angle.append(delta)
    return angle,num,num_1



#computation of velocity
def vel_computation(pos_x, pos_y, dt,num,num_1):
    for i in range(num, num_1+1):
        dx1 = pos_x[i] - pos_x[i -1]
        dy1 = pos_y[i] - pos_y[i -1]
        d1 = math.sqrt(dx1 ** 2 + dy1 ** 2)
        v1 = d1 / dt

        dx2 = pos_x[i + 1] - pos_x[i]
        dy2 = pos_y[i + 1] - pos_y[i]
        d2 = math.sqrt(dx2 ** 2 + dy2 ** 2)
        v2 = d2 / dt

        vel.append((v2 + v1) / 2)
    return vel

def generate_time(num,num_1):
    time_step = []
    for i in range(num,num_1+1):
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

      cd = load_data_files('allAmsterdamerRing.pickle',8)


      pos_x = np.array(cd['pos_x'])

      pos_y = np.array(cd['pos_y'])

      vel_t = np.array(cd['v_total'])

      angle = []
      vel = []
      dis = 2

      dt = sample_time(pos_x, pos_y, vel_t)


      angle,num,num_1 = geo2(pos_x,pos_y)

      vel = vel_computation(pos_x,pos_y,dt,num,num_1)

      time_step = generate_time(num,num_1)

      pre_x, pre_y= akman(vel, angle, tranum=8, num=num)
      print(pre_x)
      print(pre_y)

      figure_plot2(pre_x, pre_y)
