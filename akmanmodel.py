import pickle
import numpy as np
import math

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

def geo2(pos_x,pos_y,dis,l,lr):
    angle =[]

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

    for i in range(num,num_1):

       nub = i
       diss = 0
       while (diss < dis):
           nub = nub - 1
           diss = math.sqrt((pos_x[nub] - pos_x[i]) ** 2 + (pos_y[nub] - pos_y[i]) ** 2)

       nub_1 = i
       diss_1 = 0
       while (diss_1 < dis):
           if nub_1 == (len(pos_x)-1):
               nub_1 = len(pos_x)-1
               diss_1 = 2.1
           else:
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


def vel_computation(pos_x, pos_y, dt,num,num_1):
    vel = []
    for i in range(num, num_1):
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


def target_data(file_path):
    lr = 1.25
    l = 2 * lr
    with open(file_path,'rb') as fp:
        data = pickle.load(fp)

    target_index = []
    target_data = []

    for i in range(len(data)):
        info = []

        curve_data = data[i]
        cd = curve_data[1]

        pos_x = np.array(cd['pos_x'])

        pos_y = np.array(cd['pos_y'])

        vel_t = np.array(cd['v_total'])

        dis = 2

        dt = sample_time(pos_x, pos_y, vel_t)

        angle, num, num_1 = geo2(pos_x, pos_y,dis,l,lr)

        vel = vel_computation(pos_x, pos_y, dt, num, num_1)

        if len(vel)>150:

           target_index.append(i)
           for i in range(len(vel)):
               if i < 150:
                  info.append([angle[i],vel[i]])
           target_data.append(info)


    return target_data,target_index





if __name__ == '__main__':
    target_data,target_index = target_data('allAmsterdamerRingV2.pickle')

    print(len(target_index))
    print(np.shape(target_data))





