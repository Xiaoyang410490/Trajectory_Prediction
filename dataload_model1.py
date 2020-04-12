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

    num_index = []
    target_index = []
    target_data = []
    velocity = []
    winkel = []

    for i in range(20):
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
           velocity.append(vel)
           winkel.append(angle)
           target_index.append(i)
           num_index.append(num)
           for i in range(len(vel)):
               if i < 150:
                  info.append([angle[i],vel[i]])
           target_data.append(info)

    return target_data,target_index,num_index,velocity,winkel


def akman(file_path, vel, angle, tranum, num):


    lr = 1.25
    l = 2 * lr
    with open(file_path, 'rb') as fp:
        data = pickle.load(fp)
    cd = data[tranum]
    cd = cd[1]

    pos_xa = np.array(cd['pos_x'])
    x_init = pos_xa[num]

    pos_ya = np.array(cd['pos_y'])
    y_init = pos_ya[num]

    vel_ta = np.array(cd['v_total'])

    dt = sample_time(pos_xa,pos_ya,vel_ta)

    vel_xa = np.array(cd['vel_x'])
    vel_xinit = vel_xa[num]

    vel_ya = np.array(cd['vel_y'])
    vel_yinit = vel_ya[num]

    x_zeichen = pos_xa[num + 3] - pos_xa[num]
    y_zeichen = pos_ya[num + 3] - pos_ya[num]

    if x_zeichen > 0:
        pfai_init = math.atan(y_zeichen / x_zeichen)
    else:
        pfai_init = math.atan(y_zeichen / x_zeichen) + math.pi

    if vel_xinit != 0:
        beta_init = math.atan(vel_yinit / vel_xinit) - pfai_init
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

    return pfai,beta

def input_data(target_index,num_index,file_path,velocity,winkel):
    lr = 1.25
    l = 2 * lr
    with open(file_path, 'rb') as fp:
        data = pickle.load(fp)

    input_data = []

    for i in range(len(target_index)):
        info = []

        index = target_index[i]
        num = num_index[i]
        vel = velocity[i]
        angle = winkel[i]
        kurve_data = data[index]
        kd = kurve_data[1]

        pos_x = np.array(kd['pos_x'])
        pos_y = np.array(kd['pos_y'])
        pfai,beta = akman(file_path,vel,angle,index,num)

        for i in range(num,num+150):
            info.append([pos_x[i],pos_y[i],pfai[i-num],beta[i-num]])

        input_data.append(info)

    return input_data

def univariate_data(inputset, targetset,start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = 150 - target_size

    for i in range(start_index,end_index):
        input_seq = []
        target_seq = []
        for num in range(i-history_size,i):
            input_seq.append(inputset[num])
        for num in range(i,i+target_size):
            target_seq.append(targetset[num])

        data.append(input_seq)
        labels.append(target_seq)

    data = np.array(data)
    labels = np.array(labels)

    return data,labels














