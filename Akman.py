import pickle
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy.linalg as LA

#get the data I need,position on x-y koordinate,and the total velocity at each time step
def load_data_files(filepath,tra_num):
    with open(filepath,'rb') as fp:
        data = pickle.load(fp)
    curve_data = np.array(data[tra_num])
    cd = np.array(curve_data[1])
    return cd

#compute the sampling time
def sample_time(pos_x,pos_y,vel_t):
    sum = 0
    for i in range(len(pos_x)-1):

        dis_x = pos_x[i+1]-pos_x[i]
        dis_y = pos_y[i+1]-pos_y[i]
        dis_t = math.sqrt(dis_x*dis_x+dis_y*dis_y)
        vel = vel_t[i]
        sum = sum+dis_t/vel
    dt = sum/(len(pos_x)-1)
    return dt

#Use the akman model(bicycle model),generate the trajectory with the information of acceleration and steering angle
def akman(acc,angle,beta0,x0,y0,pfai0,vel0):
    beta = []
    beta.append(beta0)
    pre_x = []
    pre_x.append(x0)
    pre_y = []
    pre_y.append(y0)
    pfai = []
    pfai.append(pfai0)
    vel = []
    vel.append(vel0)
    for i in range(len(acc)):
        beta.append(math.atan(lr * math.tan(angle[i]) / (lf + lr)))
        pre_x.append(pre_x[i] + vel[i] * math.cos(pfai[i] + beta[i]) * dt)
        pre_y.append(pos_y[i] + vel[i] * math.sin(pfai[i] + beta[i]) * dt)
        pfai.append(pfai[i] + vel[i] * math.sin(beta[i]) * dt / lr)
        vel.append(vel[i] + acc[i] * dt)
    return beta,pre_x,pre_y,pfai,vel


# The following is used to get the information of steering angle and acceleration from the position data of trajectory


#calculate the steering angle with the position-data on x-y koordinate
def cal1(pos_x,pos_y,angle):
     for i in range(1,len(pos_x)-1):
        #compute the radius of curvature first
        dx = pos_x[i+1]-pos_x[i-1]
        dy = pos_y[i+1]-pos_y[i-1]
        d1 = dy/dx

        dx1 = pos_x[i] - pos_x[i - 1]
        dy1 = pos_y[i] - pos_y[i - 1]

        dx2 = pos_x[i + 1] - pos_x[i]
        dy2 = pos_y[i + 1] - pos_y[i]

        d2 = (dy2 / dx2 - dy1 / dx1) * 2 / (dx1 + dx2)
        k = d2/math.pow(1+d1*d1,1.5)
        r = 1 / k
        # use the akman model to compute the steering angle
        tan_delta = math.sqrt(l * l / abs(r * r - lr * lr))
        delta = math.atan(tan_delta)
        angle.append(delta*180/math.pi)
     return angle


def geo1(pos_x,pos_y,angle):
     for i in range(1,len(pos_x)-1):

        #compute the radius of curvature first
        k1 = (pos_y[i] - pos_y[i-1])/(pos_x[i] - pos_x[i-1])
        k1_1 = -1/k1

        k2 = (pos_y[i+1] - pos_y[i])/(pos_x[i+1] - pos_x[i])
        k2_1 = -1/k2

        x_1 = (k1_1*pos_x[i-1] - k2_1*pos_x[i] - pos_y[i-1] + pos_y[i])/(k1_1 - k2_1)
        y_1 = (-k2_1*k1_1*pos_x[i-1] + pos_y[i-1]*k2_1 + k1_1*k2_1*pos_x[i] - k1_1*pos_y[i])/(k2_1 - k1_1)

        r1 = math.sqrt((x_1 - pos_x[i-1])**2+(y_1 - pos_y[i-1])**2)
        r2 = math.sqrt((x_1 - pos_x[i])**2+(y_1 - pos_y[i])**2)
        r = (r1+r2)/2
        # use the akman model to compute the steering angle
        tan_delta = math.sqrt(l * l / abs(r **2 - lr **2))
        delta = math.atan(tan_delta)

        angle.append(delta*180/math.pi)
     return angle



def cal2(pos_x,pos_y,angle):
     for i in range(1, len(pos_x) - 1):
         # compute the radius of curvature first
         x = [pos_x[i - 1], pos_x[i], pos_x[i + 1]]
         y = [pos_y[i - 1], pos_y[i], pos_y[i + 1]]

         t_a = LA.norm([x[1] - x[0], y[1] - y[0]])
         t_b = LA.norm([x[2] - x[1], y[2] - y[1]])

         M = np.array([
            [1, -t_a, t_a ** 2],
            [1, 0, 0],
            [1, t_b, t_b ** 2]
         ])

         a = np.matmul(LA.inv(M), x)
         b = np.matmul(LA.inv(M), y)

         kappa = 2 * (a[2] * b[1] - b[2] * a[1]) / (a[1] ** 2. + b[1] ** 2.) ** (1.5)
         r = 1 / kappa
         # use the akman model to compute the steering angle
         tan_delta = math.sqrt(l ** 2 / abs(r ** 2 - lr ** 2))
         delta = math.atan(tan_delta)
         angle.append(delta * 180 / math.pi)
     return angle


def geo2(pos_x,pos_y,angle):
    for i in range(1, len(pos_x) - 1):
        # compute the radius of curvature first
        k1 = (pos_y[i] - pos_y[i - 1]) / (pos_x[i] - pos_x[i - 1])
        k1_1 = -1 / k1
        x1 = (pos_x[i] + pos_x[i - 1]) / 2
        y1 = (pos_y[i] + pos_y[i - 1]) / 2

        k2 = (pos_y[i + 1] - pos_y[i]) / (pos_x[i + 1] - pos_x[i])
        k2_1 = -1 / k2
        x2 = (pos_x[i + 1] + pos_x[i]) / 2
        y2 = (pos_y[i + 1] + pos_y[i]) / 2

        x_1 = (k1_1 * x1 - k2_1 * x2 - y1 + y2) / (k1_1 - k2_1)
        y_1 = (-k2_1 * k1_1 * x1 + y1 * k2_1 + k1_1 * k2_1 * x2 - k1_1 * y2) / (k2_1 - k1_1)

        r = math.sqrt((x_1 - x1) ** 2 + (y_1 - y1) ** 2)

        # use the akman model to compute the steering angle
        tan_delta = math.sqrt(l * l / abs(r ** 2 - lr ** 2))
        delta = math.atan(tan_delta)

        angle.append(delta * 180 / math.pi)
    return angle


#calculate the acceleration
def acc_computation(pos_x,pos_y,dt,acc):
     for i in range(1,len(pos_x)-1):
        dx1 = pos_x[i]-pos_x[i-1]
        dy1 = pos_y[i]-pos_y[i-1]
        d1 = math.sqrt(dx1*dx1+dy1*dy1)
        v1 = d1/dt

        dx2 = pos_x[i+1]-pos_x[i]
        dy2 = pos_y[i+1]-pos_y[i-1]
        d2 = math.sqrt(dx2*dx2+dy2*dy2)
        v2 = d2/dt

        acc.append((v2-v1)/dt)
     return acc

#generate the timestep
def generate_time(pos_x):
    time_step = []
    for i in range(1, len(pos_x) - 1):
        time_step.append(i * dt)
    return time_step


def figure_plot(acc,angle,time_step):
      plt.figure()

      plt.subplot(2,1,1)
      plt.scatter(time_step,angle,s=1)
      plt.xlabel('time')
      plt.ylabel('steering_angle')

      plt.subplot(2,1,2)
      plt.scatter(time_step,acc,s=1)
      plt.xlabel('time')
      plt.ylabel('acceleration')

      plt.show()

lf = 1.5
lr = 1.5
l = lf+lr

cd = load_data_files('allAmsterdamerRing.pickle',2)

pos_x = cd[:,2]
pos_y = cd[:,3]
vel_t = cd[:,7]

angle = []
acc =[]

dt = sample_time(pos_x,pos_y,vel_t)

angle = geo1(pos_x,pos_y,angle)

acc = acc_computation(pos_x,pos_y,dt,acc)

time_step = generate_time(pos_x)

figure_plot(acc,angle,time_step)

