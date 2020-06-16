from findlane import *
import pickle
import numpy as np
import math

#get the target data,steering angle and velocity at each time step,geo2 is for steering angle
def ang_computation(pos_x,pos_y):
    #geometry information
    dis = 2
    #To compute the curvature of each curve, at least three points should be acquired, the other two points ought to have some distance to this point
    l = 2.5
    lr = 1.25
    angle =[]

    #num,num_1 are start and end points for each curve
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
    num_1 = len(pos_x) + num_1 + 1

    #computation of steering angle
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
               nub_1 = len(pos_x) - 1
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
       #v1_x,v1_y is the normal vector of initial velocity
       #v2_x,v2_y is the vector of second velocity
       v1_x = pos_y[i] - pos_y[nub]
       v1_y = pos_x[nub] - pos_x[i]
       v2_x = pos_x[nub_1] - pos_x[i]
       v2_y = pos_y[nub_1] - pos_y[i]
       #The steering angle of left-turn and right-turn ought to be different
       # if the angle
       if (v1_x * v2_x + v1_y * v2_y)  > 0:
           #if the angle between two vector is less than 90 degrees, it's right turn
           vorzeichen = -1
       else:
           #otherwise it's left turn
           vorzeichen = 1

       # use the akman model to compute the steering angle
       tan_delta = math.sqrt(l * l / abs(r ** 2 - lr ** 2))
       delta = math.atan(tan_delta) * vorzeichen
       angle.append(delta)

    return angle,num,num_1


#target-data, velocity and acceleration computation
def vel_computation(pos_x, pos_y,nu,nu_1):

    dt = 0.04
    acc = []
    vel = []

    #the computation of velocity
    for i in range(nu, nu_1):
        dx1 = pos_x[i] - pos_x[i -1]
        dy1 = pos_y[i] - pos_y[i -1]
        d1 = math.sqrt(dx1 ** 2 + dy1 ** 2)
        v1 = d1 / dt

        dx2 = pos_x[i + 1] - pos_x[i]
        dy2 = pos_y[i + 1] - pos_y[i]
        d2 = math.sqrt(dx2 ** 2 + dy2 ** 2)
        v2 = d2 / dt

        vel.append((v2 + v1) / 2)
        acc.append((v2 - v1) / dt)

    return vel,acc


#  This function is one filter that improves the quality of the data, remove some noise
def pool_angle(angle,vel,acc):
    for i in range(len(angle)):
        if i == 1:
            angle[i] = (angle[i] + angle[i+1] + angle[i+2]) / 3
            vel[i] = (vel[i] + vel[i+1] + vel[i+2]) / 3
            acc[i] = (acc[i] + acc[i+1] + acc[i+2]) / 3
        elif i == (len(angle)-1):
            angle[i] = (angle[i] + angle[i-1] + angle[i-2]) / 3
            vel[i] = (vel[i] + vel[i-1] + vel[i-2]) / 3
            acc[i] = (acc[i] + acc[i-1] + acc[i-2]) / 3
        else:
            angle[i] = (angle[i-1]+angle[i]+angle[i+1])/3
            vel[i] = (vel[i-1]+vel[i]+vel[i+1])/3
            acc[i] = (acc[i-1]+acc[i]+acc[i+1])/3
    return angle,vel,acc


#to generate the input data,we need to get two angle information,
def akman(pos_xa,pos_ya,vel_xa,vel_ya,vel,angle,num):

    lr = 1.25
    l = 2.5
    dt = 0.04
    dista = 0.75

    x_init = pos_xa[num]
    y_init = pos_ya[num]
    vel_xinit = vel_xa[num]
    vel_yinit = vel_ya[num]

    num_init = num
    dist = 0
    while (dist < dista):
        num_init += 1
        dist = math.sqrt((pos_xa[num_init] - pos_xa[num]) ** 2 + (pos_ya[num_init] - pos_ya[num]) ** 2)

    #beta is the angle between the direction of velocity and the direction of vehicle, pfai is the angle of vehicle
    x_zeichen = pos_xa[num_init] - pos_xa[num]
    y_zeichen = pos_ya[num_init] - pos_ya[num]

    #Computation of initial value of psi, which is the heading of vehicle
    psi_init = np.arctan2(y_zeichen, x_zeichen)

    #Computation of initial value of beta,which is the angle between velocity and heaidng of vehicle
    if vel_xinit != 0:
        beta_init = np.arctan2(vel_yinit, vel_xinit) - psi_init
    else:
        beta_init = 0

    beta = []
    beta.append(beta_init)
    pre_x = []
    pre_x.append(x_init)
    pre_y = []
    pre_y.append(y_init)
    psi = []
    psi.append(psi_init)

    for i in range(len(vel)):

        pre_x.append(pre_x[i] + vel[i] * math.cos(psi[i] + beta[i]) * dt)
        pre_y.append(pre_y[i] + vel[i] * math.sin(psi[i] + beta[i]) * dt)

        beta.append(math.atan(lr * math.tan(angle[i]) / l))
        psi.append(psi[i] + vel[i] * math.sin(beta[i]) * dt / lr)

    return psi,beta,pre_x,pre_y


def ang_diff(x,y):
    a = x - y
    a = (a + np.pi) % (2 * np.pi) - np.pi
    return a

def rela_change(x_co,y_co,num):

    rela_x = []
    rela_y = []

    for i in range(num,len(x_co)):

        if i > 5:
            rela_x.append(x_co[i] - x_co[i-5])
            rela_y.append(y_co[i] - y_co[i-5])
        else:
            rela_x.append(x_co[i] - x_co[0])
            rela_y.append(y_co[i] - y_co[0])

    return rela_x,rela_y


def sample_data(rand,ang,vel,acc,target_data,
                pos_x,pos_y,num,psi,beta,
                kru,ori,rul,wid,
                kru5,ori5,rul5,fai5,
                kru10,ori10,rul10,fai10,
                kru15,ori15,rul15,fai15,
                relx,rely,lis1,lis2,lis3,input_data,map_data):

    info = []
    info2 = []
    info3 = []

    startpoint = num + rand - 300
    endpoint = num + rand

    for t in range(startpoint, endpoint):
        m = t - startpoint
        if m % 10 == 0:
            # kinematic features
            pfai_norm = psi[t - num] / math.pi
            beta_norm = beta[t - num]
            vel_norm = vel[t - num]
            # map features
            wid_norm = wid
            kru_norm = kru[t] / 0.1
            ori_norm = ori[t]
            rul_norm = rul[t]

            kru_norm_1 = kru5[t] / 0.1
            ori_norm_1 = ori5[t] / math.pi
            rul_norm_1 = rul5[t]
            fai_norm_1 = fai5[t] / math.pi

            kru_norm_3 = kru10[t] / 0.1
            ori_norm_3 = ori10[t] / math.pi
            rul_norm_3 = rul10[t]
            fai_norm_3 = fai10[t] / math.pi

            kru_norm_5 = kru15[t] / 0.1
            ori_norm_5 = ori15[t] / math.pi
            rul_norm_5 = rul15[t]
            fai_norm_5 = fai15[t] / math.pi

            # gamma is the relative angle to the direction of centerline
            gamma = ang_diff(psi[t - num],ori[t])
            gamma_1 =ang_diff(psi[t - num],ori5[t])
            gamma_3 = ang_diff(psi[t - num],ori10[t])
            gamma_5 = ang_diff(psi[t - num],ori15[t])

            info.append([ang[t - num]*10, acc[t - num]])

            info2.append([relx[t - num], rely[t - num], pfai_norm, beta_norm,
                          vel_norm, kru_norm, rul_norm, ori_norm,
                          kru_norm_1, rul_norm_1, ori_norm_1, fai_norm_1,
                          kru_norm_3, rul_norm_3, ori_norm_3, fai_norm_3,
                          kru_norm_5, rul_norm_5, ori_norm_5, fai_norm_5,
                          ])

            info3.append([lis1[t], lis2[t], lis3[t], pos_x[t], pos_y[t]])

    target_data.append(info)
    input_data.append(info2)
    map_data.append(info3)

    return target_data, input_data, map_data


def classify_data(range,kind,ang, vel, acc, pos_x, pos_y, num, psi, beta,
                 kru, ori, rul, wid, kru5, ori5, rul5, fai5,
                 kru10, ori10, rul10, fai10,kru15, ori15, rul15, fai15,
                 relx, rely, lis1, lis2, lis3, target_data_l,input_data_l,
                 map_data_l,target_data_r,input_data_r,map_data_r,target_data_s,input_data_s,map_data_s):

    if len(vel) > range:

        if kind == 1:
            target_data_l, input_data_l, map_data_l = sample_data(range, ang, vel, acc, target_data_l,
                                                                  pos_x, pos_y, num, psi, beta,
                                                                  kru, ori, rul, wid,
                                                                  kru5, ori5, rul5, fai5,
                                                                  kru10, ori10, rul10, fai10,
                                                                  kru15, ori15, rul15, fai15,
                                                                  relx, rely, lis1, lis2, lis3, input_data_l,
                                                                  map_data_l)
        if kind == -1:
            target_data_r, input_data_r, map_data_r = sample_data(range, ang, vel, acc, target_data_r,
                                                                  pos_x, pos_y, num, psi, beta,
                                                                  kru, ori, rul, wid,
                                                                  kru5, ori5, rul5, fai5,
                                                                  kru10, ori10, rul10, fai10,
                                                                  kru15, ori15, rul15, fai15,
                                                                  relx, rely, lis1, lis2, lis3, input_data_r,
                                                                  map_data_r)
        if kind == 0:
            target_data_s, input_data_s, map_data_s = sample_data(range, ang, vel, acc, target_data_s,
                                                                  pos_x, pos_y, num, psi, beta,
                                                                  kru, ori, rul, wid,
                                                                  kru5, ori5, rul5, fai5,
                                                                  kru10, ori10, rul10, fai10,
                                                                  kru15, ori15, rul15, fai15,
                                                                  relx, rely, lis1, lis2, lis3, input_data_s,
                                                                  map_data_s)


    return target_data_l,input_data_l,map_data_l,target_data_r,input_data_r,map_data_r,target_data_s,input_data_s,map_data_s


#get the dataset of input and target
def target_input_get(fp1):

    with open(fp1, 'rb') as fp:
        data = pickle.load(fp)

    error_list = [5,10,16,21,24,26,40,43,55,58,67,68,77,78,79,84,89,91,99,106,108,134,141,142,148,159,160,163,
                  164,165,176,179,180,181,182,202,220,222,227,230,244,245,249,251,253,268,269,285,287,289,
                  290,292,299,301,302,304,314,332,339,340,352,353,357,363,381,386,388,407,411,414,421,423,
                  432,434,435,444,445,453.460,461,463,464,465,470,473,478,480,484,500,504,512,515,525,527,
                  528,530,532,535,554,555,557,580,599,602,612,623,624,625,629,642,643,648,650,655,660,663,
                  687,689,690,696,698,699,700,705,709,714,720,721,729,730,731,732,734,735,736,737,738,739,
                  740,741,742,743,744,745,746,748,749,750,751,752,753,755,758,766,773,782,784,785,788,789,
                  791,806,811,812,815,816,818,820,826,833,834,854,858,859,870,871,877,883,887,893,897,899,
                  904,914,916,921,922,925,926,930,932,936,944,949,958,967,969,970,971,978,984,992,993,994,
                  995,996,998,999,1000,1001,1002,1006,1007,1008,1009,1012,1022,1025,1048,1049,1051,1052,1059,
                  1062,1071,1081,1083,1093,1094,1096,1098,1101,1109,1124,1129,1130,1134,1137,1139,1140,1141,1142,
                  1145,1151,1154,1155,1162,1166,1167,1182,1184,1198,1200,1203,1204,1210,1211,1212,1213,1214,1215,1216,
                  1217,1218,1219,1220,1221,1222,1223,1225,1232,1233,1242,1255,1270,1274,1276,1284,1293,1303,
                  1309,1312,1313,1315,1319,1325,1329,1332,1340,1348,1349,1351,1356,1366,1368,1379,1380,1381,
                  1382,1383,1395,1397,1398,1399,1408,1410,1426,1427,1428,1430,1441,1447,1448,1449,1450,
                  1463,1473,1475,1481,1488,1510,1511,1512,1519,1520,1522,1529,
                  46,900,1200]

    target_data_l = []
    input_data_l = []
    map_data_l = []
    target_data_r = []
    input_data_r = []
    map_data_r = []
    target_data_s = []
    input_data_s = []
    map_data_s = []

    #the number in "range" determined how many curves do we need
    for i in range(len(data)):

        if i in error_list:
            continue

        pos_x = []
        pos_y = []
        vel_xa = []
        vel_ya = []

        #firstly we reas the data we need
        curve_data = data[i]
        cd = curve_data[1]
        x = np.array(cd['pos_x'])
        y = np.array(cd['pos_y'])
        xa = np.array(cd['vel_x'])
        ya = np.array(cd['vel_y'])


        #The first few points of data often contains lots of noise
        for k in range(20,len(x)):
            pos_x.append(x[k])
            pos_y.append(y[k])
            vel_xa.append(xa[k])
            vel_ya.append(ya[k])

        #compute the steering angle,choose the start point and the end point
        ang, num, num_1 = ang_computation(pos_x,pos_y)
        #compute the velocity and acceleration
        vel, acc = vel_computation(pos_x, pos_y, num, num_1)

        #distinguish the maneuver of trajectory
        v1_x = pos_y[num] - pos_y[0]
        v1_y = pos_x[0] - pos_x[num]
        v2_x = pos_x[-1] - pos_x[num_1]
        v2_y = pos_y[-1] - pos_y[num_1]
        Tmp = (v1_x * v2_x + v1_y * v2_y) / math.sqrt((v1_x ** 2 + v1_y ** 2) * (v2_x ** 2 + v2_y ** 2))

        if Tmp > 0.45:
            # right-turn curve
            kind = -1
        elif Tmp < -0.4:
            # left-turn curve
            kind = 1
        else:
            kind = 0

        #implement an filter to remove some noise in data
        ang, vel, acc = pool_angle(ang, vel, acc)
        #compute the direction and the angle between direction and velocity
        psi, beta, pre_x, pre_y = akman(pos_x, pos_y, vel_xa, vel_ya, vel, ang, num)
        #compute the relative change in x and y coordinates
        relx, rely = rela_change(pos_x,pos_y,num)

        #get the traffic environment from lanelet
        lis0, lis1, lis2, lis3, \
        wid, kru, ori, rul, \
        kru5, ori5, rul5, fai5, \
        kru10, ori10, rul10, fai10, \
        kru15, ori15, rul15, fai15 = findlane_index(pos_x,pos_y,kind,num)

        cons_list = [300,600,900,1200,1500]

        for konst in cons_list:

            target_data_l, input_data_l, map_data_l, target_data_r, input_data_r, map_data_r, target_data_s, input_data_s, map_data_s = classify_data(
                konst, kind, ang, vel, acc, pos_x, pos_y, num, psi, beta,
                kru, ori, rul, wid,
                kru5, ori5, rul5, fai5,
                kru10, ori10, rul10, fai10,
                kru15, ori15, rul15, fai15,
                relx, rely, lis1, lis2, lis3, target_data_l, input_data_l,
                map_data_l, target_data_r, input_data_r, map_data_r, target_data_s, input_data_s, map_data_s)

        random_lsit = [350, 400, 450, 500, 550, 650, 700, 750, 800, 850, 950, 1000, 1050, 1100, 1150, 1250, 1300, 1350, 1400, 1450]

        if kind == -1 or kind == 1:

            for ran in random_lsit:
                target_data_l, input_data_l, map_data_l, target_data_r, input_data_r, map_data_r, target_data_s, input_data_s, map_data_s = classify_data(
                                                                ran, kind, ang, vel, acc, pos_x, pos_y, num, psi, beta,
                                                                kru, ori, rul, wid,
                                                                kru5, ori5, rul5, fai5,
                                                                kru10, ori10, rul10, fai10,
                                                                kru15, ori15, rul15, fai15,
                                                                relx, rely, lis1, lis2, lis3, target_data_l, input_data_l,
                                                                map_data_l, target_data_r, input_data_r, map_data_r, target_data_s, input_data_s, map_data_s)

    return target_data_l, input_data_l, map_data_l, target_data_r, input_data_r, map_data_r, target_data_s, input_data_s, map_data_s


def train_valid_get():

    target_data_l, input_data_l, map_data_l, target_data_r, input_data_r, map_data_r, target_data_s, input_data_s, map_data_s = target_input_get('allAmsterdamerRing.pickle')

    right = [9, 10, 11, 12, 16, 32, 33, 34, 36, 37, 45, 46, 47, 48, 49, 50, 54, 55, 56, 73, 74, 76, 77, 78, 79, 80, 81,
             82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 103,
             110, 111, 112, 113, 114, 115, 118, 119, 120, 121, 122, 123, 124, 126, 130, 131, 132, 133, 134, 135, 137,
             138, 139, 140, 141, 142, 143, 146, 148, 149, 150, 151, 152, 163, 164, 165, 166, 167, 171, 181, 182,
             185, 193, 194, 195, 196, 197, 198, 206, 207, 208, 209, 210, 211, 212, 215, 217, 218, 219, 221, 222, 223,
             224, 225, 226, 227, 228, 229, 230, 231, 232, 235, 251, 252, 266, 267, 268]

    left = [1, 2, 3, 4, 5, 6, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 32, 33, 34, 35, 52, 58, 59, 60, 61, 62, 63, 64,
            65, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78, 79, 80, 81, 83, 84, 85, 86, 106, 114, 115,
            134, 136, 141, 142, 143, 144, 145, 154, 155, 156, 157, 160, 161, 162, 163, 164, 165, 166, 167, 168, 171,
            173, 174, 175, 176, 177, 181, 189, 190, 191, 193, 194, 195, 196, 197, 198,
            202, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 233, 234, 235, 236, 237, 238, 240,
            241, 242, 243, 244, 245, 246, 252, 253, 254, 271, 273, 274, 275, 276, 278, 279,280]

    straight = [0, 3, 4, 5, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 36, 37,
                38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 51, 54, 55, 56, 58, 61, 62, 63, 64, 65, 66,
                67, 68, 69, 70, 71, 72, 73, 74, 75, 79, 81, 84, 86, 89, 90, 91, 93, 95, 96, 97, 99, 101, 102, 104, 105,
                106, 108, 110, 111, 112, 114, 115, 116, 117, 118, 119, 120, 122, 123, 124, 125,
                126, 127, 128, 129, 130, 131, 132, 135, 136, 138, 143, 145, 146, 147, 149, 150, 151, 152, 154, 155, 158,
                159, 161, 162, 163, 164, 167, 168, 169, 170, 171, 172, 173, 174, 176, 177,
                178, 180, 181, 182, 185, 187, 191, 192, 193, 194, 195, 196, 197, 198, 198, 199, 200, 201, 202, 203, 208,
                209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 223, 225,
                225, 226, 228, 231, 232, 233, 235, 236, 237, 238, 240, 241, 242, 244, 245, 246, 248, 249, 256, 263, 266,
                268, 269, 270, 273, 274, 275, 276, 277, 279, 283, 285, 288, 289, 292, 293,
                295]

    input_data = []
    target_data = []
    map_data = []


    for i in range(295):

        if i in right:

            input_data.append(input_data_r[i])
            target_data.append(target_data_r[i])
            map_data.append(map_data_r[i])

        if i in left:

            input_data.append(input_data_l[i])
            target_data.append(target_data_l[i])
            map_data.append(map_data_l[i])

        if i in straight:

            input_data.append(input_data_s[i])
            target_data.append(target_data_s[i])
            map_data.append(map_data_s[i])

    test_split = 16
    valid_split = int((len(input_data)-16) * 0.2) + 16

    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    x_test = []
    y_test = []
    m_train = []
    m_valid = []
    m_test = []

    for i in range(test_split):
        x_test.append(input_data[i])
        y_test.append(target_data[i])
        m_test.append(map_data[i])

    for i in range(test_split, valid_split):
        x_valid.append(input_data[i])
        y_valid.append(target_data[i])
        m_valid.append(map_data[i])

    for i in range(valid_split,len(input_data)):
        x_train.append(input_data[i])
        y_train.append(target_data[i])
        m_train.append(map_data[i])

    return x_train,y_train,x_valid,y_valid,x_test,y_test,m_train,m_valid,m_test


#This function is to check if the sampled data is correct
def check_data():

    target_data_l, input_data_l, map_data_l, target_data_r, input_data_r, map_data_r, target_data_s, input_data_s, map_data_s = target_input_get('allAmsterdamerRing.pickle')

    for i in range(len(target_data_s)):

        straight = [0, 3, 4, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 22, 24, 25, 26, 27, 28, 29, 30, 31, 36,
                    37, 38, 39, 41, 43, 44, 45, 46, 47, 48, 49, 51, 54, 55, 56, 58, 61, 62, 63, 64, 65, 66,
                    67, 68, 69, 70, 71, 72, 73, 74, 75, 79, 81, 84, 86, 89, 90, 91, 93, 95, 96, 97, 99, 101, 102, 104,
                    105, 106, 108, 110, 111, 112, 114, 115, 116, 117, 118, 119, 120, 122, 123, 124, 125,
                    126, 127, 128, 129, 130, 131, 132, 136, 138, 143, 145, 146, 147, 149, 150, 151, 152, 154, 155,
                    158, 159, 161, 162, 164, 167, 168, 170, 171, 172, 173, 174, 176, 177,
                    178, 180, 181, 182, 185, 187, 191, 192, 193, 194, 195, 196, 197, 198, 198, 199, 200, 201, 202, 203,
                    208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 223, 225,
                    225, 226, 228, 231, 232, 233, 235, 236, 237, 238, 240, 241, 242, 244, 245, 246, 248, 249, 256, 263,
                    266, 268, 269, 270, 273, 274, 275, 276, 277, 279, 283, 285, 288, 289, 292, 293,
                    295]
        if i not in straight:
            continue
        print(i)
        training_d = input_data_s[i]
        target_d = target_data_s[i]
        map_d = map_data_s[i]

        pox = []
        poy = []

        kru = []
        kru5 = []
        kru10 = []
        kru15 = []

        ori = []
        ori5 = []
        ori10 = []
        ori15 = []

        gam = []
        gam5 = []
        gam10 = []
        gam15 = []

        ang = []
        acc = []
        velo = []
        psi = []
        beta = []

        for t in range(len(training_d)):

            input_per_time = training_d[t]
            targte_per_time = target_d[t]
            map_per_time = map_d[t]

            ang.append(targte_per_time[0])
            acc.append(targte_per_time[1])

            pox.append(map_per_time[3])
            poy.append(map_per_time[4])

            kru.append(input_per_time[5] * 0.1)
            kru5.append(input_per_time[8] * 0.1)
            kru10.append(input_per_time[12] * 0.1)
            kru15.append(input_per_time[16] * 0.1)

            ori.append(input_per_time[7])
            ori5.append(input_per_time[10])
            ori10.append(input_per_time[14])
            ori15.append(input_per_time[18])

            psi.append(input_per_time[2])
            beta.append(input_per_time[3])
            velo.append(input_per_time[4])

        plt.scatter(pox, poy)
        plt.show()

        plt.title('acceleration and velocity')
        plt.plot(acc)
        plt.plot(velo)
        plt.show()

        plt.title('curvature at current state, 5, 10, 15 meter after now')
        plt.plot(kru)
        plt.plot(kru5)
        plt.plot(kru10)
        plt.plot(kru15)
        plt.show()

        plt.title('steering angle and beta angle')
        plt.plot(beta)
        plt.plot(ang)
        plt.show()

        plt.title('orientation and psi angle')
        plt.plot(ori)
        plt.plot(ori5)
        plt.plot(ori10)
        plt.plot(ori15)
        plt.plot(psi)
        plt.show()

x_train,y_train,x_valid,y_valid,x_test,y_test,m_train,m_valid,m_test = train_valid_get()
print(np.shape(x_train))
print(np.shape(x_valid))
print(np.shape(x_test))
f = open("x_t.pickle",'wb+')
pickle.dump(x_train,f)
pickle.dump(y_train,f)
pickle.dump(m_train,f)
f1 = open("x_v.pickle",'wb+')
pickle.dump(x_valid,f1)
pickle.dump(y_valid,f1)
pickle.dump(m_valid,f1)
f2 = open("x_e.pickle",'wb+')
pickle.dump(x_test,f2)
pickle.dump(y_test,f2)
pickle.dump(m_test,f2)

