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


# The pooling operation improves the quality of the data, remove some noise
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


def rela_change(x_co,y_co,num):

    rela_x = []
    rela_y = []
    for i in range(num,len(x_co)):

        if i>5:
            rela_x.append(x_co[i]-x_co[i-5])
            rela_y.append(y_co[i]-y_co[i-5])
        else:
            rela_x.append(x_co[i] - x_co[0])
            rela_y.append(y_co[i] - y_co[0])

    return rela_x,rela_y


#get the dataset of input and target
def target_input_get(fp1):

    with open(fp1, 'rb') as fp:
        data = pickle.load(fp)

    error_list = [10,16,24,26,40,43,55,58,67,68,77,78,79,84,89,91,99,106,108,134,141,142,148,159,160,163,
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
                  1463,1473,1475,1481,1488,1510,1511,1512,1519,1520,1522,1529]
    target_data = []
    input_data = []
    map_data = []

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
        #implement an filter to remove some noise in data
        ang, vel, acc = pool_angle(ang, vel, acc)
        #compute the direction and the angle between direction and velocity
        psi, beta, pre_x, pre_y = akman(pos_x, pos_y, vel_xa, vel_ya, vel, ang, num)
        #compute the relative change in x and y coordinates
        relx, rely = rela_change(pos_x,pos_y,num)
        #get the traffic environment from lanelet
        lis1,lis2,wid, kru, ori, rul, kru1, ori1, rul1, fai1, kru3, ori3, rul3, fai3, kru5, ori5, rul5, fai5 = findlane_index(pos_x,pos_y)

        if len(vel)>300:
            info = []
            info2 = []
            info3 = []

            #get traget data for every 0.2 second
            for m in range(300):
                if m % 5 ==0:
                   info.append([ang[m]*10,acc[m]/10])
            target_data.append(info)

            x_init = pre_x[0]
            y_init = pre_y[0]

            for t in range(num,num+300):
                m = t - num
                if m % 5 ==0:
                    #kinematic features
                    x_norm = (pos_x[t] - x_init)/10
                    y_norm = (pos_y[t] - y_init)/10
                    pfai_norm = psi[t - num] / math.pi
                    beta_norm = beta[t - num]
                    vel_norm = vel[t - num]
                    #map features
                    wid_norm = wid
                    kru_norm = kru[t] / 0.1
                    ori_norm = ori[t]
                    rul_norm = rul[t]
                    kru_norm_1 = kru1[t] / 0.1
                    ori_norm_1 = ori1[t] / math.pi
                    rul_norm_1 = rul1[t]
                    fai_norm_1 = fai1[t] / math.pi
                    kru_norm_3 = kru3[t] / 0.1
                    ori_norm_3 = ori3[t] / math.pi
                    rul_norm_3 = rul3[t]
                    fai_norm_3 = fai3[t] / math.pi
                    kru_norm_5 = kru5[t] / 0.1
                    ori_norm_5 = ori5[t] / math.pi
                    rul_norm_5 = rul5[t]
                    fai_norm_5 = fai5[t] / math.pi

                    #gamma is the relative angle to the direction of centerline
                    gamma = psi[t - num] - ori[t]
                    if abs(gamma) > math.pi:
                        gamma = psi[t-num] - 2 * math.pi - ori[t]

                    gamma_1 = psi[t - num] - ori1[t]
                    if abs(gamma_1) > math.pi:
                        gamma_1 = psi[t - num] - 2 * math.pi - ori1[t]

                    gamma_3 = psi[t - num] - ori3[t]
                    if abs(gamma_3) > math.pi:
                        gamma_3 = psi[t - num] - 2 * math.pi - ori3[t]

                    gamma_5 = psi[t - num] - ori5[t]
                    if abs(gamma_5) > math.pi:
                        gamma_5 = psi[t - num] - 2 * math.pi - ori5[t]

                    info2.append([x_norm, y_norm, relx[t-num],rely[t-num],
                                  pfai_norm,beta_norm,vel_norm,
                                  kru_norm,rul_norm,ori_norm,
                                  kru_norm_1,rul_norm_1,ori_norm_1,fai_norm_1,
                                  kru_norm_3,rul_norm_3,ori_norm_3,fai_norm_3,
                                  kru_norm_5,rul_norm_5,ori_norm_5,fai_norm_5
                                  ])
                    info3.append([lis1[t],lis2[t]])

            input_data.append(info2)
            map_data.append(info3)

        if len(vel) > 600:

            info = []
            info2 = []
            info3 = []

            for m in range(300):
                if m % 5 == 0:
                    info.append([ang[m+300] * 10, acc[m+300]/10])
            target_data.append(info)

            x_init = pre_x[0]
            y_init = pre_y[0]

            for t in range(num+300, num+600):
                m = t - num -300
                if m % 5 == 0:
                    # kinematic features
                    x_norm = (pos_x[t] - x_init) / 10
                    y_norm = (pos_y[t] - y_init) / 10
                    pfai_norm = psi[t - num] / math.pi
                    beta_norm = beta[t - num]
                    vel_norm = vel[t - num]
                    # map features
                    wid_norm = wid
                    kru_norm = kru[t] / 0.1
                    ori_norm = ori[t]
                    rul_norm = rul[t]
                    kru_norm_1 = kru1[t] / 0.1
                    ori_norm_1 = ori1[t] / math.pi
                    rul_norm_1 = rul1[t]
                    fai_norm_1 = fai1[t] / math.pi
                    kru_norm_3 = kru3[t] / 0.1
                    ori_norm_3 = ori3[t] / math.pi
                    rul_norm_3 = rul3[t]
                    fai_norm_3 = fai3[t] / math.pi
                    kru_norm_5 = kru5[t] / 0.1
                    ori_norm_5 = ori5[t] / math.pi
                    rul_norm_5 = rul5[t]
                    fai_norm_5 = fai5[t] / math.pi

                    # gamma is the relative angle to the direction of centerline
                    gamma = psi[t - num] - ori[t]
                    if abs(gamma) > math.pi:
                        gamma = psi[t - num] - 2 * math.pi - ori[t]

                    gamma_1 = psi[t - num] - ori1[t]
                    if abs(gamma_1) > math.pi:
                        gamma_1 = psi[t - num] - 2 * math.pi - ori1[t]

                    gamma_3 = psi[t - num] - ori3[t]
                    if abs(gamma_3) > math.pi:
                        gamma_3 = psi[t - num] - 2 * math.pi - ori3[t]

                    gamma_5 = psi[t - num] - ori5[t]
                    if abs(gamma_5) > math.pi:
                        gamma_5 = psi[t - num] - 2 * math.pi - ori5[t]

                    info2.append([x_norm, y_norm, relx[t - num], rely[t - num],
                                  pfai_norm, beta_norm, vel_norm,
                                  kru_norm,rul_norm, ori_norm,
                                  kru_norm_1,rul_norm_1,ori_norm_1,fai_norm_1,
                                  kru_norm_3,rul_norm_3,ori_norm_3,fai_norm_3,
                                  kru_norm_5,rul_norm_5,ori_norm_5,fai_norm_5
                                  ])
                    info3.append([lis1[t], lis2[t]])

            input_data.append(info2)
            map_data.append(info3)


        if len(vel) > 900:
            info = []
            info2 = []
            info3 = []

            for m in range(300):
                if m % 5 == 0:
                    info.append([ang[m+600] * 10, acc[m+600]/10])
            target_data.append(info)

            x_init = pre_x[0]
            y_init = pre_y[0]

            for t in range(num+600, num+900):
                m = t - num -600
                if m % 5 == 0:
                    # kinematic features
                    x_norm = (pos_x[t] - x_init) / 10
                    y_norm = (pos_y[t] - y_init) / 10
                    pfai_norm = psi[t - num] / math.pi
                    beta_norm = beta[t - num]
                    vel_norm = vel[t - num]
                    # map features
                    wid_norm = wid
                    kru_norm = kru[t] / 0.1
                    ori_norm = ori[t]
                    rul_norm = rul[t]
                    kru_norm_1 = kru1[t] / 0.1
                    ori_norm_1 = ori1[t] / math.pi
                    rul_norm_1 = rul1[t]
                    fai_norm_1 = fai1[t] / math.pi
                    kru_norm_3 = kru3[t] / 0.1
                    ori_norm_3 = ori3[t] / math.pi
                    rul_norm_3 = rul3[t]
                    fai_norm_3 = fai3[t] / math.pi
                    kru_norm_5 = kru5[t] / 0.1
                    ori_norm_5 = ori5[t] / math.pi
                    rul_norm_5 = rul5[t]
                    fai_norm_5 = fai5[t] / math.pi

                    # gamma is the relative angle to the direction of centerline
                    gamma = psi[t - num] - ori[t]
                    if abs(gamma) > math.pi:
                        gamma = psi[t - num] - 2 * math.pi - ori[t]

                    gamma_1 = psi[t - num] - ori1[t]
                    if abs(gamma_1) > math.pi:
                        gamma_1 = psi[t - num] - 2 * math.pi - ori1[t]

                    gamma_3 = psi[t - num] - ori3[t]
                    if abs(gamma_3) > math.pi:
                        gamma_3 = psi[t - num] - 2 * math.pi - ori3[t]

                    gamma_5 = psi[t - num] - ori5[t]
                    if abs(gamma_5) > math.pi:
                        gamma_5 = psi[t - num] - 2 * math.pi - ori5[t]

                    info2.append([x_norm, y_norm, relx[t - num], rely[t - num],
                                  pfai_norm, beta_norm, vel_norm,
                                  kru_norm,rul_norm,ori_norm,
                                  kru_norm_1,rul_norm_1,ori_norm_1,fai_norm_1,
                                  kru_norm_3,rul_norm_3,ori_norm_3,fai_norm_3,
                                  kru_norm_5,rul_norm_5,ori_norm_5,fai_norm_5
                                  ])
                    info3.append([lis1[t], lis2[t]])

            input_data.append(info2)
            map_data.append(info3)

        if len(vel) > 1200:
            info = []
            info2 = []
            info3 = []

            for m in range(300):
                if m % 5 == 0:
                    info.append([ang[m+900] * 10, acc[m+900]/10])
            target_data.append(info)

            x_init = pre_x[0]
            y_init = pre_y[0]

            for t in range(num+900, num+1200):
                m = t - num -900
                if m % 5 == 0:
                    # kinematic features
                    x_norm = (pos_x[t] - x_init) / 10
                    y_norm = (pos_y[t] - y_init) / 10
                    pfai_norm = psi[t - num] / math.pi
                    beta_norm = beta[t - num]
                    vel_norm = vel[t - num]
                    # map features
                    wid_norm = wid
                    kru_norm = kru[t] / 0.1
                    ori_norm = ori[t]
                    rul_norm = rul[t]
                    kru_norm_1 = kru1[t] / 0.1
                    ori_norm_1 = ori1[t] / math.pi
                    rul_norm_1 = rul1[t]
                    fai_norm_1 = fai1[t] / math.pi
                    kru_norm_3 = kru3[t] / 0.1
                    ori_norm_3 = ori3[t] / math.pi
                    rul_norm_3 = rul3[t]
                    fai_norm_3 = fai3[t] / math.pi
                    kru_norm_5 = kru5[t] / 0.1
                    ori_norm_5 = ori5[t] / math.pi
                    rul_norm_5 = rul5[t]
                    fai_norm_5 = fai5[t] / math.pi

                    # gamma is the relative angle to the direction of centerline
                    gamma = psi[t - num] - ori[t]
                    if abs(gamma) > math.pi:
                        gamma = psi[t - num] - 2 * math.pi - ori[t]

                    gamma_1 = psi[t - num] - ori1[t]
                    if abs(gamma_1) > math.pi:
                        gamma_1 = psi[t - num] - 2 * math.pi - ori1[t]

                    gamma_3 = psi[t - num] - ori3[t]
                    if abs(gamma_3) > math.pi:
                        gamma_3 = psi[t - num] - 2 * math.pi - ori3[t]

                    gamma_5 = psi[t - num] - ori5[t]
                    if abs(gamma_5) > math.pi:
                        gamma_5 = psi[t - num] - 2 * math.pi - ori5[t]

                    info2.append([x_norm, y_norm, relx[t - num], rely[t - num],
                                  pfai_norm, beta_norm, vel_norm,
                                  kru_norm,rul_norm,ori_norm,
                                  kru_norm_1,rul_norm_1,ori_norm_1,fai_norm_1,
                                  kru_norm_3,rul_norm_3,ori_norm_3,fai_norm_3,
                                  kru_norm_5,rul_norm_5,ori_norm_5,fai_norm_5
                                  ])
                    info3.append([lis1[t], lis2[t]])

            input_data.append(info2)
            map_data.append(info3)


        if len(vel) > 1500:
            info = []
            info2 = []
            info3 = []

            for m in range(300):
                if m % 5 == 0:
                    info.append([ang[m+1200] * 10, acc[m+1200]/10])
            target_data.append(info)

            x_init = pre_x[0]
            y_init = pre_y[0]

            for t in range(num+1200, num+1500):
                m = t - num - 1200
                if m % 5 == 0:
                    # kinematic features
                    x_norm = (pos_x[t] - x_init) / 10
                    y_norm = (pos_y[t] - y_init) / 10
                    pfai_norm = psi[t - num] / math.pi
                    beta_norm = beta[t - num]
                    vel_norm = vel[t - num]
                    # map features
                    wid_norm = wid
                    kru_norm = kru[t] / 0.1
                    ori_norm = ori[t]
                    rul_norm = rul[t]
                    kru_norm_1 = kru1[t] / 0.1
                    ori_norm_1 = ori1[t] / math.pi
                    rul_norm_1 = rul1[t]
                    fai_norm_1 = fai1[t] / math.pi
                    kru_norm_3 = kru3[t] / 0.1
                    ori_norm_3 = ori3[t] / math.pi
                    rul_norm_3 = rul3[t]
                    fai_norm_3 = fai3[t] / math.pi
                    kru_norm_5 = kru5[t] / 0.1
                    ori_norm_5 = ori5[t] / math.pi
                    rul_norm_5 = rul5[t]
                    fai_norm_5 = fai5[t] / math.pi

                    # gamma is the relative angle to the direction of centerline
                    gamma = psi[t - num] - ori[t]
                    if abs(gamma) > math.pi:
                        gamma = psi[t - num] - 2 * math.pi - ori[t]

                    gamma_1 = psi[t - num] - ori1[t]
                    if abs(gamma_1) > math.pi:
                        gamma_1 = psi[t - num] - 2 * math.pi - ori1[t]

                    gamma_3 = psi[t - num] - ori3[t]
                    if abs(gamma_3) > math.pi:
                        gamma_3 = psi[t - num] - 2 * math.pi - ori3[t]

                    gamma_5 = psi[t - num] - ori5[t]
                    if abs(gamma_5) > math.pi:
                        gamma_5 = psi[t - num] - 2 * math.pi - ori5[t]

                    info2.append([x_norm, y_norm, relx[t - num], rely[t - num],
                                  pfai_norm, beta_norm, vel_norm,
                                  kru_norm, rul_norm, ori_norm,
                                  kru_norm_1,rul_norm_1,ori_norm_1,fai_norm_1,
                                  kru_norm_3,rul_norm_3,ori_norm_3,fai_norm_3,
                                  kru_norm_5,rul_norm_5,ori_norm_5,fai_norm_5
                                  ])
                    info3.append([lis1[t], lis2[t]])

            input_data.append(info2)
            map_data.append(info3)

    return target_data,input_data,map_data


def train_valid_get():

    target_data, input_data, map_data = target_input_get('allAmsterdamerRing.pickle')

    #traing_set:validation_set = 4:1
    train_split = 416
    valid_split = 448
    test_split = 464

    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    x_test = []
    y_test = []
    m_train = []
    m_valid = []
    m_test = []

    for i in range(train_split):
        x_train.append(input_data[i])
        y_train.append(target_data[i])
        m_train.append(map_data[i])

    for i in range(train_split, valid_split):
        x_valid.append(input_data[i])
        y_valid.append(target_data[i])
        m_valid.append(map_data[i])

    for i in range(valid_split,test_split):
        x_test.append(input_data[i])
        y_test.append(target_data[i])
        m_test.append(map_data[i])

    return x_train,y_train,x_valid,y_valid,x_test,y_test,m_train,m_valid,m_test

