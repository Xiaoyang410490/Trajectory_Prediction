from findlane import *
import pickle
import numpy as np
import math
from scipy.signal import sosfiltfilt,butter


def curve_filter(posx,posy):

    sos = butter(4, 0.125, output='sos')
    positionx = sosfiltfilt(sos, posx)
    positiony = sosfiltfilt(sos, posy)

    return positionx,positiony

def target_filter(ac,an):

    sos = butter(4, 0.125, output='sos')
    acce = sosfiltfilt(sos, ac)
    angl = sosfiltfilt(sos, an)

    return acce,angl

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
       tan_delta = math.sqrt( l * l / abs(r ** 2 - lr ** 2) )
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
    if abs(a) > 2.5:
        a = -np.sign(a) * (abs(a) - np.pi)
    return a


def rela_change(x_co,y_co,num):

    rela_x = []
    rela_y = []

    for i in range(num,len(x_co)):

        if i > 10:
            rela_x.append(x_co[i] - x_co[i-10])
            rela_y.append(y_co[i] - y_co[i-10])
        else:
            rela_x.append(x_co[i] - x_co[0])
            rela_y.append(y_co[i] - y_co[0])

    return rela_x,rela_y


def sample_data(rand,ang,vel,acc,target_data,
                pos_x,pos_y,psi,beta,
                kru,ori,rul,dic,
                kru5,ori5,rul5,
                kru10,ori10,rul10,
                kru15,ori15,rul15,
                relx,rely,lis1,lis2,lis3,input_data,map_data):

    info = []
    info2 = []
    info3 = []

    startpoint = rand - 250
    endpoint = rand

    angt = []
    acct = []

    for t in range(startpoint, endpoint):

        m = t - startpoint
        if m % 10 == 0:

            # kinematic features
            pfai_norm = psi[t] / math.pi
            vel_norm = vel[t]

            # map features
            kru_norm = kru[t] / 0.1
            ori_norm = ori[t] / math.pi
            rul_norm = rul[t]

            kru_norm_1 = kru5[t] / 0.1
            ori_norm_1 = ori5[t] / math.pi
            rul_norm_1 = rul5[t]

            kru_norm_3 = kru10[t] / 0.1
            ori_norm_3 = ori10[t] / math.pi
            rul_norm_3 = rul10[t]

            kru_norm_5 = kru15[t] / 0.1
            ori_norm_5 = ori15[t] / math.pi
            rul_norm_5 = rul15[t]

            # gamma is the relative angle to the direction of centerline
            gamma = ang_diff(psi[t],ori[t]) / math.pi
            gamma_1 = ang_diff(psi[t],ori5[t]) / math.pi
            gamma_3 = ang_diff(psi[t],ori10[t]) / math.pi
            gamma_5 = ang_diff(psi[t],ori15[t]) / math.pi

            dic_norm = dic[t] / 10

            angt.append(ang[t])
            acct.append(acc[t])

            info2.append([relx[t], rely[t], pfai_norm, dic_norm, vel_norm,
                          kru_norm, rul_norm, ori_norm, gamma,
                          kru_norm_1, rul_norm_1, ori_norm_1, gamma_1,
                          kru_norm_3, rul_norm_3, ori_norm_3, gamma_3,
                          kru_norm_5, rul_norm_5, ori_norm_5, gamma_5,
                          ])

            info3.append([lis1[t], lis2[t], lis3[t], pos_x[t], pos_y[t]])

    acct,angt = target_filter(acc,angt)

    for i in range(len(angt)):

        info.append([angt[i]*10,acct[i]])

    target_data.append(info)
    input_data.append(info2)
    map_data.append(info3)

    return target_data, input_data, map_data


def classify_data(range,kind,ang, vel, acc, pos_x, pos_y,psi, beta,
                 kru, ori, rul, dic, kru5, ori5, rul5,
                 kru10, ori10, rul10, kru15, ori15, rul15,
                 relx, rely, lis1, lis2, lis3, target_data_l,input_data_l,
                 map_data_l,target_data_r,input_data_r,map_data_r,target_data_s,input_data_s,map_data_s):

    if len(vel) > range:

        if kind == 1:
            target_data_l, input_data_l, map_data_l = sample_data(range, ang, vel, acc, target_data_l,
                                                                  pos_x, pos_y, psi, beta,
                                                                  kru, ori, rul, dic,
                                                                  kru5, ori5, rul5,
                                                                  kru10, ori10, rul10,
                                                                  kru15, ori15, rul15,
                                                                  relx, rely, lis1, lis2, lis3, input_data_l,
                                                                  map_data_l)
        if kind == -1:
            target_data_r, input_data_r, map_data_r = sample_data(range, ang, vel, acc, target_data_r,
                                                                  pos_x, pos_y, psi, beta,
                                                                  kru, ori, rul, dic,
                                                                  kru5, ori5, rul5,
                                                                  kru10, ori10, rul10,
                                                                  kru15, ori15, rul15,
                                                                  relx, rely, lis1, lis2, lis3, input_data_r,
                                                                  map_data_r)
        if kind == 0:
            target_data_s, input_data_s, map_data_s = sample_data(range, ang, vel, acc, target_data_s,
                                                                  pos_x, pos_y,  psi, beta,
                                                                  kru, ori, rul, dic,
                                                                  kru5, ori5, rul5,
                                                                  kru10, ori10, rul10,
                                                                  kru15, ori15, rul15,
                                                                  relx, rely, lis1, lis2, lis3, input_data_s,
                                                                  map_data_s)


    return target_data_l,input_data_l,map_data_l,target_data_r,input_data_r,map_data_r,target_data_s,input_data_s,map_data_s


#get the dataset of input and target
def target_input_get(fp1):

    with open(fp1, 'rb') as fp:
        data = pickle.load(fp)

    #By the checking process using the reconstruction trajectory, some of the curves are given up.
    error_list = [10,15,16,24,26,41,55,58,67,68,77,106,108,159,160,179,180,181,220,244,
                  253,268,269,287,290,291,304,331,332,340,464,465,470,473,478,500,504,512,525,
                  528,531,532,535,543,554,555,557,580,625,642,650,654,655,677,678,
                  696,698,699,700,705,709,720,723,729,730,731,732,733,734,736,737,738,739,
                  740,741,742,743,744,745,746,751,758,773,785,788,789,791,807,810,811,
                  812,826,834,858,871,897,913,914,921,922,925,932,958,970,978,984,993,999,
                  1001,1002,1004,1005,1006,1022,1023,1025,1031,1049,1051,1052,1054,1062,1084,
                  1093,1094,1098,1115,1137,1141,1142,1155,1198,1203,1211,1213,1215,1217,1219,
                  1222,1223,1224,1233,1242,1243,1244,1255,1270,1293,1309,1310,1312,1313,1319,1325,
                  1349,1351,1354,1356,1368,1379,1382,1397,1398,1399,1410,1426,1428,1429,1430,1441,
                  1448,1449,1450,1475,1510,1511,1512,1519,1520,1522,1529]

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

        #firstly we reas the data we need
        curve_data = data[i]
        cd = curve_data[1]
        pos_x = np.array(cd['pos_x'])
        pos_y = np.array(cd['pos_y'])
        vel_xa = np.array(cd['vel_x'])
        vel_ya = np.array(cd['vel_y'])

        #we use forward-backward filter to remove the noise in the dataset
        pos_x, pos_y = curve_filter(pos_x, pos_y)

        #compute the steering angle,choose the start point and the end point
        ang, num, num_1 = ang_computation(pos_x,pos_y)
        #compute the velocity and acceleration
        vel, acc = vel_computation(pos_x, pos_y, num, num_1)
        # implement an filter to remove some noise in data

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

        #compute the direction and the angle between direction and velocity
        psi, beta, pre_x, pre_y = akman(pos_x, pos_y, vel_xa, vel_ya, vel, ang, num)
        #compute the relative change in x and y coordinates
        relx, rely = rela_change(pos_x,pos_y,num)

        #get the traffic environment from lanelet
        lis0, lis1, lis2, lis3, \
        dic, kru, ori, rul, \
        kru5, ori5, rul5,  \
        kru10, ori10, rul10, \
        kru15, ori15, rul15 = findlane_index(pos_x,pos_y,kind,num)

        cons_list = [250,500,750,1000,1250,1500]

        for konst in cons_list:

            target_data_l, input_data_l, map_data_l, target_data_r, input_data_r, map_data_r, target_data_s, input_data_s, map_data_s = classify_data(
                konst, kind, ang, vel, acc, pos_x, pos_y, psi, beta,
                kru, ori, rul, dic,
                kru5, ori5, rul5,
                kru10, ori10, rul10,
                kru15, ori15, rul15,
                relx, rely, lis1, lis2, lis3, target_data_l, input_data_l,
                map_data_l, target_data_r, input_data_r, map_data_r, target_data_s, input_data_s, map_data_s)

        #instead of using np.random.randint, here is an extra list manually defined,because using the random lsit may cause two problems:
        #(1) It's inconvenient to check the data (2) there is some highly repeated dataset
        extra_lsit = [300,350,400,450,550,600,650,700,800,850,900,950,1050,1100,1150,1200,1300,1350,1400,1450]

        if kind == -1 or kind == 1:

            for ran in extra_lsit:
                target_data_l, input_data_l, map_data_l, target_data_r, input_data_r, map_data_r, target_data_s, input_data_s, map_data_s = classify_data(
                                                                ran, kind, ang, vel, acc, pos_x, pos_y, psi, beta,
                                                                kru, ori, rul, dic,
                                                                kru5, ori5, rul5,
                                                                kru10, ori10, rul10,
                                                                kru15, ori15, rul15,
                                                                relx, rely, lis1, lis2, lis3, target_data_l, input_data_l,
                                                                map_data_l, target_data_r, input_data_r, map_data_r, target_data_s, input_data_s, map_data_s)

    return target_data_l, input_data_l, map_data_l, target_data_r, input_data_r, map_data_r, target_data_s, input_data_s, map_data_s


def train_valid_get():

    target_data_l, input_data_l, map_data_l, target_data_r, input_data_r, map_data_r, target_data_s, input_data_s, map_data_s = target_input_get('allAmsterdamerRing.pickle')

    right = [11,12,34,35,47,48,49,50,54,65,66,70,80,81,82,84,85,87,90,91,93,98,99,104,105,106,123,124,126,130,131,132,133,134,135,
             136,137,138,139,140,141,142,143,144,145,146,147,148,158,159,162,163,165,166,167,168,170,171,172,173,174,175,
             176,177,178,179,180,181,187,195,196,197,198,199,200,201,202,204,205,209,210,211,212,213,214,215,216,217,218,219,
             220,223,231,232,233,234,235,236,237,241,249,250,252,253,256,257,258,259,262,263,264,276,277,278,279,290,291,292,293,
             297,298,299,300,307,318,319,320,321,322,323,324,325,326,331,332,333,334,335,339,352,353,354,355,357,362,363,364,365,
             366,378,379,380,381,382,383,384,385,386,393,394,438,439,442,443,474,477,485,486,487,490,498,499,500,501,503,504,506,
             507,508,509,512,513,514,515,516,519,527,528,529,530,532,533,534,536,537,538,543,559,563,576,582,606,618,619,620,621,
             624,627,628,629,630,653,654,655,656,657,659,660,661,662,663,664]

    left = [0,1,3,4,5,6,7,8,17,18,19,21,22,23,24,25,26,27,28,30,31,32,33,34,35,48,53,54,60,61,62,63,64,65,90,99,100,101,102,105,106,
            107,108,109,110,113,114,116,117,120,121,122,123,124,125,126,127,128,129,130,131,132,133,137,138,139,140,147,148,149,152,
            152,153,154,163,174,175,178,179,180,185,188,197,217,224,225,226,227,228,229,236,237,238,239,241,249,250,253,262,263,267,
            268,269,274,275,276,277,278,279,280,281,282,283,295,303,304,305,306,307,308,312,315,316,317,318,319,322,323,324,325,334,
            335,336,340,341,342,343,344,348,349,350,351,352,353,354,371,372,377,393,394,395,396,423,424,425,430,446,447,448,449,453,
            455,456,457,458,459,460,467,469,470,471,475,476,477,478,480,481,483,484,485,486,488,497,498,499,502,503,510,531,532,553,
            574,575,576,577,578,579]

    straight = [0,1,2,3,9,10,11,12,13,14,15,16,17,18,19,20,23,24,25,26,29,30,31,32,34,39,40,41,42,46,47,48,49,51,52,53,55,56,57,58,
                60,62,63,64,65,66,69,70,72,73,75,76,77,78,79,80,81,82,84,85,87,88,90,92,93,94,95,96,97,101,102,103,110,111,114,116,
                117,118,119,121,124,127,129,131,132,134,135,137,139,141,142,143,144,145,147,151,152,153,155,156,157,158,159,160,162,
                163,164,165,169,171,172,175,177,178,180,181,182,184,185,186,187,188,189,190,191,193,195,196,198,199,200,201,203,204,
                205,206,207,208,211,213,214,216,217,221,224,226,231,232,233,234,235,236,241,243,244,246,248,252,254,255,260,261,262,
                263,267,268,269,271,273,274,279,281,282,283,288,290,292,293,294,295,297,298,299,300,301,302,304,305,310,311,312,313,
                314,318,321,322,323,324,325,326,328,329,332,333,334,336,337,338,339,340,341,350,357,358,359,361,362,363,364,365,366,
                367,368,369,371,372,373,374,375,382,383,384,385,386,387,388,389,390,391,396,397,398,399,400,401,403,406,409,410,414,
                415,416,417,420,421,427,428,432,434,437,438,442,445,446,447,448,455,457,459,460,462,465,467,468,471,472,474,477,481,
                488,489,490,491,495,497,498,499,500,501,502,513,517,518,530,534,540,541,543,546,550,554,557,558,559,565,566,568,569,
                572,574,575,577,578,580,583,584,585,587,588,589,591,593,594,595,596,598,602,609,610,612,616,620,624,626,627,628,630,
                632,633,639,640,652,659,661,662,664,669,670,673]

    input_data = []
    target_data = []
    map_data = []

    for i in range(673):

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

    test_list = [4, 9, 10, 26, 33, 47, 58, 60, 63, 66, 76, 88, 104, 124, 134, 195]

    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    x_test = []
    y_test = []
    m_train = []
    m_valid = []
    m_test = []
    xtv = []
    ytv = []
    mtv = []

    print(len(input_data))

    for i in range(len(input_data)):

        if i in test_list:
            x_test.append(input_data[i])
            y_test.append(target_data[i])
            m_test.append(map_data[i])
        else:
            xtv.append(input_data[i])
            ytv.append(target_data[i])
            mtv.append(map_data[i])

    for k in range(len(xtv)):
        x_train.append(xtv[k])
        y_train.append(ytv[k])
        m_train.append(mtv[k])

    return x_train,y_train,x_valid,y_valid,x_test,y_test,m_train,m_valid,m_test


def plotCenterlineCharacteristics(lanelet):
    # Get lanelet centerline 2d coordinates
    coords = np.array([(c.x, c.y) for c in lanelet.leftBound])
    plt.plot(coords[:, 0], coords[:, 1], 'o-', label='leftbound')
    coords_1 = np.array([(c.x, c.y) for c in lanelet.rightBound])
    plt.plot(coords_1[:, 0], coords_1[:, 1], 'o-', label='centerline')


#This function is to check if the sampled data is correct
def check_data():

    target_data_l, input_data_l, map_data_l, target_data_r, input_data_r, map_data_r, target_data_s, input_data_s, map_data_s = target_input_get('allAmsterdamerRing.pickle')

    for i in range(len(target_data_s)):

        straight = [3, 4, 5, 8, 10, 11, 12]

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
        map1 = []

        for t in range(len(training_d)):

            input_per_time = training_d[t]
            targte_per_time = target_d[t]
            map_per_time = map_d[t]
            map_1 = map_per_time[0]
            map1.append(map_1)

            ang.append(targte_per_time[0])
            acc.append(targte_per_time[1])

            pox.append(map_per_time[3])
            poy.append(map_per_time[4])

            kru.append(input_per_time[5] * 0.1)
            kru5.append(input_per_time[9] * 0.1)
            kru10.append(input_per_time[13] * 0.1)
            kru15.append(input_per_time[17] * 0.1)

            ori.append(input_per_time[7])
            ori5.append(input_per_time[11])
            ori10.append(input_per_time[15])
            ori15.append(input_per_time[19])

            gam.append(input_per_time[8])
            gam5.append(input_per_time[12])
            gam10.append(input_per_time[16])
            gam15.append(input_per_time[20])

            psi.append(input_per_time[2])
            beta.append(input_per_time[3])
            velo.append(input_per_time[4])


        file_path = "Amsterdamer_Intersection_Lanelet.osm"
        # Use the projector to get the map
        projector = UtmProjector(lanelet2.io.Origin(50.76599713889, 6.06099834167))
        map = lanelet2.io.load(file_path, projector)
        plt.axis("equal")
        for m in range(len(map1)):
            lane = map.laneletLayer[map1[m]]
            plotCenterlineCharacteristics(lane)
        plt.scatter(pox, poy)
        plt.show()

        plt.title('acceleration and velocity')
        plt.plot(acc)
        plt.plot(velo)
        plt.show()

        plt.title('curvature at current state, 5, 10, 15 meter after now')
        plt.plot(kru,label='0')
        plt.plot(kru5,label='5')
        plt.plot(kru10,label='10')
        plt.plot(kru15,label='15')
        plt.legend()
        plt.show()

        plt.title('steering angle and beta angle')
        plt.plot(beta)
        plt.plot(ang)
        plt.show()

        plt.title('orientation and psi angle')
        plt.plot(ori,label='0')
        plt.plot(ori5,label='5')
        plt.plot(ori10,label='10')
        plt.plot(ori15,label='15')
        plt.plot(psi,label='psi')
        plt.legend()
        plt.show()

        plt.title('relative angle')
        plt.plot(gam,label='0')
        plt.plot(gam5,label='5')
        plt.plot(gam10,label='10')
        plt.plot(gam15,label='15')
        plt.legend()
        plt.show()


def check_data_2():

    target_data_l, input_data_l, map_data_l, target_data_r, input_data_r, map_data_r, target_data_s, input_data_s, map_data_s = target_input_get('allAmsterdamerRing.pickle')

    for i in range(len(target_data_r)):

        print(i)

        training_d = input_data_r[i]
        target_d = target_data_r[i]
        map_d = map_data_r[i]

        pox = []
        poy = []

        krua = []

        gama = []

        ang = []
        acc = []
        velo = []
        psi = []
        dis = []
        map1 = []

        for t in range(len(training_d)):

            input_per_time = training_d[t]
            targte_per_time = target_d[t]
            map_per_time = map_d[t]
            map_1 = map_per_time[0]
            map1.append(map_1)

            ang.append(targte_per_time[0])
            acc.append(targte_per_time[1])

            pox.append(map_per_time[3])
            poy.append(map_per_time[4])

            kru_average = (input_per_time[5]+input_per_time[8]+input_per_time[11]+input_per_time[14])/4
            krua.append(kru_average)

            gam_average = (input_per_time[7]+input_per_time[10]+input_per_time[13]+input_per_time[16])/4
            gama.append(gam_average)

            psi.append(input_per_time[2])
            dis.append(input_per_time[3])
            velo.append(input_per_time[4])

        plt.scatter(pox, poy)
        plt.show()

        plt.title('curvature, steeringangle,  and relative angle')
        plt.plot(krua,label= 'curvature')
        plt.plot(gama,label= 'relative angle (average)')
        plt.plot(ang,label='steering angle')
        plt.xlabel('time steps')
        plt.legend()
        plt.show()

        plt.title('curvature, acceleration and relative angle')
        plt.plot(krua, label='curvature')
        plt.plot(gama, label='relative angle (average)')
        plt.plot(acc, label='acceleration')
        plt.xlabel('time steps')
        plt.legend()
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
