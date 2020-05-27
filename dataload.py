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


#target-data, velocity computation
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
    #the computation of acceleration
    for i in range(nu-1, nu_1-1):
        dx1 = pos_x[i] - pos_x[i -1]
        dy1 = pos_y[i] - pos_y[i -1]
        d1 = math.sqrt(dx1 ** 2 + dy1 ** 2)
        v1 = d1 / dt

        dx2 = pos_x[i + 1] - pos_x[i]
        dy2 = pos_y[i + 1] - pos_y[i]
        d2 = math.sqrt(dx2 ** 2 + dy2 ** 2)
        v2 = d2 / dt

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
    #Computation of initial value of pfai
    if x_zeichen > 0:
        pfai_init = math.atan(y_zeichen / x_zeichen)
    else:
        pfai_init = math.atan(y_zeichen / x_zeichen) + math.pi

    #Computation of initial value of beta
    if vel_xinit > 0:
        beta_init = math.atan(vel_yinit / vel_xinit) - pfai_init
    elif vel_xinit < 0:
        beta_init = math.atan(vel_yinit / vel_xinit) + math.pi - pfai_init
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

    return pfai,beta,pre_x,pre_y


#get the dataset of input and target
def target_input_get(fp1):

    with open(fp1, 'rb') as fp:
        data = pickle.load(fp)

    error_list = [10,16,24,26,38,40,41,55,58,60,61,67,68,70,77,78,79,82,84,85,86,89,92,99,106,108,128,129,134,141,143,145,148,151,159,160,163,
                  164,176,179,180,181,182,193,202,213,216,220,221,222,227,228,233,241,243,244,245,248,249,251,
                  253,268,269,283,285,286,287,289,290,292,293,294,296,298,299,301,302,304,308,309,320,328,332,339,
                  342,352,353,357,360,368,383,386,387,390,401,407,411,414,417,421,423,429,432,434,435,444,445,453,455,458,
                  461,464,465,466,470,473,478,480,484,500,512,515,519,527,528,530,532,534,535,554,555,557,564,565,569,576,580,
                  583,591,596,599,602,612,619,623,624,625,632,640,643,650,651,652,655,660,666,675,687,690,692,696,699,700,
                  705,709,720,721,725,726,729,730,731,734,735,736,737,738,739,740,741,742,743,744,745,746,747,748,749,750,751,
                  752,753,755,758,759,761,766,769,773,782,784,786,789,791,795,799,806,809,810,811,812,813,816,818,819,820,
                  834,854,855,859,870,878,882,883,884,887,888,893,897,899,904,913,914,916,922,925,926,930,936,949,958,970,
                  971,978,984,992,993,994,995,996,998,999,1000,1001,1002,1006,1007,1008,1009,1012,1022,1025,1035,1048,1049,
                  1051,1052,1059,1062,1071,1075,1081,1083,1091,1093,1094,1096,1098,1101,1103,1123,1129,1130,1134,1139,1141,1142,
                  1143,1144,1145,1146,1149,1151,1162,1163,1169,1182,1183,1184,1198,1200,1203,1204,1210,1211,1213,1214,1215,
                  1217,1218,1219,1220,1221,1222,1223,1225,1232,1233,1242,1243,1247,1249,1252,1253,1255,1268,1270,1274,1276,1293,1299,
                  1309,1310,1312,1313,1315,1317,1319,1325,1329,1332,1340,1344,1348,1349,1351,1354,1356,1357,1366,1367,1368,
                  1379,1380,1381,1382,1383,1387,1395,1397,1398,1399,1408,1410,1426,1427,1428,1430,1441,1448,1449,1450,1456,
                  1458,1463,1473,1475,1481,1488,1492,1496,1509,1510,1511,1512,1519,1520,1522,1525,1529]
    target_data = []
    input_data = []
    #the number in "range" determined how many curves do we need
    for i in range(len(data)):

        if i in error_list:
            continue

        info = []
        info2 = []
        info3 = []
        info4 = []
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

        for k in range(20, len(x)):
            pos_x.append(x[k])
            pos_y.append(y[k])
            vel_xa.append(xa[k])
            vel_ya.append(ya[k])

        angle, num, num_1 = ang_computation(pos_x,pos_y)
        vel,acc = vel_computation(pos_x, pos_y, num, num_1)
        angle,vel,acc = pool_angle(angle,vel,acc)

        # for every curve we take 150 position data
        if len(vel)>600:
            for m in range(300):
                if m % 5 == 0:
                    info.append([angle[m]*100, acc[m]])
            target_data.append(info)

            krummung, dis_to_line, angle_to_line = findlane_index(pos_x, pos_y)
            pfai, beta, pre_x, pre_y = akman(pos_x, pos_y, vel_xa, vel_ya, vel, angle, num)
            x_init = pre_x[0]
            y_init = pre_y[0]

            for t in range(num, num + 300):
                m = t - num
                if m % 5 == 0:
                    x_norm = pos_x[t] - x_init
                    y_norm = pos_y[t] - y_init
                    angle_deviation = pfai[t -num] - angle_to_line[t]
                    info2.append([x_norm, y_norm, pfai[t-num], beta[t-num], krummung[t], dis_to_line[t], angle_deviation,vel[t-num]])
            input_data.append(info2)

            for m in range(300,600):
                if m % 5 == 0:
                    info3.append([angle[m]*100, acc[m]])
            target_data.append(info3)

            x_init_1 = pre_x[300]
            y_init_1 = pre_y[300]

            for t in range(num+300, num + 600):
                m = t - num
                if m % 5 == 0:
                    x_norm = pos_x[t] - x_init_1
                    y_norm = pos_y[t] - y_init_1
                    angle_deviation = pfai[t-num] - angle_to_line[t]
                    info4.append([x_norm, y_norm, pfai[t-num], beta[t-num], krummung[t], dis_to_line[t], angle_deviation,vel[t-num]])
            input_data.append(info4)

        elif len(vel)>300:

            for m in range(300):
                if m % 5 ==0:
                   info.append([angle[m]*100,acc[m]])
            target_data.append(info)

            krummung, dis_to_line, angle_to_line = findlane_index(pos_x,pos_y)
            pfai,beta,pre_x,pre_y = akman(pos_x,pos_y,vel_xa,vel_ya,vel,angle,num)

            x_init = pre_x[0]
            y_init = pre_y[0]

            for t in range(num,num+300):
                m = t-num
                if m % 5 ==0:
                    x_norm = pos_x[t] - x_init
                    y_norm = pos_y[t] - y_init
                    angle_deviation = pfai[t-num] - angle_to_line[t]
                    info2.append([x_norm,y_norm,pfai[t-num],beta[t-num],krummung[t],dis_to_line[t],angle_deviation,vel[t-num]])
            input_data.append(info2)

    return target_data,input_data


def target_input_get_1(fp1):

    with open(fp1, 'rb') as fp:
        data = pickle.load(fp)

    error_list = [10,16,24,26,38,40,41,55,58,60,61,67,68,70,77,78,79,82,84,85,86,89,92,99,106,108,128,129,134,141,143,145,148,151,159,160,163,
                  164,176,179,180,181,182,193,202,213,216,220,221,222,227,228,233,241,243,244,245,248,249,251,
                  253,268,269,283,285,286,287,289,290,292,293,294,296,298,299,301,302,304,308,309,320,328,332,339,
                  342,352,353,357,360,368,383,386,387,390,401,407,411,414,417,421,423,429,432,434,435,444,445,453,455,458,
                  461,464,465,466,470,473,478,480,484,500,512,515,519,527,528,530,532,534,535,554,555,557,564,565,569,576,580,
                  583,591,596,599,602,612,619,623,624,625,632,640,643,650,651,652,655,660,666,675,687,690,692,696,699,700,
                  705,709,720,721,725,726,729,730,731,734,735,736,737,738,739,740,741,742,743,744,745,746,747,748,749,750,751,
                  752,753,755,758,759,761,766,769,773,782,784,786,789,791,795,799,806,809,810,811,812,813,816,818,819,820,
                  823,834,854,855,859,870,878,882,883,884,887,888,893,897,899,904,913,914,916,922,925,926,930,936,949,958,970,
                  971,978,984,992,993,994,995,996,998,999,1000,1001,1002,1006,1007,1008,1009,1012,1022,1025,1035,1048,1049,
                  1051,1052,1059,1062,1071,1075,1081,1083,1091,1093,1094,1096,1098,1101,1103,1123,1129,1130,1134,1139,1141,1142,
                  1143,1144,1145,1146,1149,1151,1162,1163,1169,1182,1183,1184,1198,1200,1203,1204,1210,1211,1213,1214,1215,
                  1217,1218,1219,1220,1221,1222,1223,1225,1232,1233,1242,1243,1247,1249,1252,1253,1255,1268,1270,1274,1276,1293,1299,
                  1309,1310,1312,1313,1315,1317,1319,1325,1329,1332,1340,1344,1348,1349,1351,1354,1356,1357,1366,1367,1368,
                  1379,1380,1381,1382,1383,1387,1395,1397,1398,1399,1408,1410,1426,1427,1428,1430,1441,1448,1449,1450,1456,
                  1458,1463,1473,1475,1481,1488,1492,1496,1509,1510,1511,1512,1519,1520,1522,1525,1529]
    target_data = []
    input_data = []
    #the number in "range" determined how many curves do we need
    for i in range(208):

        if i in error_list:
            continue

        info = []
        info2 = []
        info3 = []
        info4 = []
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

        for k in range(20, len(x)):
            pos_x.append(x[k])
            pos_y.append(y[k])
            vel_xa.append(xa[k])
            vel_ya.append(ya[k])

        angle, num, num_1 = ang_computation(pos_x,pos_y)
        vel,acc = vel_computation(pos_x, pos_y, num, num_1)
        angle,vel,acc = pool_angle(angle,vel,acc)

        # for every curve we take 150 position data
        if len(vel)>300:
            for m in range(150):
                info.append([angle[m]*100, acc[m]])
            target_data.append(info)

            krummung, dis_to_line, angle_to_line = findlane_index(pos_x, pos_y)
            pfai, beta, pre_x, pre_y = akman(pos_x, pos_y, vel_xa, vel_ya, vel, angle, num)

            x_init = pre_x[0]
            y_init = pre_y[0]

            for t in range(num, num + 150):
                x_norm = pos_x[t] - x_init
                y_norm = pos_y[t] - y_init
                angle_deviation = pfai[t -num] - angle_to_line[t]
                pfai_norm = pfai[t -num]
                info2.append([x_norm, y_norm, pfai_norm, beta[t - num], krummung[t], dis_to_line[t], angle_deviation,vel[t - num]])
            input_data.append(info2)

            for m in range(150,300):
                info3.append([angle[m]*100, acc[m]])
            target_data.append(info3)

            x_init_1 = pre_x[150]
            y_init_1 = pre_y[150]

            for t in range(num+150, num + 300):

                x_norm = pos_x[t] - x_init_1
                y_norm = pos_y[t] - y_init_1
                angle_deviation = pfai[t - num] - angle_to_line[t]
                pfai_norm = pfai[t - num]
                info4.append([x_norm, y_norm, pfai_norm, beta[t - num], krummung[t], dis_to_line[t], angle_deviation,vel[t - num]])
            input_data.append(info4)

        elif len(vel)>150:

            for m in range(150):
                info.append([angle[m]*100,acc[m]])
            target_data.append(info)

            krummung, dis_to_line, angle_to_line = findlane_index(pos_x,pos_y)
            pfai,beta,pre_x,pre_y = akman(pos_x,pos_y,vel_xa,vel_ya,vel,angle,num)

            x_init = pre_x[0]
            y_init = pre_y[0]

            for t in range(num,num+150):
                x_norm = pos_x[t] - x_init
                y_norm = pos_y[t] - y_init
                angle_deviation = pfai[t-num] - angle_to_line[t]
                pfai_norm = pfai[t-num]
                info2.append([x_norm,y_norm,pfai_norm,beta[t-num],krummung[t],dis_to_line[t],angle_deviation,vel[t-num]])
            input_data.append(info2)

    return target_data,input_data

def train_valid_get():

    target_data, input_data= target_input_get('allAmsterdamerRing.pickle')

    #traing_set:validation_set = 4:1
    train_split = 384

    x_train = []
    y_train = []
    x_valid = []
    y_valid = []

    for i in range(train_split):
        x_train.append(input_data[i])
        y_train.append(target_data[i])

    for i in range(train_split, len(target_data)):
        x_valid.append(input_data[i])
        y_valid.append(target_data[i])

    return x_train,y_train,x_valid,y_valid

