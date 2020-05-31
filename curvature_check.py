import lanelet2
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
from lanelet2.core import BasicPoint2d
from lanelet2.projection import UtmProjector
file_path = "Amsterdamer_Intersection_Lanelet.osm"

def find_lane1(xf,yf,x,y,map):
    #Firstly the closest lanelets will be collected.
    plocal = BasicPoint2d(x, y)
    lanelets = lanelet2.geometry.findNearest(map.laneletLayer, plocal, 1)

    if len(lanelets) == 1:
        # there is only one lane,especially in the area before the intersection, That lane is the one we need
        lane = lanelets[0]
        #For every lane in "lanelets", 0 is the distance between point and lane, 1 is the lanelet
        lan = lane[1]

    else:
        #If there is more than one lanelets, the lanelets having 0 distance have priority
        Lane = []
        for i in range(len(lanelets)):
            lane_index = lanelets[i]
            if lane_index[0] == 0:
                Lane.append(lane_index)

        lane_init = lanelets[0]
        amin = lane_init[0]   #smallest distance
        lan = lane_init[1]   #lanelet
        # If this point belongs to no lanelet, then the closest one will be chosen
        if len(Lane)==0:
            for n in range(len(lanelets)):
                lane_index = lanelets[n]
                dis = lane_index[0]
                if dis < amin:
                    amin = dis
                    lan = lane_index[1]

        # For the intersectant lanes, the one having smallst angle with velocity will be chosen
        else:
            cos_max = -1
            for i in range(len(Lane)):
                lane_index = Lane[i]
                centerline = lane_index[1].centerline
                vx = x - xf
                vy = y - yf
                vecx = centerline[-1].x - centerline[0].x
                vecy = centerline[-1].y - centerline[0].y
                cos_t = (vecx * vx + vecy * vy) / (math.sqrt((vecx ** 2 + vecy ** 2) * (vx ** 2 + vy ** 2)))
                if cos_t > cos_max:
                    cos_max = cos_t
                    lan = lane_index[1]
    ind = lan.id
    return ind


def find_lane2(xf,yf,x,y,map,lane_bevor,graph,kind):

    plocal = BasicPoint2d(x, y)
    lanelets = lanelet2.geometry.findNearest(map.laneletLayer, plocal, 1)

    if len(lanelets) == 1:
        lane = lanelets[0]
        lan = lane[1]
    else:
        Lane = []
        for i in range(len(lanelets)):
            lane_index = lanelets[i]
            if lane_index[0] == 0:
                Lane.append(lane_index)

        lane_init = lanelets[0]
        amin = lane_init[0]
        lan = lane_init[1]

        if len(Lane)==0:
            for n in range(len(lanelets)):
                lane_index = lanelets[n]
                dis = lane_index[0]
                if dis < amin:
                    amin = dis
                    lan = lane_index[1]

        else:
            cos_max = -1
            for i in range(len(Lane)):
               lane_index = Lane[i]
               lanelet =  map.laneletLayer[lane_index[1].id]
               toLanelet2 = map.laneletLayer[lane_bevor]
               route = graph.getRoute(lanelet, toLanelet2)
               #The lane must have succeeding relationship to the lane_bevor
               if route != None:
                   centerline = lane_index[1].centerline
                   vecx = centerline[-1].x - centerline[0].x
                   vecy = centerline[-1].y - centerline[0].y
                   vx = x - xf
                   vy = y - yf
                   cos_t = (vecx * vx + vecy * vy) / (math.sqrt((vecx ** 2 + vecy ** 2) * (vx ** 2 + vy ** 2)))
                   if cos_t > cos_max:
                       cos_max = cos_t
                       lan = lane_index[1]

    ind = lan.id
    if kind==-1:
        if ind in [208860,188989]:
            ind = 188986
        elif ind in [188990,198781,198786,208603]:
            ind = 198738
        elif ind in [208553]:
            ind = 208661
        elif ind in [208543]:
            ind = 198799
        else:
            ind = ind
    elif kind == 1:
        if ind in [198786]:
            ind = 208603
        elif ind in [208860,188989]:
            ind = 208710
        else:
            ind = ind
    else:
        if ind in [188984]:
            ind = 188985
        elif ind in [188986]:
            ind = 188989
        elif ind in [198738]:
            ind = 188990
        elif ind in [208603]:
            ind = 198786
        elif ind in [208661]:
            ind = 208553
        elif ind in [208710]:
            ind = 208860
        ind = ind

    return ind


def Krummung_rechnen(laneid,lane,posx,posy):

    centerline = lane.centerline
    center_2d =  lanelet2.geometry.to2D(centerline)
    # computation of distance to the centerline, left is positive, right is negative
    arc_coord = lanelet2.geometry.toArcCoordinates(center_2d, BasicPoint2d(posx, posy))
    dis_to_center = arc_coord.distance

    rightline =  lane.rightBound
    leftline = lane.leftBound
    #The line having more points will be chosen
    if len(rightline)>len(leftline):
        line = rightline
    else:
        line = leftline

    #find the two closest points in the lane
    short_dis = math.sqrt((posx - line[0].x) ** 2 + (posy - line[0].y) ** 2)
    short_ind = 0

    for i in range(1,len(line)):
        distance = math.sqrt((posx - line[i].x) ** 2 + (posy - line[i].y) ** 2)
        if distance < short_dis:
            short_dis = distance
            short_ind = i
    # After choosing the closest point, the second closest point will be chosen
    if short_ind == 0:
        short_ind_1 = 1
    elif short_ind == len(line) - 1:
        short_ind_1 = len(line) - 2
    else:
        ind_1 = short_ind - 1
        ind_2 = short_ind + 1
        dis_1 = math.sqrt((line[ind_1].x - posx) ** 2 + (line[ind_1].y - posy) ** 2)
        dis_2 = math.sqrt((line[ind_2].x - posx) ** 2 + (line[ind_2].y - posy) ** 2)
        if dis_1 < dis_2:
            short_ind_1 = ind_1
        else:
            short_ind_1 = ind_2

    #for each direction there exsits one lane for left-turn and one lane for right-turn
    turn_list = [188986, 198738, 208661, 208834, 188993, 208603, 208710, 208641]

    if laneid in turn_list:
            # distinguish between right and left
            # Normal vector of initial curve
            v1_x = line[1].y - line[0].y
            v1_y = line[0].x - line[1].x
            v2_x = line[-1].x - line[-2].x
            v2_y = line[-1].y - line[-2].y
            # The steering angle of left-turn and right-turn ought to be different
            if (v1_x * v2_x + v1_y * v2_y) > 0:
                # right-turn curve
                vorzeichen = -1
            else:
                vorzeichen = 1

            #At the start or at the end of each curve
            if short_ind == 0 or short_ind == len(line) - 1:
                kappa = 0.05 * vorzeichen
            #choose the three points to calculate the curvature
            else:
                p1_x = line[short_ind - 1].x
                p1_y = line[short_ind - 1].y
                p2_x = line[short_ind].x
                p2_y = line[short_ind].y
                p3_x = line[short_ind + 1].x
                p3_y = line[short_ind + 1].y

                k1 = (p2_y - p1_y) / (p2_x - p1_x)
                k1_1 = -1 / k1
                x1 = (p2_x + p1_x) / 2
                y1 = (p2_y + p1_y) / 2

                k2 = (p3_y - p2_y) / (p3_x - p2_x)
                k2_1 = -1 / k2
                x2 = (p3_x + p2_x) / 2
                y2 = (p3_y + p2_y) / 2

                x_1 = (k1_1 * x1 - k2_1 * x2 - y1 + y2) / (k1_1 - k2_1)
                y_1 = (-k2_1 * k1_1 * x1 + y1 * k2_1 + k1_1 * k2_1 * x2 - k1_1 * y2) / (k2_1 - k1_1)
                r = math.sqrt((x_1 - p2_x) ** 2 + (y_1 - p2_y) ** 2)
                kappa = 1 / r
                kappa = kappa * vorzeichen

    else:
        kappa = 0

    # choose the two-closest points on the curve
    # compute the direction
    if short_ind<short_ind_1:
        l_x1 = line[short_ind].x
        l_y1 = line[short_ind].y
        l_x2 = line[short_ind_1].x
        l_y2 = line[short_ind_1].y
    else:
        l_x1 = line[short_ind_1].x
        l_y1 = line[short_ind_1].y
        l_x2 = line[short_ind].x
        l_y2 = line[short_ind].y

    # computation of angle deviation
    x_zeichen = l_x2 - l_x1
    y_zeichen = l_y2 - l_y1
    zeichen = (l_y2 -l_y1)/(l_x2-l_x1)
    if x_zeichen > 0:
        if y_zeichen>0:
            fai = math.atan(zeichen)
        else:
            fai = math.atan(zeichen)+2*math.pi
    else:
        fai = math.atan(zeichen) + math.pi

    return kappa, dis_to_center,fai


def rule_info(lane):

    right_turn = [188986,198738,208661,208834]
    left_turn = [188993,208603,208710,208641]
    right_straight = [198702,208553,188998,208809,188984]
    left_straight = [188984,198781,208636,208705]

    if lane in right_turn:
        rule = -1
    elif lane in left_turn:
        rule = 1
    elif lane in right_straight:
        rule = -0.5
    elif lane in left_straight:
        rule = 0.5
    else:
        rule = 0
    return rule


def findlane_index(pos_x,pos_y):
    #Use the projector to get the map
    projector = UtmProjector(lanelet2.io.Origin(50.76599713889, 6.06099834167))
    path = file_path
    map = lanelet2.io.load(path, projector)
    #Use routing to get the graph
    trafficRules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                 lanelet2.traffic_rules.Participants.Vehicle)
    graph = lanelet2.routing.RoutingGraph(map,trafficRules)

    # distinguish between left-turn,right-turn,go-straight
    dis = 3
    num = 0
    dist = 0
    while (dist < dis):
        num = num + 1
        dist = math.sqrt((pos_x[num] - pos_x[0]) ** 2 + (pos_y[num] - pos_y[0]) ** 2)

    num_1 = -1
    dist_1 = 0
    while (dist_1 < dis):
        num_1 = num_1 - 1
        dist_1 = math.sqrt((pos_x[num_1] - pos_x[-1]) ** 2 + (pos_y[num_1] - pos_y[-1]) ** 2)
    num_1 = len(pos_x) + num_1 + 1
    v1_x = pos_y[num] - pos_y[0]
    v1_y = pos_x[0] - pos_x[num]
    v2_x = pos_x[-1] - pos_x[num_1]
    v2_y = pos_y[-1] - pos_y[num_1]

    # The steering angle of left-turn and right-turn ought to be different
    Tmp = (v1_x * v2_x + v1_y * v2_y) / math.sqrt((v1_x**2 + v1_y**2) * (v2_x**2+ v2_y**2))
    print(Tmp)
    if Tmp > 0.45:
        # right-turn curve
        kind = -1
    elif Tmp < -0.4:
        # left-turn curve
        kind = 1
    else:
        kind = 0
    print(kind)
    #map each point to the lane it belongs to
    j = 1
    change = pos_x[j] - pos_x[0]
    while(change ==0):
        j += 1
        change = pos_x[j] - pos_x[0]

    lane_bevor = find_lane1(pos_x[0],pos_y[0],pos_x[j],pos_y[j],map)
    list1 = []
    list1.append(lane_bevor)

    for i in range(1, len(pos_x)):
        k = i
        change = pos_x[i] - pos_x[k]
        while(change==0):
            k = k - 1
            change = pos_x[i] - pos_x[k]
        laneid = find_lane2(pos_x[k],pos_y[k],pos_x[i],pos_y[i],map,lane_bevor,graph,kind)
        list1.append(laneid)
        lane_bevor = laneid

    print(list1)
    kru = []
    dis = []
    fai = []
    rul = []
    for i in range(len(list1)):
        rul.append(rule_info(list1[i]))
        lanelet_layer= map.laneletLayer[list1[i]]
        kru_, dis_,fai_ = Krummung_rechnen(list1[i],lanelet_layer,pos_x[i],pos_y[i])
        kru.append(kru_)
        dis.append(dis_)
        fai.append(fai_)
    return rul,kru,dis,fai

fp1 = 'allAmsterdamerRingV2.pickle'
with open(fp1, 'rb') as fp:
    data = pickle.load(fp)

for i in range(100):
    curve = data[i]
    curve_data = curve[1]
    pos_x = curve_data['pos_x']
    pos_y = curve_data['pos_y']
    pos_x = np.array(pos_x)
    pos_y = np.array(pos_y)

    plt.subplot(3, 2, 1)
    plt.title("curve")
    plt.scatter(pos_x, pos_y, s=1)

    rul, kru, dis, fai = findlane_index(pos_x, pos_y)
    plt.subplot(3, 2, 2)
    plt.title("regular info")
    plt.plot(rul, color="r")
    plt.subplot(3, 2, 3)
    plt.title("curvature")
    plt.plot(kru, color="y")
    plt.subplot(3, 2, 4)
    plt.title("distance")
    plt.plot(dis, color="g")
    plt.subplot(3, 2, 5)
    plt.title("direction")
    plt.plot(fai, color="b")
    plt.show()

