import lanelet2
import pickle
import math
from lanelet2.core import BasicPoint2d
from lanelet2.projection import UtmProjector
import matplotlib.pyplot as plt
file_path = "Amsterdamer_Intersection_Lanelet.osm"

def find_lane1(xf,yf,x,y,map):

    plocal = BasicPoint2d(xf, yf)
    lanelets = lanelet2.geometry.findNearest(map.laneletLayer, plocal, 1)

    vx = x - xf
    vy = y - yf

    # here we will use the vector of velocity. In our neural network, the position of the last time step will be used.
    if len(lanelets) == 1:
        # there is only one lane,especially in the area before the intersection
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
              centerline = lane_index[1].centerline
              vecx = centerline[-1].x - centerline[0].x
              vecy = centerline[-1].y - centerline[0].y
              cos_t = (vecx * vx + vecy * vy) / (math.sqrt((vecx ** 2 + vecy ** 2) * (vx ** 2 + vy ** 2)))
              if cos_t > cos_max:
                 cos_max = cos_t
                 lan = lane_index[1]

    return lan


def find_lane2(xf,yf,x,y,lane_bevor,map,graph):

    plocal = BasicPoint2d(x, y)
    lanelets = lanelet2.geometry.findNearest(map.laneletLayer, plocal, 1)

    vx = x - xf
    vy = y - yf
    cos_max = -1
    # here we will use the vector of velocity. In our neural network, the position of the last time step will be used.
    if len(lanelets) == 1:
        # there is only one lane,especially in the area before the intersection
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
            for i in range(len(Lane)):

               lane_index = Lane[i]
               lanelet =  map.laneletLayer[lane_index[1].id]
               toLanelet2 = map.laneletLayer[lane_bevor.id]
               route = graph.getRoute(lanelet, toLanelet2)

               if route != None:
                  centerline = lane_index[1].centerline
                  vecx = centerline[-1].x - centerline[0].x
                  vecy = centerline[-1].y - centerline[0].y
                  cos_t = (vecx * vx + vecy * vy) / (math.sqrt((vecx ** 2 + vecy ** 2) * (vx ** 2 + vy ** 2)))
                  if cos_t > cos_max:
                     cos_max = cos_t
                     lan = lane_index[1]

    return lan


#this function is for computation of curvature, distance to centerline, angle deviation
def Krummung_rechnen(lane,x,y):

    centerline= lane.centerline
    rightline = lane.rightBound

    #choose two closest points
    short_dis = math.sqrt((x-centerline[0].x)**2 + (y-centerline[0].y)**2)
    short_ind = 0

    #firstly is the closest point chosen
    for i in range(1,len(centerline)-1):
        p_x = centerline[i].x
        p_y = centerline[i].y
        distance = math.sqrt((x-p_x)**2 + (y-p_y)**2)
        if distance < short_dis:
            short_dis = distance
            short_ind = i

    if short_ind==0:
        short_ind_1 = 1
    elif short_ind==len(centerline)-1:
        short_ind_1 = len(centerline)-2
    else:
        ind_1 = short_ind - 1
        ind_2 = short_ind + 1
        dis_1 = math.sqrt((centerline[ind_1].x - x)**2 + (centerline[ind_1].y - y)**2)
        dis_2 = math.sqrt((centerline[ind_2].x - x)**2 + (centerline[ind_2].y - y)**2)
        if dis_1<dis_2:
            short_ind_1 = ind_1
        else:
            short_ind_1 = ind_2
   
    #computation of curvature
    if len(rightline) <= 2:
        kappa = 0
    else:
        p1_x = centerline[short_ind-1].x
        p1_y = centerline[short_ind-1].y
        p2_x = centerline[short_ind].x
        p2_y = centerline[short_ind].y
        p3_x = centerline[short_ind+1].x
        p3_y = centerline[short_ind+1].y

        k1 = (p2_y - p1_y) / (p2_x - p1_x)
        k1_1 = -1 / k1
        x1 = (p2_x + p1_x) / 2
        y1 = (p2_y + p1_y) / 2

        k2 = (p3_y - p2_y) / (p3_x - p2_x)
        k2_1 = -1 / k2
        x2 = (p3_x + p2_x) / 2
        y2 = (p3_y + p2_y) / 2

        if (k1_1 - k2_1) <0.0001:
            kappa = 0
        else:
            x_1 = (k1_1 * x1 - k2_1 * x2 - y1 + y2) / (k1_1 - k2_1)
            y_1 = (-k2_1 * k1_1 * x1 + y1 * k2_1 + k1_1 * k2_1 * x2 - k1_1 * y2) / (k2_1 - k1_1)
            r = math.sqrt((x_1 - p2_x) ** 2 + (y_1 - p2_y) ** 2)
            kappa = 1/r

    #computation of distance to centerline
    l_x1 = centerline[short_ind].x
    l_y1 = centerline[short_ind].y
    l_x2 = centerline[short_ind_1].x
    l_y2 = centerline[short_ind_1].y

    A = l_y2 - l_y1
    B = l_x1 - l_x2
    C = l_x2*l_y1 - l_x1*l_y2
    Tmp = (l_y1 - l_y2)*x + (l_x2 - l_x1)*y + l_x1*l_y2 - l_x2*l_y1
    dis_to_center = (A * x + B * y + C) / math.sqrt(A ** 2 + B ** 2)
    if Tmp < 0:
        dis_to_center = -dis_to_center

    #computation of angle deviation
    fai = math.atan((l_y2-l_y1)/(l_x2-l_x1))

    return kappa,dis_to_center,fai


def rule_info(lane):

    right_turn = [188986,198738,208661,208834]
    left_turn = [188993,208603,208710,208641]
    right_straight = [198702,208553,188998,208697,188984]
    left_straight = [188984,198781,208636,208705]

    if lane in right_turn:
        rule = 1
    elif lane in left_turn:
        rule = -1
    elif lane in right_straight:
        rule = 0.5
    elif lane in left_straight:
        rule = -0.5
    else:
        rule = 0

    return rule


def findlane_index(pos_x,pos_y):

    projector = UtmProjector(lanelet2.io.Origin(50.76599713889, 6.06099834167))
    path = file_path
    map = lanelet2.io.load(path, projector)
    trafficRules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                 lanelet2.traffic_rules.Participants.Vehicle)
    graph = lanelet2.routing.RoutingGraph(map, trafficRules)
    #map each point to the lane it belongs to
    j = 1
    change = pos_x[j] - pos_x[0]
    while(change==0):
        j += 1
        change = pos_x[j] - pos_x[0]

    lane_bevor = find_lane1(pos_x[0], pos_y[0], pos_x[j], pos_y[j],map)
    list1 = []
    list1.append(lane_bevor.id)

    for i in range(1,len(pos_x)):
        k = i
        change = pos_x[i] - pos_x[k]
        while(change==0):
            k = k - 1
            change = pos_x[i] - pos_x[k]
        lane = find_lane2(pos_x[k], pos_y[k], pos_x[i], pos_y[i], lane_bevor, map, graph)
        list1.append(lane.id)
        lane_bevor = lane

    n = len(pos_x) - 2
    p = n + 1
    dis_re = pos_x[n] - pos_x[p]
    while(dis_re==0):
        n -= 1
        dis_re = pos_x[n] - pos_x[p]
    lane_bevor_re = find_lane1(pos_x[p], pos_y[p], pos_x[n], pos_y[n], map)

    list2 = []
    list2.append(lane_bevor_re.id)

    for i in range(1, len(pos_x)):
        j = len(pos_x) - 1 - i
        k = j
        change = pos_x[j] - pos_x[k]
        while(change==0):
            k = k + 1
            change = pos_x[j] - pos_x[k]
        lane_re = find_lane2(pos_x[k], pos_y[k], pos_x[j], pos_y[j], lane_bevor_re, map, graph)
        list2.append(lane_re.id)
        lane_bevor_re = lane_re
    list2.reverse()

    for i in range(len(list1)):
        if list1[i] != list2[i]:
            if i < (len(list1) / 2):
                list1[i] = list2[i]

    kru = []
    dis = []
    fai = []
    for i in range(len(list1)):

        lanelet_layer = map.laneletLayer[list1[i]]
        x = pos_x[i]
        y = pos_y[i]
        kru_, dis_, fai_ = Krummung_rechnen(lanelet_layer, x, y)
        kru.append(kru_)
        dis.append(dis_)
        fai.append(fai_)

    return kru, dis, fai

fp1 = 'allAmsterdamerRingV2.pickle'
with open(fp1, 'rb') as fp:
    data = pickle.load(fp)

curve = data[0]
curve_data = curve[1]
pos_x = curve_data['pos_x']
pos_y = curve_data['pos_y']
plt.scatter(pos_x,pos_y)
plt.show()
index,dis,fai = findlane_index(pos_x,pos_y)
print(index)
print(dis)
print(fai)