import lanelet2
import math
from lanelet2.core import BasicPoint2d
from lanelet2.projection import UtmProjector
file_path = "Amsterdamer_Intersection_Lanelet.osm"

def find_lane1(xf,yf,x,y,map):

    plocal = BasicPoint2d(x, y)
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
              Stringline_left = lane_index[1].leftBound
              vecx = Stringline_left[-1].x - Stringline_left[0].x
              vecy = Stringline_left[-1].y - Stringline_left[0].y
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
                  Stringline_left = lane_index[1].leftBound
                  vecx = Stringline_left[-1].x - Stringline_left[0].x
                  vecy = Stringline_left[-1].y - Stringline_left[0].y
                  cos_t = (vecx * vx + vecy * vy) / (math.sqrt((vecx ** 2 + vecy ** 2) * (vx ** 2 + vy ** 2)))
                  if cos_t > cos_max:
                     cos_max = cos_t
                     lan = lane_index[1]

    return lan


def Krummung_rechnen(lane):

    Stringline_right = lane.rightBound

    p1_x =  Stringline_right[0].x
    p1_y =  Stringline_right[0].y
    p2_x =  Stringline_right[1].x
    p2_y =  Stringline_right[1].y
    p3_x =  Stringline_right[-1].x
    p3_y =  Stringline_right[-1].y

    if len(Stringline_right) <= 2:
        kappa = 0

    elif len(Stringline_right) ==3:
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
    else:
        p4_x = Stringline_right[-2].x
        p4_y = Stringline_right[-2].y

        k1 = (p2_y - p1_y) / (p2_x - p1_x)
        k1_1 = -1 / k1
        x1 = (p2_x + p1_x) / 2
        y1 = (p2_y + p1_y) / 2

        k2 = (p3_y - p4_y) / (p3_x - p4_x)
        k2_1 = -1 / k2
        x2 = (p3_x + p4_x) / 2
        y2 = (p3_y + p4_y) / 2

        if (k1_1 - k2_1) <0.0001:
            kappa = 0
        else:
            x_1 = (k1_1 * x1 - k2_1 * x2 - y1 + y2) / (k1_1 - k2_1)
            y_1 = (-k2_1 * k1_1 * x1 + y1 * k2_1 + k1_1 * k2_1 * x2 - k1_1 * y2) / (k2_1 - k1_1)
            r1 = math.sqrt((x_1 - x1) ** 2 + (y_1 - y1) ** 2)
            r2 = math.sqrt((x_1 - x2) ** 2 + (y_1 - y2) ** 2)
            kappa = 2/(r1+r2)

    return kappa


def findlane_index(pos_x,pos_y):
    projector = UtmProjector(lanelet2.io.Origin(50.76599713889, 6.06099834167))
    path = file_path
    map = lanelet2.io.load(path, projector)
    trafficRules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                 lanelet2.traffic_rules.Participants.Vehicle)
    graph = lanelet2.routing.RoutingGraph(map, trafficRules)

    pos_xre = pos_x[::-1]
    pos_yre = pos_y[::-1]

    j = 1
    change = pos_x[j] - pos_x[0]
    while(change ==0):
        j += 1
        change = pos_x[j] - pos_x[0]
    lane_bevor = find_lane1(pos_x[0], pos_y[0], pos_x[j], pos_y[j],map)
    list1 = []


    for i in range(1, len(pos_x)):
        k = i
        change = pos_x[i] - pos_x[k]
        while(change==0):
            k = k -1
            change = pos_x[i] - pos_x[k]
        lane = find_lane2(pos_x[k], pos_y[k], pos_x[i], pos_y[i], lane_bevor,map,graph)
        list1.append(lane)
        lane_bevor = lane

    j = 1
    dis_re = pos_xre[j] - pos_xre[0]
    while(dis_re==0):
        j += 1
        dis_re = pos_xre[j] - pos_xre[0]
    lane_bevor_re = find_lane1(pos_xre[0],pos_yre[0], pos_xre[j], pos_yre[j],map)
    list2 = []
    list2.append(lane_bevor_re)

    for i in range(1, len(pos_xre)):
        k = i
        change = pos_xre[i] - pos_xre[k]
        while(change==0):
            k = k - 1
            change = pos_xre[i] - pos_xre[k]
        lane = find_lane2(pos_xre[k], pos_yre[k], pos_xre[i], pos_yre[i], lane_bevor_re,map,graph)
        list2.append(lane)
        lane_bevor_re = lane

    list2 = list2[::-1]

    index_list2 = []
    for i in range(len(list2)):
        id = list2[i].id
        index_list2.append(id)

    index_list1 = []
    for i in range(len(list1)):
        id = list1[i].id
        index_list1.append(id)

    for i in range(len(list1)):
        if index_list1[i] != index_list2[i]:
            if i < len(list1) / 2:
                index_list1[i] = index_list2[i]
    kru = []
    for i in range(len(index_list1)):
        lanelet_layer= map.laneletLayer[index_list1[i]]
        kru.append(Krummung_rechnen(lanelet_layer))

    return kru

