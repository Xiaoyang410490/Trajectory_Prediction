import lanelet2
import math
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

    return lan


def find_lane2(xf,yf,x,y,map,lane_bevor,graph):

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
               toLanelet2 = map.laneletLayer[lane_bevor.id]
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

    return lan


def Krummung_rechnen(lane,posx,posy):

    centerline = lane.centerline
    rightline = lane.rightBound

    #Two closest points will be chosen
    #Firstly the closest point will be chosen
    short_dis = math.sqrt((posx - centerline[1].x) ** 2 + (posy - centerline[1].y) ** 2)
    short_ind = 1

    for i in range(2, len(centerline) - 1):
        distance = math.sqrt((posx - centerline[i].x) ** 2 + (posy - centerline[i].y) ** 2)
        if distance < short_dis:
            short_dis = distance
            short_ind = i
    #After choosing the closest point, the second closest point will be chosen
    if short_ind == 0:
        short_ind_1 = 1
    elif short_ind == len(centerline) - 1:
        short_ind_1 = len(centerline) - 2
    else:
        ind_1 = short_ind - 1
        ind_2 = short_ind + 1
        dis_1 = math.sqrt((centerline[ind_1].x - posx) ** 2 + (centerline[ind_1].y - posy) ** 2)
        dis_2 = math.sqrt((centerline[ind_2].x - posx) ** 2 + (centerline[ind_2].y - posy) ** 2)
        if dis_1 < dis_2:
            short_ind_1 = ind_1
        else:
            short_ind_1 = ind_2

    #The curvature of centerline will be computed
    if len(rightline) <= 2:
        kappa = 0
    else:
        # distinguish between right and left
        # Normal vector of initial curve
        v1_x = rightline[1].y - rightline[0].y
        v1_y = rightline[0].x - rightline[1].x
        v2_x = rightline[-1].x - rightline[-2].x
        v2_y = rightline[-1].y - rightline[-2].y
        # The steering angle of left-turn and right-turn ought to be different
        if (v1_x * v2_x + v1_y * v2_y) > 0:
            #right-turn curve
            vorzeichen = -1
        else:
            vorzeichen = 1

        p1_x = centerline[short_ind - 1].x
        p1_y = centerline[short_ind - 1].y
        p2_x = centerline[short_ind].x
        p2_y = centerline[short_ind].y
        p3_x = centerline[short_ind + 1].x
        p3_y = centerline[short_ind + 1].y

        k1 = (p2_y - p1_y) / (p2_x - p1_x)
        k1_1 = -1 / k1
        x1 = (p2_x + p1_x) / 2
        y1 = (p2_y + p1_y) / 2

        k2 = (p3_y - p2_y) / (p3_x - p2_x)
        k2_1 = -1 / k2
        x2 = (p3_x + p2_x) / 2
        y2 = (p3_y + p2_y) / 2

        if (k1_1 - k2_1) < 0.0001:
            kappa = 0
        else:
            x_1 = (k1_1 * x1 - k2_1 * x2 - y1 + y2) / (k1_1 - k2_1)
            y_1 = (-k2_1 * k1_1 * x1 + y1 * k2_1 + k1_1 * k2_1 * x2 - k1_1 * y2) / (k2_1 - k1_1)
            r = math.sqrt((x_1 - p2_x) ** 2 + (y_1 - p2_y) ** 2)
            kappa = 1 / r
            kappa = kappa*vorzeichen

    # computation of distance to centerline
    # choose the two-closest points on the curve
    l_x1 = centerline[short_ind].x
    l_y1 = centerline[short_ind].y
    l_x2 = centerline[short_ind_1].x
    l_y2 = centerline[short_ind_1].y
    # to see if the point on the right side of the centerline
    A = l_y2 - l_y1
    B = l_x1 - l_x2
    C = l_x2 * l_y1 - l_x1 * l_y2
    Tmp = (l_y1 - l_y2) * posx + (l_x2 - l_x1) * posy + l_x1 * l_y2 - l_x2 * l_y1
    dis_to_center = abs(A * posx + B * posy + C) / math.sqrt(A ** 2 + B ** 2)
    if Tmp < 0:
        # if it's on the right side, then distance is negative.
        dis_to_center = -dis_to_center

    # computation of angle deviation
    x_zeichen = l_x2 - l_x1
    zeichen = (l_y2 - l_y1) / (l_x2 - l_x1)
    if x_zeichen > 0:
        fai = math.atan(zeichen)
    else:
        fai = math.atan(zeichen) + math.pi

    return kappa, dis_to_center,fai


def findlane_index(pos_x,pos_y):
    #Use the projector to get the map
    projector = UtmProjector(lanelet2.io.Origin(50.76599713889, 6.06099834167))
    path = file_path
    map = lanelet2.io.load(path, projector)
    #Use routing to get the graph
    trafficRules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                 lanelet2.traffic_rules.Participants.Vehicle)
    graph = lanelet2.routing.RoutingGraph(map,trafficRules)
    #map each point to the lane it belongs to
    j = 1
    change = pos_x[j] - pos_x[0]
    while(change ==0):
        j += 1
        change = pos_x[j] - pos_x[0]

    lane_bevor = find_lane1(pos_x[0],pos_y[0],pos_x[j],pos_y[j],map)
    list1 = []
    list1.append(lane_bevor.id)

    for i in range(1, len(pos_x)):
        k = i
        change = pos_x[i] - pos_x[k]
        while(change==0):
            k = k - 1
            change = pos_x[i] - pos_x[k]
        lane = find_lane2(pos_x[k],pos_y[k],pos_x[i],pos_y[i],map,lane_bevor,graph)
        list1.append(lane.id)
        lane_bevor = lane

    #Reverse the order of lanelets
    n = len(pos_x) - 2
    p = n + 1
    dis_re = pos_x[n] - pos_x[p]
    while (dis_re == 0):
        n -= 1
        dis_re = pos_x[n] - pos_x[p]
    lane_bevor_re = find_lane1(pos_x[p], pos_y[p], pos_x[n], pos_y[n], map)

    list2 = []
    list2.append(lane_bevor_re.id)

    for i in range(1, len(pos_x)):
        j = len(pos_x) - 1 - i
        k = j
        change = pos_x[j] - pos_x[k]
        while (change == 0):
            k = k + 1
            change = pos_x[j] - pos_x[k]
        lane_re = find_lane2(pos_x[k],pos_y[k],pos_x[j],pos_y[j],map,lane_bevor_re,graph)
        list2.append(lane_re.id)
        lane_bevor_re = lane_re

    list2.reverse()
    #To solve the dilemma1, The index in lanelet1 at the first half will be replaced by that in list2
    for i in range(len(list1)):
        if list1[i]!=list2[i] and i<(len(list1)/2) :
                list1[i] = list2[i]

    kru = []
    dis = []
    fai = []
    for i in range(len(list1)):
        lanelet_layer= map.laneletLayer[list1[i]]
        kru_, dis_, fai_ = Krummung_rechnen(lanelet_layer,pos_x[i],pos_y[i])
        kru.append(kru_)
        dis.append(dis_)
        fai.append(fai_)

    return kru,dis,fai
