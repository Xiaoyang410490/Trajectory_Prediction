import lanelet2
import pickle
import math
import numpy as np
from lanelet2.core import Lanelet,LineString3d,Point3d,LaneletMap,getId,BasicPoint2d
from lanelet2.projection import UtmProjector
file_path = "Amsterdamer_Intersection_Lanelet.osm"

def load_data_files(filepath,tra_num):
    with open(filepath,'rb') as fp:
        data = pickle.load(fp)
    curve_data = np.array(data[tra_num])
    cd = np.array(curve_data[1])
    return cd


def find_lane1(xf,yf,x,y):
    projector = UtmProjector(lanelet2.io.Origin(50.76599713889, 6.06099834167))
    path = file_path
    map = lanelet2.io.load(path, projector)

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
        lan = lane_init

        if len(Lane)==0:
            for i in range(len(lanelets)):
                lane_index = lanelets[i]
                dis = lane_index[0]
                if dis < amin:
                    amin = dis
                    lan = lane_index[1]

        else:
            for i in range(len(Lane)):
              lane_index = Lane[i]
              Stringline_left = lane_index[1].leftBound
              vecx = Stringline_left[-1].x - Stringline_left[0].x
              vecy = Stringline_left[-1].y - Stringline_left[0].y
              cos_t = (vecx * vx + vecy * vy) / (math.sqrt(vecx ** 2 + vecy ** 2) * math.sqrt(vx ** 2 + vy ** 2))
              if cos_t > cos_max:
                 cos_max = cos_t
                 lan = lane_index[1]
    return lan


def find_lane2(xf,yf,x,y,lane_bevor):

    projector = UtmProjector(lanelet2.io.Origin(50.76599713889, 6.06099834167))
    path = file_path
    map = lanelet2.io.load(path, projector)
    trafficRules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                 lanelet2.traffic_rules.Participants.Vehicle)

    graph = lanelet2.routing.RoutingGraph(map, trafficRules)
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
            for i in range(len(lanelets)):
                lane_index = lanelets[i]
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
                  cos_t = (vecx * vx + vecy * vy) / (math.sqrt(vecx ** 2 + vecy ** 2) * math.sqrt(vx ** 2 + vy ** 2))
                  if cos_t > cos_max:
                     cos_max = cos_t
                     lan = lane_index[1]

    return lan




def Krumung_rechnen(laneLET):
    string_a = laneLET.leftBound
    string_b = laneLET.rightBound

    k1 = (string_a[-1].y - string_a[-2].y) / (string_a[-1].x - string_a[-2].x)
    k1_1 = -1 / k1
    x1 = (string_a[-1].x + string_a[-2].x) / 2
    y1 = (string_a[-1].y + string_a[-2].y) / 2

    k2 = (string_a[1].y - string_a[0].y) / (string_a[1].x - string_a[0].x)
    k2_1 = -1 / k2
    x2 = (string_a[1].x + string_a[0].x) / 2
    y2 = (string_a[1].y + string_a[0].y) / 2

    if (k2_1 - k1_1) == 0:
        r1 = 1000
    else:
        x_1 = (k1_1 * x1 - k2_1 * x2 - y1 + y2) / (k1_1 - k2_1)
        y_1 = (-k2_1 * k1_1 * x1 + y1 * k2_1 + k1_1 * k2_1 * x2 - k1_1 * y2) / (k2_1 - k1_1)
        r1 = (math.sqrt((x_1 - x1) ** 2 + (y_1 - y1) ** 2) + math.sqrt((x_1 - x2) ** 2 + (y_1 - y2) ** 2)) / 2

    # krummung right Bound
    k1 = (string_b[-1].y - string_b[-2].y) / (string_b[-1].x - string_b[-2].x)
    k1_1 = -1 / k1
    x1 = (string_b[-1].x + string_b[-2].x) / 2
    y1 = (string_b[-1].y + string_b[-2].y) / 2

    k2 = (string_b[1].y - string_b[0].y) / (string_b[1].x - string_b[0].x)
    k2_1 = -1 / k2
    x2 = (string_b[1].x + string_b[0].x) / 2
    y2 = (string_b[1].y + string_b[0].y) / 2

    if (k2_1 - k1_1) == 0:
        r2 = 1000
    else:
        x_1 = (k1_1 * x1 - k2_1 * x2 - y1 + y2) / (k1_1 - k2_1)
        y_1 = (-k2_1 * k1_1 * x1 + y1 * k2_1 + k1_1 * k2_1 * x2 - k1_1 * y2) / (k2_1 - k1_1)
        r2 = (math.sqrt((x_1 - x1) ** 2 + (y_1 - y1) ** 2) + math.sqrt((x_1 - x2) ** 2 + (y_1 - y2) ** 2)) / 2

    if r1 == 1000 and r2 == 1000:
        kappa = 0
    elif r1 == 1000 and r2 != 1000:
        kappa = 1 / r2
    elif r1 != 1000 and r2 == 1000:
        kappa = 1 / r1
    else:
        R = (r1 + r2) / 2
        kappa = 1 / R

    return kappa


if __name__ == '__main__':

    cd = load_data_files('allAmsterdamerRingV2.pickle',1300)

    x = cd[:, 2]
    y = cd[:, 3]

    # in the recurrent neural network,this data will be pos_x[i],pos_y[i]
    # because of the noise, the first two data may choose pos_x[i-3],pos_y[i-3]
    lane_bevor = find_lane1(x[0],y[0],x[1],y[1])

    for i in range(1, len(x)):
        lane = find_lane2(x[i - 1], y[i - 1], x[i], y[i],lane_bevor)
        print(lane)
        lane_bevor = lane













