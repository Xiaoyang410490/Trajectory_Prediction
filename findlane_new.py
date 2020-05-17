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

    for i in range(len(lanelets)):
        let = lanelets[i]
        le = let[1]
        print(le.id)
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

fp1 = 'allAmsterdamerRing.pickle'
with open(fp1, 'rb') as fp:
    data = pickle.load(fp)

curve = data[0]
curve_data = curve[1]
pos_x = curve_data['pos_x']
pos_y = curve_data['pos_y']
plt.scatter(pos_x, pos_y)
plt.show()

projector = UtmProjector(lanelet2.io.Origin(50.76599713889, 6.06099834167))
path = file_path
map = lanelet2.io.load(path, projector)
trafficRules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                 lanelet2.traffic_rules.Participants.Vehicle)
graph = lanelet2.routing.RoutingGraph(map, trafficRules)

print(pos_x[0])
print(pos_x[1])
print(pos_y[0])
print(pos_y[1])
lane = find_lane1(pos_x[0],pos_y[0],pos_x[1],pos_y[1],map)
print(lane)
