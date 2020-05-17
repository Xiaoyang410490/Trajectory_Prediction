import lanelet2
import tempfile
import os
import numpy as np
import matplotlib.pyplot as plt
from lanelet2.core import AttributeMap,TrafficLight,Lanelet,LineString3d,Point2d,Point3d,LaneletMap,getId,BoundingBox2d,BasicPoint2d
from lanelet2.projection import UtmProjector
file_path = "Amsterdamer_Intersection_Lanelet.osm"

projector = UtmProjector(lanelet2.io.Origin(50.76599713889, 6.06099834167))
path = file_path
map = lanelet2.io.load(path, projector)
lane = map.laneletLayer[198738]
left = lane.leftBound
right = lane.rightBound
center = lane.centerline

x = [left[0].x,left[1].x]
y = [left[0].y,left[1].y]
x1 = [right[0].x,right[1].x]
y1 = [right[0].y,right[1].y]


plt.scatter(x,y)
plt.scatter(x1,y1)

c_x = []
c_y = []
for i in range(len(center)):
    c_x.append(center[i].x)
    c_y.append(center[i].y)

plt.scatter(c_x,c_y)
plt.show()


