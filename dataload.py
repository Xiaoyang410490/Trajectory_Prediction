import lanelet2
import pickle
import math
import numpy as np
from lanelet2.core import Lanelet,LineString3d,Point3d,LaneletMap,getId,BasicPoint2d
from lanelet2.projection import UtmProjector
from findlanelet import *
from akmanmodel import *
file_path = "Amsterdamer_Intersection_Lanelet.osm"

class Dataload():
    def __init__(self,batch_size = 50,seq_length = 150,map_path = "Amsterdamer_Intersection_Lanelet.osm",pos_path = "allAmsterdamerRingV2.pickle"):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.map_path = map_path
        self.pos_path = pos_path

    def load_position(self):
        with open(self.pos_path, 'rb') as fp:
            data = pickle.load(fp)
        pos_info =[]
        for i in range(len(data)):
            kurve_info = []
            curve = data[i]
            curve_data = curve[1]
            x = np.array(curve_data['pos_x'])
            y = np.array(curve_data['pos_y'])
            for i in range(len(y)):
                pos_x = x[i]
                pos_y = y[i]
                kurve_info.append([pos_x,pos_y])
            pos_info.append(kurve_info)

        return pos_info

    def load_geometry(self):
        projector = UtmProjector(lanelet2.io.Origin(50.76599713889, 6.06099834167))
        path = self.map_path
        map = lanelet2.io.load(path, projector)
        return map








if __name__ == '__main__':
    pos = Dataload().load_position()
    print(pos)


