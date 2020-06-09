from lanegeometry import *
import pickle
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


def geo_rechnen(lane,lane2,laneid,laneid_2,posx,posy,mapgeometry):

    centerline = lane.centerline
    centerline2 = lane2.centerline
    center_2d =  lanelet2.geometry.to2D(centerline)
    center_2d2 = lanelet2.geometry.to2D(centerline2)
    #computation of the width of lanelet
    left_line = lane.leftBound
    right_line = lane.rightBound
    width = lanelet2.geometry.distance(left_line, right_line)

    # computation of distance to the centerline, left is positive, right is negative
    arc_coord = lanelet2.geometry.toArcCoordinates(center_2d, BasicPoint2d(posx, posy))
    arc_coord2 = lanelet2.geometry.toArcCoordinates(center_2d2, BasicPoint2d(posx, posy))
    length_along_center = arc_coord.length
    length_whole_center = length(centerline)

    (x_spl, y_spl) = mapgeometry.centerline_interpolations[laneid]
    (x_spl_1, y_spl_1) = mapgeometry.centerline_interpolations[laneid_2]

    #the compuation of curvature
    turn_list = [188986, 198738, 208661, 208834, 188993, 208603, 208710, 208641]
    if laneid in turn_list:
        kappa = mapgeometry.calculateSplineCurvature(x_spl, y_spl, length_along_center)
    else:
        kappa = 0

    #the computation of orientation of centerline
    orien = mapgeometry.calculateSplineOrientation(x_spl,y_spl, length_along_center)
    rul = rule_info(laneid)

    #the compuattion of curavture and orientation after 1 meter
    if (length_along_center+1)<length_whole_center:

        if laneid in turn_list:
            kappa_1 = mapgeometry.calculateSplineCurvature(x_spl, y_spl, length_along_center+1)
        else:
            kappa_1 = 0

        ori_1 = mapgeometry.calculateSplineOrientation(x_spl, y_spl, length_along_center+1)
        rul_1 = rule_info(laneid)

        interp_point1 = lanelet2.geometry.interpolatedPointAtDistance(center_2d, length_along_center + 1)
        fai_1 = np.arctan2(interp_point1.y - posy,interp_point1.x - posx)


    else:
        length_along_center_1 = 1 - (length_whole_center - length_along_center)
        if laneid_2 in turn_list:
            kappa_1 = mapgeometry.calculateSplineCurvature(x_spl_1, y_spl_1, length_along_center_1)
        else:
            kappa_1 = 0

        ori_1 = mapgeometry.calculateSplineOrientation(x_spl_1, y_spl_1, length_along_center_1)
        rul_1 = rule_info(laneid_2)

        interp_point1 = lanelet2.geometry.interpolatedPointAtDistance(center_2d2, length_along_center_1)
        fai_1 = np.arctan2(interp_point1.y - posy, interp_point1.x - posx)


    #the computation of curvature and orientation after 3 meter
    if (length_along_center + 3)<length_whole_center:

        if laneid in turn_list:
            kappa_3 = mapgeometry.calculateSplineCurvature(x_spl, y_spl, length_along_center+3)
        else:
            kappa_3 = 0

        ori_3 = mapgeometry.calculateSplineOrientation(x_spl, y_spl, length_along_center+3)
        rul_3 = rule_info(laneid)

        interp_point3 = lanelet2.geometry.interpolatedPointAtDistance(center_2d, length_along_center + 3)
        fai_3 = np.arctan2(interp_point3.y - posy, interp_point3.x - posx)

    else:
        length_along_center_3 = 3 - (length_whole_center - length_along_center)
        if laneid_2 in turn_list:
            kappa_3 = mapgeometry.calculateSplineCurvature(x_spl_1, y_spl_1, length_along_center_3)
        else:
            kappa_3 = 0

        ori_3 = mapgeometry.calculateSplineOrientation(x_spl_1, y_spl_1, length_along_center_3)
        rul_3 = rule_info(laneid_2)

        interp_point3 = lanelet2.geometry.interpolatedPointAtDistance(center_2d2, length_along_center_3)
        fai_3 = np.arctan2(interp_point3.y - posy, interp_point3.x - posx)


    # the computation of curvature and orientation after 3 meter
    if (length_along_center + 5) < length_whole_center:

        if laneid in turn_list:
            kappa_5 = mapgeometry.calculateSplineCurvature(x_spl, y_spl, length_along_center + 5)
        else:
            kappa_5 = 0

        ori_5 = mapgeometry.calculateSplineOrientation(x_spl, y_spl, length_along_center + 5)
        rul_5 = rule_info(laneid)

        interp_point5 = lanelet2.geometry.interpolatedPointAtDistance(center_2d, length_along_center + 5)
        fai_5 = np.arctan2(interp_point5.y - posy, interp_point5.x - posx)

    else:
        length_along_center_5 = 5 - (length_whole_center - length_along_center)
        if laneid_2 in turn_list:
            kappa_5 = mapgeometry.calculateSplineCurvature(x_spl_1, y_spl_1, length_along_center_5)
        else:
            kappa_5 = 0

        ori_5 = mapgeometry.calculateSplineOrientation(x_spl_1, y_spl_1, length_along_center_5)
        rul_5 = rule_info(laneid_2)

        interp_point5 = lanelet2.geometry.interpolatedPointAtDistance(center_2d2, length_along_center_5)
        fai_5 = np.arctan2(interp_point5.y - posy, interp_point5.x - posx)

    return width,kappa,orien,rul,kappa_1,ori_1,rul_1,fai_1,kappa_3,ori_3,rul_3,fai_3,kappa_5,ori_5,rul_5,fai_5


def rule_info(lane):

    right_turn = [188986,198738,208661,208834]
    left_turn = [188993,208603,208710,208641]
    right_straight = [198702,208553,188998,208809,188984]
    left_straight = [188984,198781,208636,208705]

    if lane in right_turn:
        rule = -1.0
    elif lane in left_turn:
        rule = 1.0
    elif lane in right_straight:
        rule = -0.5
    elif lane in left_straight:
        rule = 0.5
    else:
        rule = 0.0
    return rule


def findlane_index(pos_x,pos_y):

    # Use the projector to get the map
    projector = UtmProjector(lanelet2.io.Origin(50.76599713889, 6.06099834167))
    path = file_path
    map = lanelet2.io.load(path, projector)
    # Use routing to get the graph
    trafficRules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                 lanelet2.traffic_rules.Participants.Vehicle)
    graph = lanelet2.routing.RoutingGraph(map, trafficRules)
    mapgeometry = MapGeometry(file_path, 50.76599713889, 6.06099834167)

    #map each point to the lane it belongs to
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
    Tmp = (v1_x * v2_x + v1_y * v2_y) / math.sqrt((v1_x ** 2 + v1_y ** 2) * (v2_x ** 2 + v2_y ** 2))

    if Tmp > 0.45:
        # right-turn curve
        kind = -1
    elif Tmp < -0.4:
        # left-turn curve
        kind = 1
    else:
        kind = 0

    # map each point to the lane it belongs to
    j = 1
    change = pos_x[j] - pos_x[0]
    while (change == 0):
        j += 1
        change = pos_x[j] - pos_x[0]

    lane_bevor = find_lane1(pos_x[0], pos_y[0], pos_x[j], pos_y[j], map)
    list1 = []
    list2 = []
    list1.append(lane_bevor)
    list2.append(lane_bevor)

    for i in range(1, len(pos_x)):
        k = i
        change = pos_x[i] - pos_x[k]
        while(change==0):
            k = k - 1
            change = pos_x[i] - pos_x[k]
        laneid = find_lane2(pos_x[k],pos_y[k],pos_x[i],pos_y[i],map,lane_bevor,graph,kind)
        list1.append(laneid)
        if laneid not in list2:
            list2.append(laneid)
        lane_bevor = laneid

    wid = []
    rul = []
    kru = []
    ori = []
    rul1 = []
    kru1 = []
    ori1 = []
    fai1 = []
    rul3 = []
    kru3 = []
    ori3 = []
    fai3 = []
    rul5 = []
    kru5 = []
    ori5 = []
    fai5 = []

    list3 = []

    for i in range(len(list1)):

        lane = map.laneletLayer[list1[i]]

        #get the lanelet next to the current lanelet
        for k in range(len(list2)):
            if list2[k] == list1[i] and k != (len(list2)-1):
                id_2 = list2[k+1]
            elif list2[k] == list1[i] and k != (len(list2)-1):
                id_2 = list2[k]

        lane2 = map.laneletLayer[id_2]
        list3.append(id_2)

        width, kappa, orien, rul_, kappa_1, ori_1, rul_1, fai_1, kappa_3, ori_3, rul_3, fai_3, kappa_5, ori_5, rul_5, fai_5 = geo_rechnen(lane, lane2, list1[i], id_2, pos_x[i], pos_y[i],mapgeometry)

        wid.append(width)
        kru.append(kappa)
        ori.append(orien)
        rul.append(rul_)
        kru1.append(kappa_1)
        ori1.append(ori_1)
        rul1.append(rul_1)
        fai1.append(fai_1)
        kru3.append(kappa_3)
        ori3.append(ori_3)
        rul3.append(rul_3)
        fai3.append(fai_3)
        kru5.append(kappa_5)
        ori5.append(ori_5)
        rul5.append(rul_5)
        fai5.append(fai_5)

    return list1,list3,wid,kru,ori,rul,kru1,ori1,rul1,fai1,kru3,ori3,rul3,fai3,kru5,ori5,rul5,fai5

