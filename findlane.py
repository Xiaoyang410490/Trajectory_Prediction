from lanegeometry import *
import pickle

#get the lane of initial point
def find_lane1(xf,yf,x,y,map,mapgeometry):
    #Firstly the closest lanelets will be collected.
    plocal = BasicPoint2d(x, y)
    lanelets = lanelet2.geometry.findNearest(map.laneletLayer, plocal, 1)

    if len(lanelets) == 1:

        # there is only one lane,especially in the area before the intersection, that lane is the one we need
        lane = lanelets[0]
        #For every lane in "lanelets", 0 is the distance between point and lane, 1 is the lanelet
        lan = lane[1]

    else:

        #If there is more than one lanelets, the lanelets having 0 distance have priority
        Lane = []
        for i in range(len(lanelets)):
            lane_index = lanelets[i]
            if lane_index[0] < 0.1:
                Lane.append(lane_index)

        lane_init = lanelets[0]
        amin = lane_init[0]   #smallest distance
        lan = lane_init[1]   #lanelet

        # If this point belongs to no lanelet, then the closest one will be chosen
        if len(Lane) == 0:
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
                center_2d = lanelet2.geometry.to2D(centerline)
                arc_coord = lanelet2.geometry.toArcCoordinates(center_2d, BasicPoint2d(x, y))
                length_along_center = arc_coord.length

                (x_spl, y_spl) = mapgeometry.centerline_interpolations[lane_index[1].id]
                x_p = x_spl(length_along_center, 1)
                y_p = y_spl(length_along_center, 1)
                vecx = x_p
                vecy = y_p
                vx = x - xf
                vy = y - yf

                cos_t = (vecx * vx + vecy * vy) / (math.sqrt((vecx ** 2 + vecy ** 2) * (vx ** 2 + vy ** 2)))
                if cos_t > cos_max:
                    cos_max = cos_t
                    lan = lane_index[1]

    ind = lan.id
    return ind


def find_lane2(xf,yf,x,y,map,lane_bevor,graph,kind,mapgeometry):

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
        # distance from the point to the lanelet
        amin = lane_init[0]
        # the lanelet
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
               lanelet0 = map.laneletLayer[lane_bevor]
               route = graph.getRoute(lanelet, lanelet0)

               #The lane must have succeeding relationship to the lane_bevor
               if route != None:

                   centerline = lane_index[1].centerline
                   center_2d = lanelet2.geometry.to2D(centerline)
                   arc_coord = lanelet2.geometry.toArcCoordinates(center_2d, BasicPoint2d(x, y))
                   length_along_center = arc_coord.length

                   (x_spl, y_spl) = mapgeometry.centerline_interpolations[lane_index[1].id]
                   x_p = x_spl(length_along_center, 1)
                   y_p = y_spl(length_along_center, 1)
                   vecx = x_p
                   vecy = y_p

                   vx = x - xf
                   vy = y - yf
                   #The angle between the direction of lanelet to the heading of vehicle
                   cos_t = (vecx * vx + vecy * vy) / (math.sqrt((vecx ** 2 + vecy ** 2) * (vx ** 2 + vy ** 2)))
                   if cos_t > cos_max:
                       cos_max = cos_t
                       lan = lane_index[1]

    ind = lan.id

    if kind == -1:
        if ind in [208860,188989]:
            ind = 188986
        elif ind in [188990,198781,198786,208603]:
            ind = 198738
        elif ind in [208553,208528]:
            ind = 208661
        elif ind in [208543]:
            ind = 208899
        else:
            ind = ind

    elif kind == 1:
        if ind in [198786,198738]:
            ind = 208603
        elif ind in [188986]:
            ind = 208641
        elif ind in [208860,188989]:
            ind = 208710
        elif ind in [208661]:
            ind = 188993
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
        else:
            ind = ind

    return ind


def future_parameter(future_len,mapgeometry,posx,posy,laneid,laneid_2,laneid_3,lane,lane2,lane3):

    turn_list = [188986, 198738, 208661, 208834, 188993, 208603, 208710, 208641]

    centerline = lane.centerline
    centerline2 = lane2.centerline
    center_2d = lanelet2.geometry.to2D(centerline)

    # computation of distance to the centerline, left is positive, right is negative
    arc_coord = lanelet2.geometry.toArcCoordinates(center_2d, BasicPoint2d(posx, posy))
    length_whole_center = length(centerline)
    length_whole_center_1 = length(centerline2)

    length_along_center = arc_coord.length
    length_along_center_1 = future_len - (length_whole_center - length_along_center)
    length_along_center_2 = future_len - (length_whole_center + length_whole_center_1 - length_along_center)

    if length_along_center_1 < 0:

        (x_spl, y_spl) = mapgeometry.centerline_interpolations[laneid]

        if laneid in turn_list:
            kappa_f = mapgeometry.calculateSplineCurvature(x_spl, y_spl, length_along_center + future_len)
        else:
            kappa_f = 0.0

        ori_f = mapgeometry.calculateSplineOrientation(x_spl, y_spl, length_along_center + future_len)
        rul_f = rule_info(laneid)

    elif length_along_center_1 > 0 and length_along_center_2 < 0:

        (x_spl_1, y_spl_1) = mapgeometry.centerline_interpolations[laneid_2]

        if laneid_2 in turn_list:
            kappa_f = mapgeometry.calculateSplineCurvature(x_spl_1, y_spl_1, length_along_center_1)
        else:
            kappa_f = 0.0

        ori_f = mapgeometry.calculateSplineOrientation(x_spl_1, y_spl_1, length_along_center_1)
        rul_f = rule_info(laneid_2)

    else:

        (x_spl_2, y_spl_2) = mapgeometry.centerline_interpolations[laneid_3]

        if laneid_3 in turn_list:
            kappa_f = mapgeometry.calculateSplineCurvature(x_spl_2, y_spl_2, length_along_center_2)
        else:
            kappa_f = 0.0

        ori_f = mapgeometry.calculateSplineOrientation(x_spl_2, y_spl_2, length_along_center_2)
        rul_f = rule_info(laneid_3)

    return kappa_f,ori_f,rul_f


def rule_info(lane):

    right_turn = [188986,198738,208661,208834]
    left_turn = [188993,208603,208710,208641]
    right_straight = [198702,208553,188998,208809,188984]
    left_straight = [188992,198781,208636,208705]
    #Other lanes are just lanes where vehicles can only go straight

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


def findlane_index(pos_x,pos_y,kind,num):

    # Use the projector to get the map

    file_path = "Amsterdamer_Intersection_Lanelet.osm"
    #initial the mapgeomerty class with the file_path and origin point for the projector
    mapgeometry = MapGeometry(file_path, 50.76599713889, 6.06099834167)
    map = mapgeometry.lanelet_map
    graph = mapgeometry.graph

    #the lanelet of the initial point
    lane_bevor = find_lane1(pos_x[0], pos_y[0], pos_x[num], pos_y[num], map, mapgeometry)

    list1 = []
    list0 = []
    list2 = []
    list3 = []
    list1.append(lane_bevor)
    list0.append(lane_bevor)

    for i in range(num + 1, len(pos_x)):

        k = i
        change = pos_x[i] - pos_x[k]

        while( change < 1):

            k = k - 1
            change = math.sqrt((pos_x[i] - pos_x[k])**2 + (pos_y[i]-pos_y[k])**2)

        laneid = find_lane2(pos_x[k],pos_y[k],pos_x[i],pos_y[i],map,lane_bevor,graph,kind,mapgeometry)

        list1.append(laneid)
        if laneid not in list0:
            list0.append(laneid)

        lane_bevor = laneid

    wid = []
    rul = []
    kru = []
    ori = []
    rul5 = []
    kru5 = []
    ori5 = []
    rul10 = []
    kru10 = []
    ori10 = []
    rul15 = []
    kru15 = []
    ori15 = []

    for i in range(len(list1)):

        id_1 = list1[i]

        #get the lanelet next to the current lanelet
        k = list0.index(id_1)
        if k > 0:
            id_0 = list0[k - 1]
        else:
            id_0 = list0[0]

        if k < (len(list0) - 1):
            id_2 = list0[k + 1]
        else:
            id_2 = list0[k]

        if k < (len(list0) - 2):
            id_3 = list0[k + 2]
        else:
            id_3 = list0[k]

        #we need to add some patch for some wrong cuvre
        #specific example curve 95
        if id_0 == 208648  and id_1 == 198738 and id_2 == 198702 and kind == -1:
            list1[i] = 198702
            id_1 = 198702

        # specific example curve 69,223
        if (208710 in list0) and ((188990 in list0) or (188991 in list0)) and (208705 in list0) and kind == 1:
            if (188990 in list0):

                i1 = list0.index(208705)
                i2 = list0.index(208710)
                i3 = list0.index(188990)
                ii = list0.index(id_1)
                if i1 < i2 and i2 < i3:
                    if ii > i1 and ii < i3:
                        id_1 = 208710
                        id_2 = 188990
                        id_3 = 188991
            else:

                i1 = list0.index(208705)
                i2 = list0.index(208710)
                i3 = list0.index(188991)
                ii = list0.index(id_1)
                if i1 < i2 and i2 < i3:
                    if ii > i1 and ii < i3:
                        id_1 = 208710
                        id_2 = 188991
                        id_3 = 188998

        #specifc example curve 133
        if (208636 in list0) and (208641 in list0) and (208648 in list0) and kind == 1:

            i1 = list0.index(208636)
            i2 = list0.index(208641)
            i3 = list0.index(208648)
            ii = list0.index(id_1)

            if i1<i2 and i2<i3:
                if ii>i1 and ii<i3:

                    id_1 = 208641
                    id_2 = 208648
                    id_3 = 188988

        #specific example curve 190,192,107
        if (208603 in list0) and (208535 in list0) and (208543 in list0) and kind == 1:

            i1 = list0.index(208603)
            i2 = list0.index(208535)
            i3 = list0.index(208543)
            ii = list0.index(id_1)

            if i1<i2 and i2<i3:
                if ii>i1 and ii<i3:

                    id_1 = 208603
                    id_2 = 208535
                    id_3 = 208543

        list1[i] = id_1
        list2.append(id_2)
        list3.append(id_3)
        lane = map.laneletLayer[id_1]
        lane2 = map.laneletLayer[id_2]
        lane3 = map.laneletLayer[id_3]

        left_line = lane.leftBound
        right_line = lane.rightBound
        width = lanelet2.geometry.distance(left_line, right_line)

        posx = pos_x[i+num]
        posy = pos_y[i+num]

        kappa_0, ori_0, rul_0 = future_parameter(0, mapgeometry, posx, posy, id_1, id_2, id_3, lane, lane2, lane3)
        kappa_5, ori_5, rul_5 = future_parameter(5, mapgeometry, posx, posy, id_1, id_2, id_3, lane, lane2, lane3)
        kappa_10, ori_10, rul_10 = future_parameter(10, mapgeometry, posx, posy, id_1, id_2, id_3, lane, lane2,lane3)
        kappa_15, ori_15, rul_15 = future_parameter(15, mapgeometry, posx, posy, id_1, id_2, id_3, lane, lane2, lane3)

        wid.append(width)
        kru.append(kappa_0)
        ori.append(ori_0)
        rul.append(rul_0)
        kru5.append(kappa_5)
        ori5.append(ori_5)
        rul5.append(rul_5)
        kru10.append(kappa_10)
        ori10.append(ori_10)
        rul10.append(rul_10)
        kru15.append(kappa_15)
        ori15.append(ori_15)
        rul15.append(rul_15)

    return list0,list1,list2,list3,wid,kru,ori,rul,kru5,ori5,rul5,kru10,ori10,rul10,kru15,ori15,rul15

