from lanegeometry import *
import pickle
from scipy.signal import sosfiltfilt,butter

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
        lan = lane_init[1]   #lanelet

        # If this point belongs to no lanelet, then the closest one will be chosen
        if len(Lane) == 0:
            lan = lan

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


def find_lane2(xf,yf,x,y,map,lane_bevor,graph,mapgeometry,kind):

    plocal = BasicPoint2d(x, y)
    lanelets = lanelet2.geometry.findNearest(map.laneletLayer, plocal, 1)

    if len(lanelets) == 1:

        lane = lanelets[0]
        lan = lane[1]

    else:

        Lane = []
        for i in range(len(lanelets)):
            lane_index = lanelets[i]
            if lane_index[0] < 1:
                Lane.append(lane_index)

        lane_init = lanelets[0]
        lan = lane_init[1]

        if len(Lane)==0:
            lan = lan

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

    if kind==0:

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

    elif kind==-1:

        if ind == 188993:
            ind = 208661

    return ind

#this function is used for correcting the error during lane matching
def replace(id1,id2,id3,l1):

    listid1 = []
    listid2 = []
    real_list0 = []

    for l in range(len(l1)):
        if l1[l] == id1:
            listid1.append(l)
        if l1[l] == id2:
            listid2.append(l)

    replace_start = listid1[-1]
    replace_end = listid2[0]

    for i in range(replace_start + 1, replace_end):
        l1[i] = id3

    real_list0.append(l1[0])
    for i in range(len(l1)):
        if l1[i] not in real_list0:
            real_list0.append(l1[i])

    return real_list0,l1


def future_parameter(future_len,mapgeometry,posx,posy,laneid,laneid_2,laneid_3,lane,lane2,lane3):

    turn_list = [188986, 198738, 208661, 208834, 188993, 208603, 208710, 208641]

    map = mapgeometry.lanelet_map
    graph = mapgeometry.graph

    centerline1 = lane.centerline
    centerline2 = lane2.centerline
    centerline3 = lane3.centerline

    center_2d = lanelet2.geometry.to2D(centerline1)
    # computation of distance to the centerline, left is positive, right is negative
    arc_coord = lanelet2.geometry.toArcCoordinates(center_2d, BasicPoint2d(posx, posy))
    length_along_center = future_len + arc_coord.length
    length_along_center_1 = length_along_center - length(centerline1)
    length_along_center_2 = length_along_center_1 - length(centerline2)
    length_along_center_3 = length_along_center_2 - length(centerline3)

    if length_along_center_1 < 0:

        (x_spl, y_spl) = mapgeometry.centerline_interpolations[laneid]

        if laneid in turn_list:
            kappa_f = mapgeometry.calculateSplineCurvature(x_spl, y_spl, length_along_center)
        else:
            kappa_f = 0.0

        ori_f = mapgeometry.calculateSplineOrientation(x_spl, y_spl, length_along_center)
        rul_f = rule_info(laneid)

    elif length_along_center_1 > 0 and length_along_center_2 < 0:

        (x_spl_1, y_spl_1) = mapgeometry.centerline_interpolations[laneid_2]

        if laneid_2 in turn_list:
            kappa_f = mapgeometry.calculateSplineCurvature(x_spl_1, y_spl_1, length_along_center_1)
        else:
            kappa_f = 0.0

        ori_f = mapgeometry.calculateSplineOrientation(x_spl_1, y_spl_1, length_along_center_1)
        rul_f = rule_info(laneid_2)

    elif length_along_center_2 > 0 and length_along_center_3 < 0:

        (x_spl_2, y_spl_2) = mapgeometry.centerline_interpolations[laneid_3]

        if laneid_3 in turn_list:
            kappa_f = mapgeometry.calculateSplineCurvature(x_spl_2, y_spl_2, length_along_center_2)
        else:
            kappa_f = 0.0

        ori_f = mapgeometry.calculateSplineOrientation(x_spl_2, y_spl_2, length_along_center_2)
        rul_f = rule_info(laneid_3)

    else:
        initial_id = laneid_3
        initial_length = length_along_center_3
        length_along = initial_length
        while (initial_length) > 0:
            next_lanelet = graph.following(map.laneletLayer[initial_id])
            if len(next_lanelet)!=0:
                next_lane= next_lanelet[0]
                next_lane = map.laneletLayer[next_lane.id]
                next_centerline = next_lane.centerline
                length_along = initial_length
                initial_length = initial_length - length(next_centerline)
                initial_id = next_lane.id
            else:
                initial_length = -1

        (x_spl_3, y_spl_3) = mapgeometry.centerline_interpolations[initial_id]

        if initial_id in turn_list:
            kappa_f = mapgeometry.calculateSplineCurvature(x_spl_3, y_spl_3, length_along)
        else:
            kappa_f = 0.0

        ori_f = mapgeometry.calculateSplineOrientation(x_spl_3, y_spl_3, length_along)
        rul_f = rule_info(initial_id)

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

    center_x = 4.77498
    center_y = 6.78343

    #the lanelet of the initial point
    lane_bevor = find_lane1(pos_x[0], pos_y[0], pos_x[num], pos_y[num], map, mapgeometry)

    list0 = []
    list1 = []
    list2 = []
    list3 = []
    list0.append(lane_bevor)
    list1.append(lane_bevor)

    for i in range(num + 1, len(pos_x)):

        k = i
        change = pos_x[i] - pos_x[k]

        while(change < 1):

            k = k - 1
            change = math.sqrt((pos_x[i] - pos_x[k])**2 + (pos_y[i]-pos_y[k])**2)

        laneid = find_lane2(pos_x[k],pos_y[k],pos_x[i],pos_y[i],map,lane_bevor,graph,mapgeometry,kind)

        list1.append(laneid)
        if laneid not in list0:
            list0.append(laneid)

        lane_bevor = laneid

    #correct the error
    if kind == -1:

        if (208697 in list0) and (208548 in list0):
            list0, list1 = replace(208697,208548,208834,list1)
        elif (198702 in list0) and (188991 in list0):
            list0, list1 = replace(198702,188991,198738,list1)
        elif (188984 in list0) and (188987 in list0):
            list0, list1 = replace(188984,188987,188986,list1)
        elif (188998 in list0) and (188994 in list0):
            list0, list1 = replace(188998,188994,208661,list1)
        else:
            list0 = list0
            list1 = list1

    elif kind == 1:

        if (208705 in list0) and (188990 in list0):
            list0, list1 = replace(208705, 188990, 208710, list1)
        elif (188990 not in list0) and (188991 in list0) and (208705 in list0):
            list0, list1 = replace(208705, 188991, 208710, list1)
        elif (188990 not in list0) and (208710 in list0) and (188991 in list0):
            list0, list1 = replace(208710, 188991, 208710, list1)
        elif (188992 in list0) and (188994 in list0):
            list0, list1 = replace(188992, 188994, 188993, list1)
        elif (198781 in list0) and (208535 in list0):
            list0, list1 = replace(198781, 208535, 208603, list1)
        elif (198781 not in list0) and (198786 in list0) and (208535 in list0):
            list0, list1 = replace(198786, 208535,208603, list1)
        elif (208636 in list0) and (208648 in list0):
            list0, list1 = replace(208636, 208648, 208641, list1)
        else:
            list0 = list0
            list1 = list1

    else:
        list0 = list0
        list1 = list1

    wid = []
    dic = []
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
        final_id = list0[-1]
        next_lanelet_1 = graph.following(map.laneletLayer[final_id])

        #get the index of the lane next to the current lane
        if k < (len(list0) - 1):
            id_2 = list0[k + 1]
        else:
            next_lanelet = graph.following(map.laneletLayer[id_1])
            if len(next_lanelet)!=0:
                next_lane = next_lanelet[0]
                id_2 = next_lane.id
            else:
                id_2 = id_1

        #get the index of the lane next to the lanelet id2
        if k < (len(list0) - 2):
            id_3 = list0[k + 2]

        elif k == (len(list0) - 2):

            if len(next_lanelet_1) != 0:
                next_lane_1 = next_lanelet_1[0]
                id_3 = next_lane_1.id
            else:
                id_3 = final_id

        else:
            if len(next_lanelet_1) != 0:
                next_lane_1 = next_lanelet_1[0]
                final_id_2=  next_lane_1.id
                next_lanelet_2 = graph.following(map.laneletLayer[final_id_2])
                if len(next_lanelet_2) != 0:
                    next_lane_2 = next_lanelet_2[0]
                    id_3 = next_lane_2.id
                else:
                    id_3 = final_id_2
            else:
                id_3 = final_id

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
        dis_to_center = math.sqrt((posx-center_x)**2+(posy-center_y)**2)
        dis_to_center = float(dis_to_center)

        kappa_0, ori_0, rul_0 = future_parameter(0, mapgeometry, posx, posy, id_1, id_2, id_3, lane,lane2,lane3)
        kappa_5, ori_5, rul_5 = future_parameter(5, mapgeometry, posx, posy, id_1, id_2, id_3, lane,lane2,lane3)
        kappa_10, ori_10, rul_10 = future_parameter(10, mapgeometry, posx, posy, id_1, id_2, id_3, lane,lane2,lane3)
        kappa_15, ori_15, rul_15 = future_parameter(15, mapgeometry, posx, posy, id_1, id_2, id_3, lane,lane2,lane3)

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
        dic.append(dis_to_center)

    return list0,list1,list2,list3,dic,kru,ori,rul,kru5,ori5,rul5,kru10,ori10,rul10,kru15,ori15,rul15


