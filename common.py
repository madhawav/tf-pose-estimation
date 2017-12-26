from collections import defaultdict
from enum import Enum
import math
import logging

import numpy as np
import itertools
import cv2
from scipy.ndimage.filters import maximum_filter

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


regularizer_conv = 0.04
regularizer_dsconv = 0.004
batchnorm_fused = True


class CocoPart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18

CocoPairs = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
    (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)
]   # = 19
CocoPairsRender = CocoPairs[:-2]
CocoPairsNetwork = [
    (12, 13), (20, 21), (14, 15), (16, 17), (22, 23), (24, 25), (0, 1), (2, 3), (4, 5),
    (6, 7), (8, 9), (10, 11), (28, 29), (30, 31), (34, 35), (32, 33), (36, 37), (18, 19), (26, 27)
 ]  # = 19

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NMS_Threshold = 0.1
InterMinAbove_Threshold = 6
Inter_Threashold = 0.1
Min_Subset_Cnt = 4
Min_Subset_Score = 0.8
Max_Human = 96


def connections_to_human(connections, heatMat):
    point_dict = defaultdict(lambda: None)
    for conn in connections:
        point_dict[conn['partIdx'][0]] = (conn['partIdx'][0], (conn['c1'][0] / heatMat.shape[2], conn['c1'][1] / heatMat.shape[1]), heatMat[conn['partIdx'][0], conn['c1'][1], conn['c1'][0]])
        point_dict[conn['partIdx'][1]] = (conn['partIdx'][1], (conn['c2'][0] / heatMat.shape[2], conn['c2'][1] / heatMat.shape[1]), heatMat[conn['partIdx'][1], conn['c2'][1], conn['c2'][0]])
    return point_dict


def non_max_suppression(np_input, window_size=3, threshold=NMS_Threshold):
    under_threshold_indices = np_input < threshold
    np_input[under_threshold_indices] = 0
    return np_input*(np_input == maximum_filter(np_input, footprint=np.ones((window_size, window_size))))


def estimate_pose(heatMat, pafMat):
    if heatMat.shape[2] == 19:
        heatMat = np.rollaxis(heatMat, 2, 0)
    if pafMat.shape[2] == 38:
        pafMat = np.rollaxis(pafMat, 2, 0)

    # reliability issue.
    logging.debug('preprocess')
    heatMat = heatMat - heatMat.min(axis=1).min(axis=1).reshape(19, 1, 1)
    heatMat = heatMat - heatMat.min(axis=2).reshape(19, heatMat.shape[1], 1)

    _NMS_Threshold = max(np.average(heatMat) * 4.0, NMS_Threshold)
    _NMS_Threshold = min(_NMS_Threshold, 0.3)

    logging.debug('nms, th=%f' % _NMS_Threshold)
    # heatMat = gaussian_filter(heatMat, sigma=0.5)
    coords = []
    for plain in heatMat[:-1]:
        nms = non_max_suppression(plain, 5, _NMS_Threshold)
        coords.append(np.where(nms >= _NMS_Threshold))

    logging.debug('estimate_pose1 : estimate pairs')
    connection_all = []
    for (idx1, idx2), (paf_x_idx, paf_y_idx) in zip(CocoPairs, CocoPairsNetwork):
        connection = estimate_pose_pair(coords, idx1, idx2, pafMat[paf_x_idx], pafMat[paf_y_idx])
        connection_all.extend(connection)

    logging.debug('estimate_pose2, connection=%d' % len(connection_all))
    connection_by_human = dict()
    for idx, c in enumerate(connection_all):
        connection_by_human['human_%d' % idx] = [c]

    no_merge_cache = defaultdict(list)
    while True:
        is_merged = False
        for k1, k2 in itertools.combinations(connection_by_human.keys(), 2):
            if k1 == k2:
                continue
            if k2 in no_merge_cache[k1]:
                continue
            for c1, c2 in itertools.product(connection_by_human[k1], connection_by_human[k2]):
                if len(set(c1['uPartIdx']) & set(c2['uPartIdx'])) > 0:
                    is_merged = True
                    connection_by_human[k1].extend(connection_by_human[k2])
                    connection_by_human.pop(k2)
                    break
            if is_merged:
                no_merge_cache.pop(k1, None)
                break
            else:
                no_merge_cache[k1].append(k2)

        if not is_merged:
            break

    logging.debug('estimate_pose3')

    # reject by subset count
    connection_by_human = {k: v for (k, v) in connection_by_human.items() if len(v) >= Min_Subset_Cnt}

    # reject by subset max score
    connection_by_human = {k: v for (k, v) in connection_by_human.items() if max([ii['score'] for ii in v]) >= Min_Subset_Score}

    logging.debug('estimate_pose4')
    return [connections_to_human(conn, heatMat) for conn in connection_by_human.values()]


def estimate_pose_pair(coords, partIdx1, partIdx2, pafMatX, pafMatY):
    connection_temp = []
    peak_coord1, peak_coord2 = coords[partIdx1], coords[partIdx2]

    cnt = 0
    for idx1, (y1, x1) in enumerate(zip(peak_coord1[0], peak_coord1[1])):
        for idx2, (y2, x2) in enumerate(zip(peak_coord2[0], peak_coord2[1])):
            score, count = get_score(x1, y1, x2, y2, pafMatX, pafMatY)
            cnt += 1
            if (partIdx1, partIdx2) in [(2, 3), (3, 4), (5, 6), (6, 7)]:
                if count < InterMinAbove_Threshold // 2 or score <= 0.0:
                    continue
            elif count < InterMinAbove_Threshold or score <= 0.0:
                continue
            connection_temp.append({
                'score': score,
                'c1': (x1, y1),
                'c2': (x2, y2),
                'idx': (idx1, idx2),
                'partIdx': (partIdx1, partIdx2),
                'uPartIdx': ('{}-{}-{}'.format(x1, y1, partIdx1), '{}-{}-{}'.format(x2, y2, partIdx2))
            })

    connection = []
    used_idx1, used_idx2 = [], []
    for candidate in sorted(connection_temp, key=lambda x: x['score'], reverse=True):
        # check not connected
        if candidate['idx'][0] in used_idx1 or candidate['idx'][1] in used_idx2:
            continue
        connection.append(candidate)
        used_idx1.append(candidate['idx'][0])
        used_idx2.append(candidate['idx'][1])

    return connection


def get_score(x1, y1, x2, y2, pafMatX, pafMatY):
    __num_inter = 10
    __num_inter_f = float(__num_inter)
    dx, dy = x2 - x1, y2 - y1
    normVec = math.sqrt(dx ** 2 + dy ** 2)

    if normVec < 1e-4:
        return 0.0, 0

    vx, vy = dx / normVec, dy / normVec

    xs = np.arange(x1, x2, dx / __num_inter_f) if x1 != x2 else np.full((__num_inter, ), x1)
    ys = np.arange(y1, y2, dy / __num_inter_f) if y1 != y2 else np.full((__num_inter, ), y1)
    xs = (xs + 0.5).astype(np.int8)
    ys = (ys + 0.5).astype(np.int8)

    # without vectorization
    pafXs = np.zeros(__num_inter)
    pafYs = np.zeros(__num_inter)
    for idx, (mx, my) in enumerate(zip(xs, ys)):
        pafXs[idx] = pafMatX[my][mx]
        pafYs[idx] = pafMatY[my][mx]

    # vectorization slow?
    # pafXs = pafMatX[ys, xs]
    # pafYs = pafMatY[ys, xs]

    local_scores = pafXs * vx + pafYs * vy
    thidxs = local_scores > Inter_Threashold

    return sum(local_scores * thidxs), sum(thidxs)


def read_imgfile(path, width, height):
    val_image = cv2.imread(path)
    return preprocess(val_image, width, height)


def preprocess(img, width, height):
    val_image = cv2.resize(img, (width, height))
    val_image = val_image.astype(float)
    val_image = val_image * (2.0 / 255.0) - 1.0
    return val_image

import face_recognition
madhawa_face_image = face_recognition.load_image_file("madhawa.jpeg")
madhawa_face_encoding = face_recognition.face_encodings(madhawa_face_image)[0]

imesha_face_image = face_recognition.load_image_file("imesha.png")
imesha_face_encoding = face_recognition.face_encodings(imesha_face_image)[0]

def draw_humans(img, human_list):
    img_copied = np.copy(img)
    image_h, image_w = img_copied.shape[:2]
    centers = {}
    _i = 0
    for human in human_list:
        part_idxs = human.keys()

        minx, miny, maxx, maxy = (30000,30000,-30000,-30000)

        # draw point
        for i in range(CocoPart.Background.value):
            if i not in part_idxs:
                continue
            part_coord = human[i][1]
            center = (int(part_coord[0] * image_w + 0.5), int(part_coord[1] * image_h + 0.5))
            centers[i] = center

            minx = min(minx,center[0])
            maxx = max(maxx,center[0])
            miny = min(miny,center[1])
            maxy = max(maxy,center[1])

        roi = img_copied[max(0,miny-50):min(image_h,maxy+50),max(0,minx-50):min(image_w,maxx+50)]

        face_locations = face_recognition.face_locations(roi, number_of_times_to_upsample=1)
        for (top,right,bottom,left) in face_locations:
            cv2.rectangle(roi,(left,top),(right,bottom),(0,0,255),3)
            cv2.rectangle(img_copied, (left + max(0,minx-50), top + max(0,miny-50)), (right + max(0,minx-50), bottom + max(0,miny-50)), (0, 0, 255), 3)

        unknown_face_encodings = face_recognition.face_encodings(roi, face_locations)
        for x in range(len(unknown_face_encodings)):
            unknown_face_encoding = unknown_face_encodings[x]
            match = face_recognition.compare_faces([madhawa_face_encoding, imesha_face_encoding], unknown_face_encoding, tolerance=0.5)
            names = ["madhawa","imesha"]
            distances = face_recognition.face_distance([madhawa_face_encoding,imesha_face_encoding],unknown_face_encoding)
            distance_names = [(distances[i],names[i]) for i in range(len(distances))]
            distance_names = sorted(distance_names,key=lambda x: x[0])

            fullname = ""

            for distance,name in distance_names:
                if distance < 0.5:
                    fullname += name + " "
            # if match[0]:
            #     name += "Madhawa "
            # if match[1]:
            #     name += "Imesha "
            cv2.putText(img_copied, fullname, (face_locations[x][3] + max(0,minx-50), face_locations[x][0] + + max(0,miny-50)), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0))

        # cv2.imshow("person_preview " + str(_i),roi)
        _i += 1

        cv2.rectangle(img_copied,(minx-50,miny-50),(maxx+50,maxy+50),(255,0,0),2)

        for i,center in centers.items():
            cv2.circle(img_copied, center, 3, CocoColors[i], thickness=3, lineType=8, shift=0)

        # draw line
        for pair_order, pair in enumerate(CocoPairsRender):
            if pair[0] not in part_idxs or pair[1] not in part_idxs:
                continue

            img_copied = cv2.line(img_copied, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3)

    return img_copied
