import matplotlib.pyplot as plt
import numpy as np
import math

from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
import torch
import cv2
import posenet.utils as ut
import time
import posenet
import pandas as pd
from openpyxl import load_workbook
import xlsxwriter
import xlrd
import openpyxl


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


def plot_distance_between_frames(hands_per_frame, nose_per_frame, frame_count, end, start, fps, draw_graphs):
    distances = {}
    i = 0

    for current_frame, next_frame in zip(list(hands_per_frame.values())[0::1], list(hands_per_frame.values())[1::1]):
        dist_skels = []
        min_length = min(len(current_frame), len(next_frame))
        # going over each skeleton

        for j in range(0, min_length):  # run for every person
            current_skel, next_skel = current_frame[j], next_frame[j]
            cur_left, cur_right = current_skel
            next_left, next_right = next_skel
            distance_left = math.sqrt((next_left[0] - cur_left[0]) ** 2 + (next_left[1] - cur_left[1]) ** 2)
            distance_right = math.sqrt((next_right[0] - cur_right[0]) ** 2 + (next_right[1] - cur_right[1]) ** 2)
            dist_skels.append(["left", distance_left])
            dist_skels.append(["right", distance_right])

        '''
        for j in range(0, 1): #run for every person
            if (min_length!=1):
                j = 1;
                current_skel, next_skel = current_frame[j], next_frame[j]
                cur_left, cur_right = current_skel
                next_left, next_right = next_skel
                distance_left = math.sqrt((next_left[0] - cur_left[0]) ** 2 + (next_left[1] - cur_left[1]) ** 2)
                distance_right = math.sqrt((next_right[0] - cur_right[0]) ** 2 + (next_right[1] - cur_right[1]) ** 2)
                dist_skels.append(["left", distance_left])
                dist_skels.append(["right", distance_right])
        '''

        distances[i] = dist_skels
        i += 1

    lst_distances = list(distances.values())
    print(lst_distances)
    # sanity check
    print(i)
    print(frame_count)
    num_of_humans = 2
    _min = 100
    _max = 0
    y_axis = []
    x_axis = []
    count = 0
    for a in lst_distances:
        # if len(a) != num_of_humans * 2:
        # break
        for aa in a:
            if aa[1] > _max:
                _max = aa[1]
            if aa[1] < _min:
                _min = aa[1]

            y_axis.append(aa[1])
            x_axis.append(count)
            # adjusting fo for 2 frames at a time
            # y_axis.append(aa[1])
            # x_axis.append(count+1)
            count += 1

    print("min: ", _min, "max: ", _max)
    print("time = ", (end - start))
    print("fps = ", fps)

    peaks, _ = find_peaks(y_axis, distance=10,
                          prominence=1)  # 10 frames distances between peaks at least, prominence=1 say that the value of the peak must be higher than 1. return peaks that is the index of the frames that higher than 1 and max locality

    np.diff(peaks)
    num_of_peaks = str(len(peaks))
    if draw_graphs:
        plt.figure(0)
        plt.subplot(1, 2, 1)
        plt.plot(x_axis, y_axis)
        max_peak = str(max([y_axis[i] for i in peaks]))
        plt.title("number of peaks: " + num_of_peaks + " biggest peak: " + max_peak)
        plt.xlabel('frame index')
        plt.ylabel('Speed')
        peaks_y = [y_axis[i] for i in peaks]
        plt.plot(peaks, peaks_y, "x")

        plt.subplot(1, 2, 2)
        y_interp = UnivariateSpline(x_axis, y_axis, k=3, s=0)  # cubic spline
        acc = y_interp.derivative()(x_axis)
        plt.plot(x_axis, acc, 'b')
        plt.xlabel('frame index')
        plt.ylabel('acceleration')
        plt.show()

        finish = False
        maybe_fight = False

    # we want to find the frame where the action occurs
    i = 0
    while i < peaks.size:
        if maybe_fight:
            break
        if finish == False:
            current_frame_id = -1
            current_peak_value = y_axis[peaks[i]]
            suspect_person_id = 0
            for current_frame, next_frame in zip(list(hands_per_frame.values())[0::1],
                                                 list(hands_per_frame.values())[1::1]):
                if finish:
                    break
                min_length = min(len(current_frame), len(next_frame))
                current_frame_id = current_frame_id + 1

                for j in range(0, min_length):  # run for every person
                    suspect_person_id = j
                    current_skel, next_skel = current_frame[j], next_frame[j]
                    cur_left, cur_right = current_skel
                    next_left, next_right = next_skel
                    distance_left = math.sqrt((next_left[0] - cur_left[0]) ** 2 + (next_left[1] - cur_left[1]) ** 2)
                    distance_right = math.sqrt(
                        (next_right[0] - cur_right[0]) ** 2 + (next_right[1] - cur_right[1]) ** 2)
                    if distance_left == current_peak_value or distance_right == current_peak_value:
                        finish = True
                        break

        else:  # calculate the distance between the kicker to everyone in the suspect frame
            finish = False
            i = i + 1
            current_frame_nose = nose_per_frame[current_frame_id + 1]
            if len(current_frame_nose) >= 2:
                for k in range(0, len(current_frame_nose)):
                    if k != suspect_person_id:
                        person_a = current_frame_nose[suspect_person_id][0]
                        person_a_x = person_a[0]
                        person_a_y = person_a[1]
                        person_b = current_frame_nose[k][0]
                        person_b_x = person_b[0]
                        person_b_y = person_b[1]
                        distance_nose = math.sqrt((person_a_x - person_b_x) ** 2 + (person_a_y - person_b_y) ** 2)
                        if distance_nose < 120:  # maybe we can change the value of 120 later
                            maybe_fight = True

    return peaks, peaks_y, acc, maybe_fight


def save_as_mp4_file(frame_array, fps, size):
    # saving file
    pathOut = 'exOUT.mp4'
    # fps = len(frame_array) / (end - start)
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write


def run_on_vid(path, name_of_vid, draw_graphs):
    # not quite sure - not gonna touch it. default value given in original work.
    scale_factor = 0.7125

    # the default model, can be (50, 75, 100, 101) it is the depth multiplier
    model = posenet.load_model(101)
    # push model to available gpu
    model = model.cuda()
    output_stride = model.output_stride
    # file handling
    file_path = path
    cap = cv2.VideoCapture(file_path)
    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened():
        print("error on loading file")
    # going over each frame
    start = time.time()
    frame_count = 0
    frame_array = []
    size = (0, 0)
    res = True

    hands_per_frame = {}
    legs_per_frame = {}
    nose_per_frame = {}
    left_leg_per_frame = {}
    right_leg_per_frame = {}

    right_leg_angle = []
    left_leg_angle = []

    is_more_than_1 = False

    while res:

        res, img = cap.read()
        if not res:
            # raise IOError("webcam failure")
            break;
        input_image, display_image, output_scale = ut._process_input(img, scale_factor, output_stride)

        # some sort of pref. acceleration
        with torch.no_grad():
            input_image = torch.Tensor(input_image).cuda()

        # call model with frame
        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

        # making results usable - mainly just fiddling with dimensions and preparing data structures
        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
            heatmaps_result.squeeze(0),
            offsets_result.squeeze(0),
            displacement_fwd_result.squeeze(0),
            displacement_bwd_result.squeeze(0),
            output_stride=output_stride,
            max_pose_detections=10,
            min_pose_score=0.15)

        # --------------------- making only hands appear-------------------------------
        status = True
        if pose_scores.max == 0:
            status = False
        # if there's only 1 person in the whole video, it's probably non violent
        length = len(pose_scores)
        not_zero = length - np.count_nonzero(pose_scores == 0)
        if not_zero > 1:
            is_more_than_1 = True
        skeletons_hands = []
        skeletons_legs = []
        skeletons_nose = []
        frame_count += 1
        left_leg = []
        right_leg = []
        for pi in range(len(pose_scores)):  # going over number of skeletons appearing in image
            hands = []
            legs = []
            nose = []
            if pose_scores[pi] == 0.:
                break
            print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
            for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                if posenet.PART_NAMES[ki] == "leftWrist" or posenet.PART_NAMES[ki] == "rightWrist":
                    print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))
                    hands.append(totuple(c))
            skeletons_hands.append(hands)
            for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                if posenet.PART_NAMES[ki] == "leftAnkle" or posenet.PART_NAMES[ki] == "rightAnkle":
                    print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))
                    legs.append(totuple(c))
            skeletons_legs.append(legs)
            for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                if posenet.PART_NAMES[ki] == "nose":
                    print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))
                    nose.append(totuple(c))
            skeletons_nose.append(nose)

            '''

             only if we use angle

            for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                if posenet.PART_NAMES[ki] == "leftHip" or posenet.PART_NAMES[ki] == "leftKnee" or posenet.PART_NAMES[ki] == "leftAnkle":
                    print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))
                    left_leg.append(totuple(c))
#            skeletons.append(left_leg)
            for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                if posenet.PART_NAMES[ki] == "rightHip" or posenet.PART_NAMES[ki] == "rightKnee" or posenet.PART_NAMES[ki] == "rightAnkle":
                    print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))
                    right_leg.append(totuple(c))
 #           skeletons.append(right_leg)

            '''

        hands_per_frame[frame_count] = skeletons_hands
        legs_per_frame[frame_count] = skeletons_legs
        nose_per_frame[frame_count] = skeletons_nose
        #      left_leg_per_frame[frame_count] = skeletons
        #      right_leg_per_frame[frame_count] = skeletons

        '''
        right_leg_ankle_sub_knee = np.subtract((right_leg[2]), (right_leg[1]))
        right_leg_hip_sub_knee = np.subtract((right_leg[0]), (right_leg[1]))

        cos_right = np.dot(right_leg_ankle_sub_knee, right_leg_hip_sub_knee) / (np.linalg.norm(right_leg_ankle_sub_knee) * np.linalg.norm(right_leg_hip_sub_knee))
        angle_right = math.degrees(np.arccos(cos_right)) #angle is in degrees
        right_leg_angle.append(angle_right)

        left_leg_ankle_sub_knee = np.subtract((left_leg[2]), (left_leg[1]))
        left_leg_hip_sub_knee = np.subtract((left_leg[0]), (left_leg[1]))

        cos_left = np.dot(left_leg_ankle_sub_knee, left_leg_hip_sub_knee) / (np.linalg.norm(left_leg_ankle_sub_knee) * np.linalg.norm(left_leg_hip_sub_knee))
        angle_left = math.degrees(np.arccos(cos_left)) #angle is in degrees
        left_leg_angle.append(angle_left)
        '''
        # ------------------------------------------------------------------------------

        # normalization?
        keypoint_coords *= output_scale

        # drawing the skeleton
        overlay_image = posenet.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.15, min_part_score=0.1)

        '''
        # showing result
        cv2.imshow('posenet', overlay_image)
        # printing to terminal
        for pi in range(len(pose_scores)):
            if pose_scores[pi] == 0.:
                break
            print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
            for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))

        '''

        # saving as a file
        frame_array.append(overlay_image)  # saving the image with the skelton in an array
        height, width, _ = overlay_image.shape
        size = (width, height)

        # halting condition for video to end.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end = time.time()
    fps = frame_count / (end - start)
    num_fo_frames_3_sec = fps * 3

    '''
    # ----------------PLOT Graph angles legs---------------------------------
    index1 = [i for i in range(0, len(right_leg_angle))]
    plt.plot(index1, right_leg_angle, 'ro--', label="right")
    index2 = [i for i in range(0, len(left_leg_angle))]
    plt.plot(index2, left_leg_angle, 'bo--', label="left")
    # -----------------------------------------------------------------------


    difference = abs(np.subtract(right_leg_angle, left_leg_angle))
    # ----------------PLOT Graph difference between angles legs---------------------------------
    index2 = [i for i in range(0, len(difference))]
    plt.plot(index2, difference, 'ro--', label="diff")
    # ------------------------------------------------------------------------------------------
    '''

    # -------------distance between frames---------------
    print(hands_per_frame)

    # calculate how many frames there is with people
    not_empty_in_hands_per_frame = 0
    for a in hands_per_frame:
        if hands_per_frame[a]:
            not_empty_in_hands_per_frame = not_empty_in_hands_per_frame + 1

    not_empty_in_legs_per_frame = 0
    for a in legs_per_frame:
        if legs_per_frame[a]:
            not_empty_in_legs_per_frame = not_empty_in_legs_per_frame + 1

    maybe_only_one_person = False

    # zero hands and legs if there is no one
    if not_empty_in_hands_per_frame < 2:
        num_of_peaks_hands = 0
        peaks_y_hands = [0]
        acc_y_hands = [0]
        maybe_fight_hands = False

        maybe_only_one_person = True
    else:
        num_of_peaks_hands, peaks_y_hands, acc_y_hands, maybe_fight_hands = plot_distance_between_frames(
            hands_per_frame, nose_per_frame, frame_count, end, start, fps, draw_graphs)

    if not_empty_in_legs_per_frame < 2:
        num_of_peaks_legs = 0
        peaks_y_legs = [0]
        acc_y_legs = [0]
        maybe_fight_legs = False

        if (maybe_only_one_person):
            is_more_than_1 = False


    else:
        num_of_peaks_legs, peaks_y_legs, acc_y_legs, maybe_fight_legs = plot_distance_between_frames(legs_per_frame,
                                                                                                     nose_per_frame,
                                                                                                     frame_count, end,
                                                                                                     start, fps,
                                                                                                     draw_graphs)

    # ----------------------------------------------------------

    # ---------------------Decision Rule------------------------
    # TODO: decide decision rule

    # ------------------------------------------------------
    # save_as_mp4_file(frame_array, start, end, size)
    video = cv2.VideoWriter(name_of_vid + ".avi", 0, fps, (width, height))  # maybe need to down the fps
    for image in frame_array:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()

    return num_of_peaks_hands, peaks_y_hands, acc_y_hands, status, is_more_than_1, fps, num_of_peaks_legs, peaks_y_legs, acc_y_legs, maybe_fight_hands, maybe_fight_legs, number_of_frames


def how_many_above(list1, list2):
    length = min(len(list1), len(list2))

    count = 0
    for j in range(0, length):
        if list1[j] > list2[j]:
            count += 1
    if length == 0:
        print("ERROR ON how_many_above")
        return 0.01
    else:
        return 100 * (count / length)  # precentage


def run_and_draw_graph_of_speed_and_acc_Pelicus(no_fight_path, fight_path, path_end_violent, path_end_non_violent,
                                                draw_graphs=True):
    # open excel session
    writer = pd.ExcelWriter('aa.xlsx', engine='openpyxl')
    # try to open an existing workbook
    writer.book = load_workbook('aa.xlsx')
    # copy existing sheets
    writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)

    # tested averages and got the following reesults
    '''
    #hands
    avg_max_peak_violent = 233.57
    avg_num_peak_violent = 0.354
    avg_max_peak_non_violent = 116.47
    avg_num_peak_non_violent = 0.183

    #legs
    avg_max_peak_violent2 = 233.57
    avg_num_peak_violent2 = 0.354
    avg_max_peak_non_violent2 = 116.47
    avg_num_peak_non_violent2 = 0.183
    '''

    # numers from chainese videos (get by the train videos):

    # hands
    avg_max_peak_violent = 218.60760711880567
    avg_num_peak_violent = 0.20097716894977116
    avg_max_peak_non_violent = 110.13890386440104
    avg_num_peak_non_violent = 0.13626484018264778

    # legs
    avg_max_peak_violent2 = 215.39246498432587
    avg_num_peak_violent2 = 0.1995707762557075
    avg_max_peak_non_violent2 = 105.68163332124655
    avg_num_peak_non_violent2 = 0.13372602739725958

    num_peaks_violent = []
    num_peaks_non_violent = []
    num_peaks_violent_legs = []
    num_peaks_non_violent_legs = []
    index = []
    max_peak_violent = []
    max_peak_non_violent = []
    max_peak_violent_legs = []
    max_peak_non_violent_legs = []
    peaks_violent = []
    peaks_non_violent = []
    false_positive = 0
    false_negative = 0
    false_negative_2 = 0
    false_positive_2 = 0
    num_of_videos = 0
    undec_counter = 0
    undec_non_violent = 0
    undec_violent = 0
    id = 1
    for i in range(701, 702):
        try:
            no_fight_path__ = no_fight_path + str(i) + path_end_non_violent
            fight_path__ = fight_path + "newfi" + str(i) + path_end_violent
            print("Current video number: ", i)
            name_of_violent_vid = str(i) + "_violent"
            peaks_violent, peaks_y_violent, acc_y_violent_hands, status_fi, is_more_than_1_violent, _, peaks_violent_legs, peaks_y_violent_legs, acc_y_violent_legs, maybe_fight_hands_violent, maybe_fight_legs_violent, number_of_frames_violent = run_on_vid(
                fight_path__, name_of_violent_vid, draw_graphs=draw_graphs)
            print(len(peaks_y_violent) / number_of_frames_violent)
            name_of_non_violent_vid = str(i) + "_non_violent"
            peaks_non_violent, peaks_y_non_violent, acc_y_non_violent_hands, status_no, is_more_than_1_non_violent, _, peaks_non_violent_legs, peaks_y_non_violent_legs, acc_y_non_violent_legs, maybe_fight_hands_non_violent, maybe_fight_legs_non_violent, number_of_frames_non_violent = run_on_vid(
                no_fight_path__, name_of_non_violent_vid, draw_graphs=draw_graphs)
            print(len(peaks_y_non_violent) / number_of_frames_non_violent)

            if status_fi and status_no:  # all went well
                num_of_videos = num_of_videos + 2

                # append data to excel file
                # read existing file
                reader = pd.read_excel(r'aa.xlsx')

                # new dataframe with same columns
                d = {'col1': [i], 'col2': [sum(peaks_y_violent) / len(peaks_y_violent)],
                     'col3': [abs(sum(acc_y_violent_hands) / len(acc_y_violent_hands))],
                     'col4': [len(peaks_y_violent) / number_of_frames_violent], 'col5': max(peaks_y_violent),
                     'col6': [sum(peaks_y_violent_legs) / len(peaks_y_violent_legs)],
                     'col7': [abs(sum(acc_y_violent_legs) / len(acc_y_violent_legs))],
                     'col8': [len(peaks_y_violent_legs) / number_of_frames_violent], 'col9': max(peaks_y_violent_legs),
                     'col10': [1]}
                df = pd.DataFrame(data=d)
                df.to_excel(writer, index=False, header=False, startrow=len(reader) + 1)
                writer.save()

                reader = pd.read_excel(r'aa.xlsx')
                d = {'col1': [i], 'col2': [sum(peaks_y_non_violent) / len(peaks_y_non_violent)],
                     'col3': [abs(sum(acc_y_non_violent_hands) / len(acc_y_non_violent_hands))],
                     'col4': [len(peaks_y_non_violent) / number_of_frames_non_violent],
                     'col5': max(peaks_y_non_violent),
                     'col6': [sum(peaks_y_non_violent_legs) / len(peaks_y_non_violent_legs)],
                     'col7': [abs(sum(acc_y_non_violent_legs) / len(acc_y_non_violent_legs))],
                     'col8': [len(peaks_y_non_violent_legs) / number_of_frames_non_violent],
                     'col9': max(peaks_y_non_violent_legs), 'col10': [0]}
                df = pd.DataFrame(data=d)
                df.to_excel(writer, index=False, header=False, startrow=len(reader) + 1)
                writer.save()

                num_peaks_violent.append(len(peaks_y_violent) / number_of_frames_violent)
                num_peaks_non_violent.append(len(peaks_y_non_violent) / number_of_frames_non_violent)
                max_peak_violent.append(max(peaks_y_violent))
                max_peak_non_violent.append(max(peaks_y_non_violent))

                num_peaks_violent_legs.append(len(peaks_y_violent_legs) / number_of_frames_violent)
                num_peaks_non_violent_legs.append(len(peaks_y_non_violent_legs) / number_of_frames_non_violent)
                max_peak_violent_legs.append(max(peaks_y_violent_legs))
                max_peak_non_violent_legs.append(max(peaks_y_non_violent_legs))

                false_negative_hands = False
                false_negative_2_hands = False
                false_positive_hands = False
                false_positive_2_hands = False
                undec_hands_violent = False
                undec_hands_non_violent = False

                print(len(peaks_y_violent))
                print(max(peaks_y_violent))

                # making a decision
                decision_violent_upper_bound = "violent" if (len(
                    peaks_y_violent) / number_of_frames_violent > avg_num_peak_violent or max(
                    peaks_y_violent) > avg_max_peak_violent) and is_more_than_1_violent and maybe_fight_hands_violent \
                    else "non-violent"
                decision_non_violent_upper_bound = "violent" if (len(
                    peaks_y_non_violent) / number_of_frames_non_violent > avg_num_peak_violent or max(
                    peaks_y_non_violent) > avg_max_peak_violent) and is_more_than_1_violent and maybe_fight_hands_non_violent \
                    else "non-violent"

                # false positive - decide it's violent when it is actually non violent
                if decision_non_violent_upper_bound == "violent":
                    false_positive_hands = True
                    false_positive += 1
                # false negative - decide it's non-violent when it is actually violent
                if decision_violent_upper_bound == "non-violent":
                    false_negative_hands = True

                # ------------------------------------------------- different approach
                if (len(peaks_y_violent) / number_of_frames_violent < avg_num_peak_non_violent or max(
                        peaks_y_violent) < avg_max_peak_non_violent) or not is_more_than_1_violent or not maybe_fight_hands_violent:
                    decision_violent_lower_bound = "non-violent"
                elif (len(peaks_y_violent) / number_of_frames_violent > avg_num_peak_violent or max(
                        peaks_y_violent) > avg_max_peak_violent) and is_more_than_1_violent and maybe_fight_hands_violent:
                    decision_violent_lower_bound = "violent"
                else:
                    decision_violent_lower_bound = "undecidable"

                if len(peaks_y_non_violent) / number_of_frames_non_violent < avg_num_peak_non_violent or max(
                        peaks_y_non_violent) < avg_max_peak_non_violent or not is_more_than_1_non_violent or not maybe_fight_hands_non_violent:
                    decision_non_violent_lower_bound = "non-violent"
                elif len(peaks_y_non_violent) / number_of_frames_non_violent > avg_num_peak_violent or max(
                        peaks_y_non_violent) > avg_max_peak_violent and is_more_than_1_non_violent and maybe_fight_hands_non_violent:
                    decision_non_violent_lower_bound = "violent"
                else:
                    decision_non_violent_lower_bound = "undecidable"

                # false positive - decide it's violent when it is actually non violent
                if decision_non_violent_lower_bound == "violent":
                    false_positive_2_hands = True
                    false_positive_2 += 1
                # false negative - decide it's non-violent when it is actually violent
                if decision_violent_lower_bound == "non-violent":
                    false_negative_2_hands = True
                if decision_non_violent_lower_bound == "undecidable":
                    undec_hands_non_violent = True
                if decision_violent_lower_bound == "undecidable":
                    undec_hands_violent = True

                ######### legs #############

                decision_violent_upper_bound = "violent" if (len(
                    peaks_y_violent_legs) / number_of_frames_violent > avg_num_peak_violent2 or max(
                    peaks_y_violent_legs) > avg_max_peak_violent2) and is_more_than_1_violent and maybe_fight_legs_violent \
                    else "non-violent"
                decision_non_violent_upper_bound = "violent" if (len(
                    peaks_y_non_violent_legs) / number_of_frames_non_violent > avg_num_peak_violent2 or max(
                    peaks_y_non_violent_legs) > avg_max_peak_violent2) and is_more_than_1_violent and maybe_fight_legs_non_violent \
                    else "non-violent"

                # false positive - decide it's violent when it is actually non violent
                if decision_non_violent_upper_bound == "violent":
                    if (false_positive_hands == False):
                        false_positive += 1
                # false negative - decide it's non-violent when it is actually violent
                if decision_violent_upper_bound == "non-violent":
                    if (false_negative_hands):
                        false_negative += 1

                # ------------------------------------------------- different approach
                if (len(peaks_y_violent_legs) / number_of_frames_violent < avg_num_peak_non_violent2 or max(
                        peaks_y_violent_legs) < avg_max_peak_non_violent2) or not is_more_than_1_violent or not maybe_fight_legs_violent:
                    decision_violent_lower_bound = "non-violent"
                elif (len(peaks_y_violent_legs) / number_of_frames_violent > avg_num_peak_violent2 or max(
                        peaks_y_violent_legs) > avg_max_peak_violent2) and is_more_than_1_violent and maybe_fight_legs_violent:
                    decision_violent_lower_bound = "violent"
                else:
                    decision_violent_lower_bound = "undecidable"

                if len(peaks_y_non_violent_legs) / number_of_frames_non_violent < avg_num_peak_non_violent2 or max(
                        peaks_y_non_violent_legs) < avg_max_peak_non_violent2 or not is_more_than_1_non_violent or not maybe_fight_legs_non_violent:
                    decision_non_violent_lower_bound = "non-violent"
                elif len(peaks_y_non_violent_legs) / number_of_frames_non_violent > avg_num_peak_violent2 or max(
                        peaks_y_non_violent_legs) > avg_max_peak_violent2 and is_more_than_1_non_violent and maybe_fight_legs_non_violent:
                    decision_non_violent_lower_bound = "violent"
                else:
                    decision_non_violent_lower_bound = "undecidable"

                # false positive - decide it's violent when it is actually non violent
                if decision_non_violent_lower_bound == "violent":
                    if (false_positive_2_hands == False):
                        false_positive_2 += 1
                # false negative - decide it's non-violent when it is actually violent
                if decision_violent_lower_bound == "non-violent":
                    if (false_negative_2_hands == True):
                        false_negative_2 += 1
                if decision_non_violent_lower_bound == "undecidable":
                    if (undec_hands_non_violent):
                        undec_counter += 1
                        undec_non_violent += 1
                if decision_violent_lower_bound == "undecidable":
                    if (undec_hands_violent):
                        undec_counter += 1
                        undec_violent += 1

        except:
            print("failed on iteration ", i)
            continue

    writer.close()

    false_positive_2 = 100 * (false_positive_2 / ((num_of_videos / 2) - undec_non_violent))
    false_negative_2 = 100 * (false_negative_2 / ((num_of_videos / 2) - undec_violent))
    false_positive = 100 * (false_positive / (num_of_videos / 2))
    false_negative = 100 * (false_negative / (num_of_videos / 2))
    print("false positive: ", false_positive, "%       false negative: ", false_negative, "%")
    print("Different approach: false positive: ", false_positive_2, "%       false negative: ", false_negative_2,
          "%    number of undecidables: ", undec_counter)

    # print(num_peaks_violent)
    # print(max_peak_violent)
    # print(max_peak_non_violent)

    # calculating averages - hands
    avg_max_peak_violent = sum(max_peak_violent) / len(max_peak_violent)
    avg_num_peak_violent = sum(num_peaks_violent) / len(num_peaks_violent)
    avg_max_peak_non_violent = sum(max_peak_non_violent) / len(max_peak_non_violent)
    avg_num_peak_non_violent = sum(num_peaks_non_violent) / len(num_peaks_non_violent)

    print("(VIOLENT-HANDS) average max peak is: ", avg_max_peak_violent)
    print("(VIOLENT-HANDS) average num peaks is: ", avg_num_peak_violent)
    print("(NON-VIOLENT-HANDS) average max peak is: ", avg_max_peak_non_violent)
    print("(NON-VIOLENT-HANDS) average num peaks is: ", avg_num_peak_non_violent)

    # calculating averages - legs
    avg_max_peak_violent_legs = sum(max_peak_violent_legs) / len(max_peak_violent_legs)
    avg_num_peak_violent_legs = sum(num_peaks_violent_legs) / len(num_peaks_violent_legs)
    avg_max_peak_non_violent_legs = sum(max_peak_non_violent_legs) / len(max_peak_non_violent_legs)
    avg_num_peak_non_violent_legs = sum(num_peaks_non_violent_legs) / len(num_peaks_non_violent_legs)

    print("(VIOLENT-LEGS) average max peak is: ", avg_max_peak_violent_legs)
    print("(VIOLENT-LEGS) average num peaks is: ", avg_num_peak_violent_legs)
    print("(NON-VIOLENT-LEGS) average max peak is: ", avg_max_peak_non_violent_legs)
    print("(NON-VIOLENT-LEGS) average num peaks is: ", avg_num_peak_non_violent_legs)
    print("num of undecidables violent videos is: ", undec_violent)
    print("num of undecidables non-violent videos is: ", undec_non_violent)
    print("num of videos is: ", num_of_videos)

    plt.subplot(1, 2, 1)
    index = [i for i in range(0, len(num_peaks_violent))]
    plt.plot(index, num_peaks_violent, 'ro--', label="violent")
    index = [i for i in range(0, len(num_peaks_non_violent))]
    plt.plot(index, num_peaks_non_violent, 'bo--', label="non-violent")

    plt.title(str(how_many_above(num_peaks_violent,
                                 num_peaks_non_violent)) + "% of violent cases are above its non-violent counterpart")
    plt.xlabel('number of peaks in a video')
    plt.ylabel('num of video in dataset')

    plt.legend(loc="upper left")

    plt.subplot(1, 2, 2)
    index = [i for i in range(0, len(max_peak_violent))]
    plt.plot(index, max_peak_violent, 'ro--', label="violent")
    index = [i for i in range(0, len(max_peak_non_violent))]
    plt.plot(index, max_peak_non_violent, 'bo--', label="non-violent")

    plt.title(str(how_many_above(max_peak_violent,
                                 max_peak_non_violent)) + "% of violent cases are above its non-violent counterpart")
    plt.xlabel('max peak in a video')
    plt.ylabel('num of video in dataset')

    plt.legend(loc="upper left")
    plt.show()
    return max_peak_violent, max_peak_non_violent, num_peaks_non_violent, num_peaks_violent


def Peliculas_results(max_peak_violent, max_peak_non_violent, num_peaks_non_violent, num_peaks_violent):
    length = min(len(max_peak_violent), len(max_peak_non_violent), len(num_peaks_non_violent), len(num_peaks_violent))
    count_or = 0
    count_and = 0
    for k in range(0, length):
        if max_peak_violent[k] > max_peak_non_violent[k] and num_peaks_violent[k] > num_peaks_non_violent[k]:
            count_and += 1
        if max_peak_violent[k] > max_peak_non_violent[k] or num_peaks_violent[k] > num_peaks_non_violent[k]:
            count_or += 1
    count_and_percentage = 100 * (count_and / length)
    count_or_percentage = 100 * (count_or / length)
    print(
        "Percentage of videos where both max peak and number of peaks is greater than its non-violent counterparts is: ",
        count_and_percentage)
    print(
        "Percentage of videos where either max peak or number of peaks is greater than its non-violent counterparts is: ",
        count_or_percentage)


def draw_borders_on_vid(path):
    peaks_indices, peaks_y, acc_y_hands, status, _, fps, peaks_legs, peaks_y_legs, acc_y_legs, maybe_fight_hands, maybe_fight_legs = run_on_vid(
        path, draw_graphs=True)
    # path = "with_skel.avi"
    cap = cv2.VideoCapture(path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened():
        print("error on loading file")

    frame_counter = 1
    marked_frames = []
    res = True
    while res:
        res, img = cap.read()
        if not res:
            break
        '''
        if frame_counter < 30 :
            frame_counter += 1
            continue
        '''

        color = [0, 255, 0]  # green
        # checking if it's within 2 frames from a peak
        if len(peaks_indices) > 19 or True:
            if 0 <= frame_counter <= 30:  # Snir and Nir do it only to the first 30 frames... (don't know why)
                for i in peaks_indices:
                    if i - 2 <= frame_counter <= i + 2:  # take 2 frames from peak up and down to be red
                        color = [0, 0, 255]  # red
        # border widths; I set them all to 30
        top, bottom, left, right = [30] * 4
        img_with_border = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        marked_frames.append(img_with_border)
        height, width, layers = img_with_border.shape
        cv2.imshow('ah', img_with_border)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_counter += 1
        time.sleep(0.1)
    video = cv2.VideoWriter("output_vid.avi", 0, fps * 0.5, (width, height))
    for image in marked_frames:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()