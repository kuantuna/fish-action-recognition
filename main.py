# Tuna Tuncer S018474 Department of Computer Science

import cv2
import numpy as np
import os


def divide_video(file_name: str):
    cap = cv2.VideoCapture(f'videos\\{file_name}')
    cnt = 0
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    y1, w1, h1 = 0, w_frame, int(h_frame/3)
    y2, w2, h2 = int(h_frame/3), w_frame, int(h_frame/3)
    y3, w3, h3 = int(h_frame/3)*2, w_frame, int(h_frame/3)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_file_names = []
    for idx in range(1, 4):
        file_name_list = file_name.split(".")
        file_name_list.insert(1, f"-{idx}")
        file_name_list.insert(2, ".")
        output_file_names.append(''.join(file_name_list))
    if not os.path.exists(f"divided-videos"):
        os.makedirs(f"divided-videos")
    out1 = cv2.VideoWriter(
        f'divided-videos\\{output_file_names[0]}', fourcc, fps, (w1, h1))
    out2 = cv2.VideoWriter(
        f'divided-videos\\{output_file_names[1]}', fourcc, fps, (w2, h2))
    out3 = cv2.VideoWriter(
        f'divided-videos\\{output_file_names[2]}', fourcc, fps, (w3, h3))

    while(cap.isOpened()):
        ret, frame = cap.read()
        cnt += 1
        if ret == True:
            crop_frame1 = frame[y1:y1+h1]
            crop_frame2 = frame[y2:y2+h2]
            crop_frame3 = frame[y3:y3+h3]
            out1.write(crop_frame1)
            out2.write(crop_frame2)
            out3.write(crop_frame3)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out1.release()
    out2.release()
    out3.release()
    cv2.destroyAllWindows()


_, bg_mask1 = cv2.threshold(cv2.imread(
    "bg_mask1.jpg", cv2.IMREAD_GRAYSCALE), 128, 255, cv2.THRESH_BINARY)
_, bg_mask2 = cv2.threshold(cv2.imread(
    "bg_mask2.jpg", cv2.IMREAD_GRAYSCALE), 128, 255, cv2.THRESH_BINARY)
_, bg_mask3 = cv2.threshold(cv2.imread(
    "bg_mask3.jpg", cv2.IMREAD_GRAYSCALE), 128, 255, cv2.THRESH_BINARY)


def get_valid_keypoints(keypoints, ref_keypoints):
    distance_lower_threshold = 5
    distance_upper_threshold = 120
    y_threshold = 1
    tmp_valid_kps = []
    tmp_valid_ref_kps = []
    while True:
        for kp, rkp in zip(keypoints, ref_keypoints):
            distance = np.linalg.norm(kp - rkp)
            if ((distance > distance_lower_threshold) or (abs(kp[1] - rkp[1]) > y_threshold)) and (distance < distance_upper_threshold):
                if ~any((kp == x).all() for x in np.array(tmp_valid_kps)):
                    tmp_valid_kps.append(kp)
                    tmp_valid_ref_kps.append(rkp)
        if len(tmp_valid_kps) > 30:
            break
        else:
            distance_lower_threshold -= 0.5
            if distance_lower_threshold <= 0:
                return keypoints, ref_keypoints
    return np.array(tmp_valid_kps), np.array(tmp_valid_ref_kps)


def track_fishes(file_name: str, bg_mask):

    video = cv2.VideoCapture(f"divided-videos\\{file_name}")
    frame_counter = 0
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.03,
                          minDistance=7,
                          blockSize=7)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10, 0.03))
    kp_cnt_by_region = {ctr: [] for ctr in [
        'oleft', 'omiddle', 'oright', 'nleft', 'nright', 'kps']}

    _, ref_frame = video.read()
    ref_frame_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    ref_keypoints = cv2.goodFeaturesToTrack(ref_frame_gray, mask=bg_mask,  # (100, 1, 2)  x and y coordinate of 100 points in the ref_frame_gray
                                            **feature_params)

    color = np.random.randint(0, 255, (1000, 3))
    mask = np.zeros_like(ref_frame)

    feature_params["maxCorners"] = 20

    while True:
        _, frame = video.read()
        if frame is None:
            break
        frame_counter += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        keypoints, status, _ = cv2.calcOpticalFlowPyrLK(
            ref_frame_gray, frame_gray, ref_keypoints, None, **lk_params)

        valid_kps = keypoints[status == 1]
        valid_ref_kps = ref_keypoints[status == 1]

        valid_kps, valid_ref_kps = get_valid_keypoints(
            valid_kps, valid_ref_kps)

        transform, inliers = cv2.estimateAffine2D(
            valid_ref_kps, valid_kps, confidence=0.999)

        # ---------------------------
        # for visualization purposes:
        for i, (kp, ref_kp) in enumerate(zip(valid_kps, valid_ref_kps)):
            x1, y1 = kp.ravel()
            x2, y2 = ref_kp.ravel()
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            mask = cv2.line(mask, (x1, y1), (x2, y2), color[i].tolist(), 2)
            frame = cv2.circle(frame, (x1, y1), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        video_name = file_name.split(".")[0]
        if not os.path.exists(f"final\\{video_name}"):
            os.makedirs(f"final\\{video_name}")
        cv2.imwrite(
            f"final\\{video_name}\\frame{frame_counter}.jpg", img)
        # ---------------------------

        left_corner_cnt, right_corner_cnt, middle_corner_cnt = 0, 0, 0
        for kp in valid_kps:
            x = kp[0]
            if x < 250:
                left_corner_cnt += 1
            elif x > 1114:
                right_corner_cnt += 1
            else:
                middle_corner_cnt += 1

        corners = cv2.goodFeaturesToTrack(frame_gray, mask=bg_mask,
                                          **feature_params)
        tmp_kps = valid_kps.tolist()
        valid_kps_rint = np.rint(valid_kps)
        left_corner_cnt_new, right_corner_cnt_new = 0, 0
        for corner in corners:
            if ~any((corner[0] == x).all() for x in valid_kps_rint):
                tmp_kps.append(corner[0])
            x = corner[0][0]
            if x < 210:
                left_corner_cnt_new += 1
            if x > 1154:
                right_corner_cnt_new += 1
        for x, y in zip(['oleft', 'omiddle', 'oright', 'nleft', 'nright', 'kps'],
                        [left_corner_cnt, middle_corner_cnt, right_corner_cnt, left_corner_cnt_new, right_corner_cnt, len(valid_kps)]):
            kp_cnt_by_region[x].append(y)
        valid_kps = np.array(tmp_kps, dtype='float32')

        ref_frame_gray = frame_gray.copy()
        ref_keypoints = valid_kps.reshape(-1, 1, 2)

    video.release()

    actions = []
    for i in range(1, frame_counter):
        kcbr = kp_cnt_by_region
        # out -> left
        if ((kcbr['nleft'][i] > 10) or (kcbr['nleft'][i] - kcbr['nleft'][i-1] > 5)) and (kcbr['omiddle'][i] >= kcbr['omiddle'][i-1]):
            actions.append({'action': 'out -> left', 'frame': i})
        # right <- out
        if ((kcbr['nright'][i] > 10) or (kcbr['nright'][i] - kcbr['nright'][i-1] > 5)) and (kcbr['omiddle'][i] >= kcbr['omiddle'][i-1]):
            actions.append({'action': 'out -> right', 'frame': i})
        # left -> middle
        if (kcbr['oleft'][i-1] - kcbr['oleft'][i] > 5) and (kcbr['omiddle'][i] - kcbr['omiddle'][i-1] > 5):
            actions.append({'action': 'left -> middle', 'frame': i})
        # middle <- right
        if (kcbr['oright'][i-1] - kcbr['oright'][i] > 5) and (kcbr['omiddle'][i] - kcbr['omiddle'][i-1] > 5):
            actions.append({'action': 'right -> middle', 'frame': i})
        # left <- middle
        if (kcbr['omiddle'][i-1] - kcbr['omiddle'][i] > 5) and (kcbr['oleft'][i] - kcbr['oleft'][i-1] > 5):
            actions.append({'action': 'middle -> left', 'frame': i})
        # middle -> right
        if (kcbr['omiddle'][i-1] - kcbr['omiddle'][i] > 5) and (kcbr['oright'][i] - kcbr['oright'][i-1] > 5):
            actions.append({'action': 'middle -> right', 'frame': i})
        # out <- left
        if (kcbr['oleft'][i-1] - kcbr['oleft'][i] > 5) and (kcbr['omiddle'][i] <= kcbr['omiddle'][i-1]):
            actions.append({'action': 'left -> out', 'frame': i})
        # right -> out
        if (kcbr['oright'][i-1] - kcbr['oright'][i] > 5) and (kcbr['omiddle'][i] <= kcbr['omiddle'][i-1]):
            actions.append({'action': 'right -> out', 'frame': i})

    print(f"{file_name}")
    for action in actions:
        print(f"{action['frame']}: {action['action']}")
    print("\n")


for filename in os.listdir("videos\\"):
    divide_video(filename)

for file_name in os.listdir("divided-videos\\"):
    x = file_name[-5]
    if x == "1":
        bg_mask = bg_mask1
    elif x == "2":
        bg_mask = bg_mask2
    elif x == "3":
        bg_mask = bg_mask3
    track_fishes(file_name, bg_mask)
