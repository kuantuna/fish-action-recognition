import cv2
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
            # xx = cnt * 100/frames
            # print(int(xx), '%')
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


for filename in os.listdir("videos/"):
    divide_video(filename)
