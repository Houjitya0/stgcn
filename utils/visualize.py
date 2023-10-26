import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def draw_adjancy_matrix(A, save_folder, file_name):
    sns.heatmap(A, annot=False, cmap='Blues', fmt=".0f", cbar=False)
    # sns.heatmap(A, annot=True, cmap='Blues',  fmt=".0f", cbar=False, annot_kws={"fontsize":18})
    plt.title(f'adjancy matrix')
    plt.savefig(f'{save_folder}/{file_name}.png')


# input (v, c)
# return img of skeleton
# input (v, c)
# return img of skeleton
def coco_normarized_one_person_img(img, keypoints):
    
    orange = [18, 132, 239]
    green = [12, 213, 56]
    blue = [245, 151, 58]
    red = [0, 0, 255]
    skeleton_links = [[0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6], [5, 7], [6, 8],  [7, 9], [8, 10], [5,11],  [6, 12], [11, 13], [12, 14], [13, 15], [14, 16], [11, 12], [17, 18]]
    skeleton_links_colors = np.array([blue, blue, blue, blue, blue, blue, green, orange, green, orange, green, orange, green, orange, green, orange, green, red, green, orange, green, orange, green, orange, green, red, green, orange, green, orange, green, orange, green, [0, 0, 0]])
    keypoint_colors = np.array([blue, blue, blue, blue, blue, red, red, green, orange, green, orange, red, red, green, orange, green, orange, blue, blue, blue, [0, 0, 0]])
    

    line_width = 2
    radius = 3


    for sk_id, sk in enumerate(skeleton_links):

        pos1 = (int(keypoints[sk[0], 0]), int(keypoints[sk[0], 1]))
        pos2 = (int(keypoints[sk[1], 0]), int(keypoints[sk[1], 1]))

        color = skeleton_links_colors[sk_id].tolist()
        cv2.line(img, pos1, pos2, color, thickness=line_width)

    for kid, kpt in enumerate(keypoints):


        x_coord, y_coord = int(kpt[0]), int(kpt[1])

        color = keypoint_colors[kid].tolist()
        cv2.circle(img, (int(x_coord), int(y_coord)), radius,
                    color, -1)
        cv2.circle(img, (int(x_coord), int(y_coord)), radius,
                    (255, 255, 255))
            
    return img



# input (v, c)
# return img of skeleton
def coco_one_person_img(img, keypoints):
    
    orange = [18, 132, 239]
    green = [12, 213, 56]
    blue = [245, 151, 58]
    skeleton_links = [[0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6], [5, 7], [6, 8],  [7, 9], [8, 10], [5,11],  [6, 12], [11, 13], [12, 14], [13, 15], [14, 16], [11, 12]]
    skeleton_links_colors = np.array([blue, blue, blue, blue, blue, blue, green, orange, green, orange, green, orange, green, orange, green, orange, green, orange, green, orange, green, orange, green, orange, green, orange, green, orange, green, orange, green, orange, green])
    keypoint_colors = np.array([blue, blue, blue, blue, blue, green, orange, green, orange, green, orange, green, orange, green, orange, green, orange, blue])
    

    line_width = 2
    radius = 3


    for sk_id, sk in enumerate(skeleton_links):

        pos1 = (int(keypoints[sk[0], 0]), int(keypoints[sk[0], 1]))
        pos2 = (int(keypoints[sk[1], 0]), int(keypoints[sk[1], 1]))

        color = skeleton_links_colors[sk_id].tolist()
        cv2.line(img, pos1, pos2, color, thickness=line_width)

    for kid, kpt in enumerate(keypoints):


        x_coord, y_coord = int(kpt[0]), int(kpt[1])

        color = keypoint_colors[kid].tolist()
        cv2.circle(img, (int(x_coord), int(y_coord)), radius,
                    color, -1)
        cv2.circle(img, (int(x_coord), int(y_coord)), radius,
                    (255, 255, 255))
            
    return img

# input (t, v, c)
# return none but create video
def coco_normarized_one_person_video(t_keypoints, video_path, width=255, height=255, fps=30, label=None):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    action_label = ['no talk', 'no active', 'active']
    for keypoints in (t_keypoints):
            
        img = np.ones((255, 255, 3), dtype=np.uint8) * 255
        
        if (label is not None):
            cv2.putText(img, action_label[label], (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.5,(0,255,0,),1,cv2.LINE_AA)
        img = coco_normarized_one_person_img(img, keypoints)
        out.write(img)

    out.release()

# input (t, v, c)
# return none but create video
def coco_one_person_video(t_keypoints, video_path, width=255, height=255, fps=30, label=None):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    action_label = ['no talk', 'no active', 'active']
    for keypoints in (t_keypoints):
            
        img = np.ones((255, 255, 3), dtype=np.uint8) * 255
        
        if (label is not None):
            cv2.putText(img, action_label[label], (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.5,(0,255,0,),1,cv2.LINE_AA)
        img = coco_one_person_img(img, keypoints)
        out.write(img)

    out.release()
    

# input (t, m, v, c)
def coco_two_person_video(t_keypoints, input_video_path, output_video_path, width, height, fps, lable=None):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    in_video = cv2.VideoCapture(input_video_path)
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # T
    for f, keypoints in enumerate(t_keypoints):
        in_video.set(cv2.CAP_PROP_POS_FRAMES, f-1)
        ret, img = in_video.read()
        # cv2.putText(img, str(f), (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.5,(0,255,0,),1,cv2.LINE_AA)
        for i, kp in enumerate(keypoints):
            # frame = in_video[f]
            img = coco_one_person_img(img, kp)
            
        out_video.write(img)

    out_video.release()
    
    
def coco_two_person_video_back(t_keypoints, output_video_path, width, height, fps, lable=None):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # T
    for f, keypoints in enumerate(t_keypoints):
        img = np.ones([256, 256, 3], dtype=np.uint8) * 255

        for i, kp in enumerate(keypoints):
            
            # frame = in_video[f]
            img = coco_one_person_img(img, kp)
            print(img)
        out_video.write(img)

    out_video.release()
    
    
def coco_two_person_one_graph(t_keypoints, output_video_path, width, height, fps, lable=None):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # T
    for f, keypoints in enumerate(t_keypoints):
        img = np.ones([256, 256, 3], dtype=np.uint8) * 255

        cv2.circle(img, (255//2, 10), 2, (100, 100, 100), -1)
        cv2.line(img, (255//2, 10), (int(keypoints[0, 0, 0]), int(keypoints[0, 0, 1])), (100, 100, 0), thickness=2)
        cv2.line(img, (255//2, 10), (int(keypoints[1, 0, 0]), int(keypoints[1, 0, 1])), (100, 100, 0), thickness=2)
        
        for i, kp in enumerate(keypoints):
            
            # frame = in_video[f]
            img = coco_one_person_img(img, kp)
        out_video.write(img)

    out_video.release()


    
def Create_Video(coordinates_list, video_path, width, height, fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame_coords in zip(*coordinates_list):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255  

        for i, coords in enumerate(frame_coords):
            color = (0, 255, 0) if i == 0 else (0, 0, 255)  # 1人目は緑色、2人目は赤色
            for coord_pair in coords:
                x, y = int(coord_pair[0]), int(coord_pair[1])
                frame = cv2.circle(frame, (x, y), 2, color, -1)

        out.write(frame)

    out.release()
    
    
def Create_Video2(coordinates_list, input_video_path, output_video_path, width, height, fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    in_video = cv2.VideoCapture(input_video_path)
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # T
    for f, frame_coords in enumerate(coordinates_list):
        frame = in_video[f]

        # M
        for i, coords in enumerate(frame_coords):
            color = (144, 255, 0)
            
            # V
            for coord_pair in coords:
                x, y = int(coord_pair[0]), int(coord_pair[1])
                frame = cv2.circle(frame, (x, y), 2, color, -1)

        out_video.write(frame)

    out_video.release()
    
    
    
def Create_Video3(coordinates_list, input_video_path, output_video_path, width, height, fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    in_video = cv2.VideoCapture(input_video_path)
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # T
    for f, frame_coords in enumerate(coordinates_list):
        in_video.set(cv2.CAP_PROP_POS_FRAMES, f-1)
        ret, frame = in_video.read()
        if frame is None:
            break
        
        frame = np.array(frame)
        
        # M
        for i, coords in enumerate(frame_coords):
            color = (144, 255, 0)
            
            # V
            for coord_pair in coords:
                x, y = int(coord_pair[0]), int(coord_pair[1])
                frame = cv2.circle(frame, (x, y), 3, color, -1)

        out_video.write(frame)

    out_video.release()
    
    
    
    
    
    
def Create_Video_One_Person(coordinates, video_path, width, height, fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 動画コーデックの指定
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))  # 動画ファイルの作成

    frame_size = len(coordinates) // 2

    
    for i, frame_coords in enumerate(coordinates):
        if (i < frame_size):
            continue
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255  # 空の黒いフレームを作成

        # if (i < (frame_size // 2)):
        #     color = (0, 255, 0)
        # else:
        #     color = (0, 0, 255)

        color = (0, 0, 255)

        
        for coord_pair in frame_coords:
            x, y = coord_pair
            x, y = int(x), int(y)
            frame = cv2.circle(frame, (x, y), 2, color, -1)  # 座標に緑色の円を描画

        out.write(frame)  # フレームを動画に書き込む
        

    out.release()  # 動画ファイルをクローズ
    
    

    
    

    