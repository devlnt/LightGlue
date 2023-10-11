from rosbag.bag import Bag
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import cv2
import json
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd, numpy_image_to_torch
from lightglue import viz2d
import torch
import numpy as np
from numba import cuda
import matplotlib
from tqdm import tqdm


bridge = CvBridge()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'mps', 'cpu'

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features='superpoint').eval().to(device)

bag = Bag('2023-10-03-17-10-26.bag', 'r')

# camera int
with open("scene_camera.json", "r") as f:
    data = json.load(f)

pupil_camera_matrix = np.array(data["camera_matrix"])
pupil_dist_coeffs = np.array(data["dist_coefs"])

with open("realsense_camera.json", "r") as f:
    data = json.load(f)

rs_camera_matrix = np.array(data["camera_matrix"])
rs_dist_coeffs = np.array(data["dist_coefs"])
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter('fusion.mp4', fourcc, 30, (1280+1088, 1080))
# def run():
world_img = None
rs_img = None
gaze = None
for index, (topic, msg, t) in tqdm(enumerate(bag.read_messages())):
    # print(topic, t, index)
    if topic == '/pupil/world':
        world_img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

        if rs_img is not None and gaze is not None:
            # undistorted_fig, undistorted_axes = plt.subplots(2, 3)
            undistorted_world_img = cv2.undistort(world_img, pupil_camera_matrix, pupil_dist_coeffs)
            undistorted_rs_img = cv2.undistort(rs_img, rs_camera_matrix, rs_dist_coeffs)

            src_gaze = np.array([np.matrix([int(gaze[0]), int(gaze[1])]).transpose()], dtype="float32")
            undistorted_gaze = cv2.undistortPoints(src_gaze, pupil_camera_matrix, pupil_dist_coeffs,
                                                   P=pupil_camera_matrix)

            circled_world_img = cv2.circle(world_img.copy(), [int(gaze[0]), int(gaze[1])], 30, (255, 0, 0), 10)
            circled_undistorted_world_img = cv2.circle(undistorted_world_img.copy(),
                                                       [int(undistorted_gaze[0][0][0]), int(undistorted_gaze[0][0][1])],
                                                       30, (255, 0, 0), 10)

            image0 = numpy_image_to_torch(undistorted_rs_img.copy())
            image1 = numpy_image_to_torch(undistorted_world_img.copy())
            feats0 = extractor.extract(image0.to(device))
            feats1 = extractor.extract(image1.to(device))
            matches01 = matcher({'image0': feats0, 'image1': feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

            kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
            m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

            # %%
            c_m_kpts0 = m_kpts0.cpu().numpy()
            c_m_kpts1 = m_kpts1.cpu().numpy()

            xa, xb = np.polyfit(c_m_kpts0[:, 0], c_m_kpts1[:, 0], 1)
            ya, yb = np.polyfit(c_m_kpts0[:, 1], c_m_kpts1[:, 1], 1)

            xg, yg = int(gaze[0]), int(gaze[1])

            xg1 = (xg - xb) / xa
            yg1 = (yg - yb) / ya

            circled_t_rs_img = cv2.circle(undistorted_rs_img.copy(), [int(xg1), int(yg1)], 30, (255, 0, 0), 10)

            rs_out = np.ones((1080, 1280, 3), dtype=np.uint8) * 255
            world_out = np.ones((1080, 1088, 3), dtype=np.uint8) * 255

            rs_out[:720, :1280, :] = circled_t_rs_img[:, :, :]
            world_out[:1080, :1088, :] = circled_undistorted_world_img[:, :, :]
            results = np.concatenate((rs_out, world_out), axis=1)
            results = cv2.cvtColor(results, cv2.COLOR_BGR2RGB)
            writer.write(results)
    if topic == '/camera/color/image_raw':
        rs_img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
    if topic == '/pupil/gaze':
        gaze = msg.data.split(',')
    if topic == '/camera/imu':
        pass
    # if index == 100:
    #     break

writer.release()
# cv2.imwrite('results.png', results)

