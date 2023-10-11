from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from rostopic import get_topic_type
import torch
from cv_bridge import CvBridge
import rospy
import cv2
import time
from pathlib import Path
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd, numpy_image_to_torch
from lightglue import viz2d
import torch
import matplotlib.pyplot as plt
import numpy as np


torch.set_grad_enabled(False)
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'mps', 'cpu'

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features='superpoint').eval().to(device)

bridge = CvBridge()

rs_img = None
# pupil_img = None
lock = False


def matching(pupil_img, stamp):
    global rs_img
    # global pupil_img
    global lock
    # print(pupil_img.ndim)
    if rs_img is None:
        print('rs_img none')
        return
    elif pupil_img is None:
        print('pupil_img none')
        return
    # if lock:
    #     return
    # print(lock)
    # lock = True
    # print('lock')
    print(stamp)
    image0 = numpy_image_to_torch(rs_img.copy())
    image1 = numpy_image_to_torch(pupil_img.copy())
    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

    kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    axes = viz2d.plot_images([image0, image1])
    viz2d.plot_matches(m_kpts0, m_kpts1, color='lime', lw=0.2)
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)

    # kpc0, kpc1 = viz2d.cm_prune(matches01['prune0']), viz2d.cm_prune(matches01['prune1'])
    # viz2d.plot_images([image0, image1])
    # viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
    fig = plt.gcf()
    fig.canvas.draw()
    # plt.savefig('test.png')
    img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    publish_img_msg = bridge.cv2_to_imgmsg(img, "bgr8")
    matching_pub.publish(publish_img_msg)
    # lock = False
    # print('Unlock')
    # viz2d.save_plot('test.png')


def pupil_callback(data):
    # global pupil_img
    # pass
    print('pupil', data.header.stamp)
    pupil_img = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
    # numpy_image_to_torch(pupil_img)
    # print(pupil_img)
    # print(pupil_img.ndim)
    matching(pupil_img, data.header.stamp)
    # print("tmp/{}.png".format(time.time_ns()))
    # cv2.imwrite("tmp/{}.png".format(time.time_ns()), im)
    # pupil_sub.unregister()
    # rospy.spin()


def rs_callback(data):
    global rs_img
    # print(data.header)
    # print('sense', data.header.stamp)
    rs_img = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
    # rs_sub.unregister()
    # rospy.spin()
    # print("tmp/{}.png".format(time.time_ns()))
    # cv2.imwrite("tmp/{}.png".format(time.time_ns()), im)


# def main():
rospy.init_node("fusion", anonymous=True)

matching_pub = rospy.Publisher("/fusion/matching", Image, queue_size=10)

rs_topic = "/camera/color/image_raw"
pupil_topic = "/pupil/world"
rs_image_type, rs_image_topic, _ = get_topic_type(rs_topic, blocking=True)
pupil_image_type, pupil_image_topic, _ = get_topic_type(pupil_topic, blocking=True)
print(rs_image_type, rs_image_topic, _)
print(pupil_image_type, pupil_image_topic, _)

rs_sub = rospy.Subscriber(rs_image_topic, Image, rs_callback, queue_size=1)
pupil_sub = rospy.Subscriber(pupil_image_topic, Image, pupil_callback, queue_size=1)
rospy.spin()


# if __name__ == '__main__':
#     main()
