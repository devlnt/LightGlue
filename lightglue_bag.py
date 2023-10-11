import rosbag
import yaml

from rosbag.bag import Bag

bag = Bag('2023-10-03-17-10-26.bag', 'r')._get_yaml_info()
print(bag)