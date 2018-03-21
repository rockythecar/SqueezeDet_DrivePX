#!/usr/bin/env python

# This code poses as the rl_state being published by sensor data
# Author = David Isele
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import rospy
import numpy as np
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
#from DeepRL.dqn.SumoCarMDP import *
from honda_msgs.msg import Object
from honda_msgs.msg import ObjectVector
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

import matplotlib.pyplot as plt
import tensorflow as tf

import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# for squeeze


import cv2
import time
import sys
import os
import glob

import numpy as np
import tensorflow as tf

from config import *
from train import _draw_box
from nets import *

import time
import random
from utils.util import sparse_to_dense, bgr_to_rgb, bbox_transform

"""
rosbag play cam_det2.bag
python
"""

# demo.py
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
	'mode', 'image', """'image' or 'video'.""")
tf.app.flags.DEFINE_string(
	'checkpoint', './data/model_checkpoints/squeezeDet/5th/model.ckpt-252000',
	"""Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
	'input_path', './data/sample.png',
	"""Input image or video to be detected. Can process glob input such as """
	"""./data/00000*.png.""")
tf.app.flags.DEFINE_string(
	'out_dir', './data/out/', """Directory to dump output image or video.""")
tf.app.flags.DEFINE_string(
	'demo_net', 'squeezeDet_ros', """Neural net architecture.""")
if FLAGS.demo_net == 'squeezeDet':
	mc = kitti_squeezeDet_config()
	mc.BATCH_SIZE = 1
	# model parameters will be restored from checkpoint
	mc.LOAD_PRETRAINED_MODEL = False
	model = SqueezeDet(mc, FLAGS.gpu)
elif FLAGS.demo_net == 'squeezeDet+':
	mc = kitti_squeezeDetPlus_config()
	mc.BATCH_SIZE = 1
	mc.LOAD_PRETRAINED_MODEL = False
	model = SqueezeDetPlus(mc, FLAGS.gpu)
elif FLAGS.demo_net == 'squeezeDet_ros':
	mc = kitti_squeezeDet_config()
	mc.BATCH_SIZE = 1
	# model parameters will be restored from checkpoint
	mc.LOAD_PRETRAINED_MODEL = False
	model = SqueezeDet_v2_ros(mc, FLAGS.gpu)

class image_converter():
	def __init__(self):
		self.image_pub = rospy.Publisher("image_topic_2", Image)
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/center/camera/image_color", Image, self.callback)


		# demo.py

		self._saver = tf.train.Saver(model.model_params)
		self._sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		print(FLAGS.checkpoint)
		self._saver = self._saver.restore(self._sess, FLAGS.checkpoint)

		# calculate FPS
		self.cnt = 0
		self.FPS = 0
		self.FPS10 = 0
		self.cnt_2 = 0
		self.FPS_2 = 0
		self.FPS10_2 = 0
		self.sec_2 = 0
		self.sec_nms = 0
		self.sec_crop = 0

		self.BGR_MEANS = np.float32(np.zeros((384,1248,3)))
		for ii in range(384):
			for jj in range(1248):
				self.BGR_MEANS[ii][jj] = mc.BGR_MEANS[0][0]

		# self.data = Float32MultiArray()
		# print (self.data)
		# # self.data.data = [0]*9
		# tmp = np.asarray([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
		# 		[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,   0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.]])
		# dim = np.shape(tmp)
		# self.data.data = tmp.ravel()
		# self.data.layout.dim.append(MultiArrayDimension())
		# self.data.layout.dim.append(MultiArrayDimension())
		# self.data.layout.dim[0].label = "height"
		# self.data.layout.dim[1].label = "width"
		# self.data.layout.dim[0].size = dim[0]
		# self.data.layout.dim[1].size = dim[1]
		# self.data.layout.dim[0].stride = dim[0]*dim[1]
		# self.data.layout.dim[1].stride = dim[1]
		# self.data.layout.data_offset = 0
		# print ("FIXED DATA", self.data)
		#
		# ## SETUP ROS NODES
		# rospy.init_node('dummy_node', anonymous=False)
        #
		# rospy.loginfo("To stop vecs_to_gonogo CTRL + C")
		# rospy.on_shutdown(self.shutdown)
        #
		# self.r = rospy.Rate(10) # 10hz
        #
		# # PUBLISHER
		# # self.pub = rospy.Publisher('/go_nogo', ObjectVector, queue_size=1)
		# self.pub = rospy.Publisher('/net_input', Float32MultiArray, queue_size=1)
        #
		# rospy.sleep(0.01)
		#
		# while not rospy.is_shutdown():
		# 	self.r.sleep()
		# 	self.pub.publish(self.data) # numpy in ros only handles 1d arrays

	def callback(self, data):
		self.cnt = self.cnt + 1
		start_2 = time.time()
		start_3 = time.time()
		# load : 2*0.00136515457666 sec
		try:
			input_raw_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		(rows, cols, channels) = input_raw_image.shape
		# if cols > 60 and rows > 60:
		# 	cv2.circle(cv_image, (50, 50), 10, 255)

		# demo.py
		# im = cv_image
		# print(input_raw_image.shape)
		input_image = input_raw_image[int((rows-mc.IMAGE_HEIGHT)/2)-200:-200+int((rows-mc.IMAGE_HEIGHT)/2)+mc.IMAGE_HEIGHT, int((cols-mc.IMAGE_WIDTH)/2):int((cols-mc.IMAGE_WIDTH)/2)+mc.IMAGE_WIDTH,:]
		# print(int((rows-mc.IMAGE_HEIGHT)/2)-200,int((cols-mc.IMAGE_WIDTH)/2))
		end_3 = time.time()
		sec_3 = (end_3 - start_3)
		# print("f:crop image")
		# print(sec_3)
		self.sec_crop = self.sec_crop + sec_3
		# print ("FPS: " + str(FPS))
		if self.cnt % 10 == 0:
			print("sec(crop image): " + str(self.sec_crop / 10))
			self.sec_crop = 0


		# print(type(im))
		# print(type(self.BGR_MEANS))

		# 2*9.16483911381e-04 sec
		# im = im.astype(np.float32, copy=False)
		# print (im.dtype)
		# print(self.BGR_MEANS.dtype)
		# print("-------")

		# end_3 = time.time()
		# sec_3 = (end_3 - start_3)
		# print("g:im.astype")
		# print(sec_3)
		# start_3 = time.time()

		# im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
		# print(mc.BGR_MEANS.dtype)
		# print(mc.BGR_MEANS)

		# input_image = im - mc.BGR_MEANS
		# input_image = input_image - self.BGR_MEANS
		# input_image = im
		# print(im.shape)
		# print(self.BGR_MEANS.shape)
		# print (input_image.dtype)
		# print(mc.BGR_MEANS.dtype)

		# end_3 = time.time()
		# sec_3 = (end_3 - start_3)
		# print("e:im - mc.BGR_MEANS")
		# print(sec_3)
		# start_3 = time.time()

		start = time.time()
		# Detect
		# det_boxes, det_probs, det_class = self._sess.run(
		# 	[model.det_boxes, model.det_probs, model.det_class],
		# 	feed_dict={model.image_input: [input_image]})
		# print(model.det_boxes)
		# print(model.image_input)
		# print(model.ph_image_input)
		# test = tf.get_default_graph().get_tensor_by_name("image_input:0")
		# print(test)  # Tensor("example:0", shape=(2, 2), dtype=float32)
		det_boxes, det_probs, det_class = self._sess.run(
			[model.det_boxes, model.det_probs, model.det_class],
			feed_dict={model.ph_image_input: [input_image]})
		# print(model.ph_raw_image_input)
		# det_boxes, det_probs, det_class = self._sess.run(
		# 	[model.det_boxes, model.det_probs, model.det_class],
		# 	feed_dict={model.ph_ori_image_input: [input_raw_image]})




		end = time.time()
		self.FPS = (1 / (end - start))
		# print("Detect:'")
		# print((end - start))
		# print(self.FPS)
		self.FPS10 = self.FPS10 + self.FPS
		# print ("FPS: " + str(FPS))
		if self.cnt % 10 == 0:
			print("FPS(detection): " + str(self.FPS10 / 10))
			print("sec(detection): " + str(1/self.FPS10 * 10))
			self.FPS10 = 0



		# Filter
		# 0.208640098572 sec for 512 TOP_N_DETECTION
		# 0.0111668109894 sec for 64 TOP_N_DETECTION
		# 0.005 for 32

		start_3 = time.time()
		# # filter
		final_boxes, final_probs, final_class = model.filter_prediction(
			det_boxes[0], det_probs[0], det_class[0])
		end_3 = time.time()
		sec_3 = (end_3 - start_3)
		# print("3:Filter")
		# print(sec_3)
		self.sec_nms = self.sec_nms + sec_3
		# print ("FPS: " + str(FPS))
		if self.cnt % 10 == 0:
			print("sec(filter): " + str(self.sec_nms / 10))
			self.sec_nms = 0

		# 4.50611114502e-05 sec
		"""Set plot threshold"""
		start_3 = time.time()

		keep_idx = [idx for idx in range(len(final_probs)) \
					if final_probs[idx] > mc.PLOT_PROB_THRESH]
		final_boxes = [final_boxes[idx] for idx in keep_idx]
		final_probs = [final_probs[idx] for idx in keep_idx]
		final_class = [final_class[idx] for idx in keep_idx]
		end_3 = time.time()
		sec_3 = (end_3 - start_3)
		# print("final_probs[idx] > mc.PLOT_PROB_THRESH")
		# print(sec_3)
		start_3 = time.time()
		# TODO(bichen): move this color dict to configuration file
		cls2clr = {
			'car': (255, 191, 0),
			'cyclist': (0, 191, 255),
			'pedestrian': (255, 0, 191)
		}

		# Draw boxes
		# 0.00111293792725 sec
		_draw_box(
			input_image, final_boxes,
			[mc.CLASS_NAMES[idx] + ': (%.2f)' % prob \
			 for idx, prob in zip(final_class, final_probs)],
			cdict=cls2clr,
		)
		end_3 = time.time()
		sec_3 = (end_3 - start_3)
		# print("c:_draw_box")
		# print(sec_3)
		start_3 = time.time()

		# file_name = os.path.split(f)[1]
		file_name = "test.png"
		out_file_name = os.path.join(FLAGS.out_dir, 'out_ros_' + file_name)
		cv2.imwrite(out_file_name, input_image)
		# print('Image detection output saved to {}'.format(out_file_name))

		# show image
		# cv2.imshow("Image window", im.astype(np.uint8, copy=False))
		# cv2.waitKey(3)

		# 2*5.80072402954e-04 sec
		try:
			# self.image_pub.publish(self.bridge.cv2_to_imgmsg(im, "16SC3"))
			pass
		except CvBridgeError as e:
			print(e)
		self.cnt_2 = self.cnt_2 + 1
		end_2 = time.time()
		self.FPS_2 = (1 / (end_2 - start_2))
		self.sec_2 = self.sec_2 + (end_2 - start_2)
		self.FPS10_2 = self.FPS10_2 + self.FPS_2
		# print("callback")
		# print(self.FPS_2)
		# print("callback")
		if self.cnt_2 % 10 == 0:
			print("-----FPS(callback): " + str(self.FPS10_2 / 10))
			print("-----Sec(callback): " + str(self.sec_2 / 10))
			self.FPS10_2 = 0
			self.sec_2 = 0

		end_3 = time.time()
		sec_3 = (end_3 - start_3)
		# print("b:save and publish")
		# print(sec_3)
		print('---end-------------------------')




	def shutdown(self):
		rospy.loginfo("Stop dummy_node")
		# rospy.sleep(1)
		return 0


def main(argv=None):
	if not tf.gfile.Exists(FLAGS.out_dir):
		tf.gfile.MakeDirs(FLAGS.out_dir)
	try:
		ic = image_converter()
		rospy.init_node('dummy_node', anonymous=True)
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
	cv2.destroyAllWindows()

if __name__ == '__main__':
	tf.app.run()


