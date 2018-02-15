# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""SqueezeDet Demo.

In image detection mode, for a given image, detect objects and draw bounding
boxes around them. In video detection mode, perform real-time detection on the
video stream.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'mode', 'image', """'image' or 'video'.""")
tf.app.flags.DEFINE_string(
    'checkpoint', './data/model_checkpoints/squeezeDet/model.ckpt-87000',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'input_path', './data/sample.png',
    """Input image or video to be detected. Can process glob input such as """
    """./data/00000*.png.""")
tf.app.flags.DEFINE_string(
    'out_dir', './data/out/', """Directory to dump output image or video.""")
tf.app.flags.DEFINE_string(
    'demo_net', 'squeezeDet', """Neural net architecture.""")
tf.app.flags.DEFINE_string(
    'out_csv', 'squeeze_th0.1_N512_pretrainxxx.csv', """CSV file to dump output detection results""")

flag_random = True
flag_load_random = False # follow the previous windows
flag_load_random_file = ""
dictJson = {}
if flag_load_random:
    import json
    with open('/home/cyeh/PycharmProjects/readCSVtoJson/data_det_random_sqeeze_kitti_999999_N512_P0005_v3.json', 'r') as f:
        dictJson = json.load(f)
def video_demo():
  """Detect videos."""

  cap = cv2.VideoCapture(FLAGS.input_path)

  # Define the codec and create VideoWriter object
  # fourcc = cv2.cv.CV_FOURCC(*'XVID')
  # fourcc = cv2.cv.CV_FOURCC(*'MJPG')
  # in_file_name = os.path.split(FLAGS.input_path)[1]
  # out_file_name = os.path.join(FLAGS.out_dir, 'out_'+in_file_name)
  # out = cv2.VideoWriter(out_file_name, fourcc, 30.0, (375,1242), True)
  # out = VideoWriter(out_file_name, frameSize=(1242, 375))
  # out.open()

  assert FLAGS.demo_net == 'squeezeDet' or FLAGS.demo_net == 'squeezeDet+', \
      'Selected nueral net architecture not supported: {}'.format(FLAGS.demo_net)

  """use random window"""


  with tf.Graph().as_default():
    # Load model
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
    print("LOAD_PRETRAINED_MODEL:")
    print(mc.LOAD_PRETRAINED_MODEL)

    saver = tf.train.Saver(model.model_params)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      saver.restore(sess, FLAGS.checkpoint)

      times = {}
      count = 0
      while cap.isOpened():
        t_start = time.time()
        count += 1
        out_im_name = os.path.join(FLAGS.out_dir, str(count).zfill(6)+'.jpg')

        # Load images from video and crop
        ret, frame = cap.read()
        if ret==True:
          # crop frames
          frame = frame[500:-205, 239:-439, :]
          im_input = frame.astype(np.float32) - mc.BGR_MEANS
        else:
          break

        t_reshape = time.time()
        times['reshape']= t_reshape - t_start

        # Detect
        det_boxes, det_probs, det_class = sess.run(
            [model.det_boxes, model.det_probs, model.det_class],
            feed_dict={model.image_input:[im_input]})

        t_detect = time.time()
        times['detect']= t_detect - t_reshape
        
        # Filter
        final_boxes, final_probs, final_class = model.filter_prediction(
            det_boxes[0], det_probs[0], det_class[0])

        keep_idx    = [idx for idx in range(len(final_probs)) \
                          if final_probs[idx] > mc.PLOT_PROB_THRESH]
        final_boxes = [final_boxes[idx] for idx in keep_idx]
        final_probs = [final_probs[idx] for idx in keep_idx]
        final_class = [final_class[idx] for idx in keep_idx]

        t_filter = time.time()
        times['filter']= t_filter - t_detect

        # Draw boxes

        # TODO(bichen): move this color dict to configuration file
        cls2clr = {
            'car': (255, 191, 0),
            'cyclist': (0, 191, 255),
            'pedestrian':(255, 0, 191)
        }
        _draw_box(
            frame, final_boxes,
            [mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
                for idx, prob in zip(final_class, final_probs)],
            cdict=cls2clr
        )

        t_draw = time.time()
        times['draw']= t_draw - t_filter

        cv2.imwrite(out_im_name, frame)
        # out.write(frame)

        times['total']= time.time() - t_start

        # time_str = ''
        # for t in times:
        #   time_str += '{} time: {:.4f} '.format(t[0], t[1])
        # time_str += '\n'
        time_str = 'Total time: {:.4f}, detection time: {:.4f}, filter time: '\
                   '{:.4f}'. \
            format(times['total'], times['detect'], times['filter'])

        print (time_str)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  # Release everything if job is finished
  cap.release()
  # out.release()
  cv2.destroyAllWindows()

flag_CSV = True

def image_demo():
  """Detect image."""

  assert FLAGS.demo_net == 'squeezeDet' or FLAGS.demo_net == 'squeezeDet+', \
      'Selected nueral net architecture not supported: {}'.format(FLAGS.demo_net)

  if flag_CSV:
      # csv = open("squeeze_th0.1_N512_v2.csv", "w") # squeeze, checkpoint999, random, plot > 0.005
      csv = open(FLAGS.out_csv, "w")  # squeeze, checkpoint999, random, plot > 0.005
      columnTitleRow = "xmin,ymin,xmax,ymax,Frame,Label,Preview URL,confidence,random,y_loc,win_sizeX,win_sizeY\n"
      csv.write(columnTitleRow)


  with tf.Graph().as_default():
    # Load model
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

    saver = tf.train.Saver(model.model_params)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      saver.restore(sess, FLAGS.checkpoint)
      cnt = 0
      FPS = 0
      FPS10 = 0

      for f in glob.iglob(FLAGS.input_path):
        im = cv2.imread(f)
        photo_name = f.split('/')[-1]
        if flag_random:
            """random"""
            randomX = random.randint(0, im.shape[1]-mc.IMAGE_WIDTH)
            randomY = 400 # for udacity
            randomY = int((im.shape[0] - mc.IMAGE_HEIGHT)/2)
            # print (--------------------------)
            # print (flag_load_random)
            # if flag_load_random:
            #     randomX = dictJson[photo_name]["random"][0]

            im = im[randomY:randomY + mc.IMAGE_HEIGHT, randomX:randomX + mc.IMAGE_WIDTH, :]
        else:
            """center"""
            im = im[400:400 + mc.IMAGE_HEIGHT, 300:300 + mc.IMAGE_WIDTH]

        im = im.astype(np.float32, copy=False)
        # im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
        """crop image"""

        input_image = im - mc.BGR_MEANS

        start = time.time()
        # Detect
        det_boxes, det_probs, det_class = sess.run(
            [model.det_boxes, model.det_probs, model.det_class],
            feed_dict={model.image_input:[input_image]})
        print (mc.PLOT_PROB_THRESH)
        cnt = cnt + 1
        end = time.time()
        FPS = (1 / (end - start))
        FPS10 = FPS10 + FPS
        # print ("FPS: " + str(FPS))
        if cnt % 10 == 0:
            print ("FPS(mean), detection: " + str(FPS10/10))
            FPS10 = 0

        # Filter
        final_boxes, final_probs, final_class = model.filter_prediction(
            det_boxes[0], det_probs[0], det_class[0])

        keep_idx    = [idx for idx in range(len(final_probs)) \
                          if final_probs[idx] > mc.PLOT_PROB_THRESH]
        final_boxes = [final_boxes[idx] for idx in keep_idx]
        final_probs = [final_probs[idx] for idx in keep_idx]
        final_class = [final_class[idx] for idx in keep_idx]

        # TODO(bichen): move this color dict to configuration file
        cls2clr = {
            'car': (255, 191, 0),
            'cyclist': (0, 191, 255),
            'pedestrian':(255, 0, 191)
        }

        # Draw boxes
        _draw_box(
            im, final_boxes,
            [mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
                for idx, prob in zip(final_class, final_probs)],
            cdict=cls2clr,
        )
        # print (final_boxes)
        print (final_probs)
        # print (final_class)
        # print (mc.CLASS_NAMES)
        # im2 = cv2.imread(f)

        if flag_CSV:

            for bbox_idx in range(len(final_boxes)):
                bbox = bbox_transform(final_boxes[bbox_idx])

                xmin, ymin, xmax, ymax = [int(b) for b in bbox]
                csv.write(str(xmin+randomX))
                csv.write(",")
                csv.write(str(ymin+randomY))
                csv.write(",")
                csv.write(str(xmax+randomX))
                csv.write(",")
                csv.write(str(ymax+randomY))
                csv.write(",")
                "file name"
                csv.write(photo_name)
                csv.write(",")
                "label"
                csv.write(mc.CLASS_NAMES[final_class[bbox_idx]])
                csv.write(",")
                csv.write(",")
                "confidence"
                csv.write(str(final_probs[bbox_idx]))
                csv.write(",")
                "random selected window, x:"
                csv.write(str(randomX))
                csv.write(",")
                "random selected window, Y:"
                csv.write(str(randomY))
                csv.write(",")
                "random selected window, sizeX, X:"
                csv.write(str(mc.IMAGE_WIDTH))
                csv.write(",")
                "random selected window, sizeY, Y:"
                csv.write(str(mc.IMAGE_HEIGHT))
                csv.write(",")
                csv.write("\n")
                # debug: offset random window size
                # cv2.rectangle(im2, (xmin + randomX, ymin + randomY), (xmax + randomX, ymax + randomY), (0, 255, 0), 1)
            if len(final_boxes) == 0:
                print ("No detection: " + photo_name)
                csv.write(",")
                csv.write(",")
                csv.write(",")
                csv.write(",")
                "file name"
                csv.write(photo_name)
                csv.write(",")
                "label"
                csv.write(",")
                csv.write(",")
                "confidence"
                csv.write(",")
                "random selected window, x:"
                csv.write(str(randomX))
                csv.write(",")
                "random selected window, Y:"
                csv.write(str(randomY))
                csv.write(",")
                "random selected window, sizeX, X:"
                csv.write(str(mc.IMAGE_WIDTH))
                csv.write(",")
                "random selected window, sizeY, Y:"
                csv.write(str(mc.IMAGE_HEIGHT))
                csv.write(",")
                csv.write("\n")
            # for bbox, label in zip(final_boxes, label_list):
            #
            #
            #     xmin, ymin, xmax, ymax = [int(b) for b in bbox]
            #
            #     l = label.split(':')[0]  # text before "CLASS: (PROB)"
            #     if cdict and l in cdict:
            #         c = cdict[l]
            #     else:
            #         c = color
            #
            #     # draw box
            #     cv2.rectangle(im, (xmin, ymin), (xmax, ymax), c, 1)

        file_name = os.path.split(f)[1]
        out_file_name = os.path.join(FLAGS.out_dir, 'out_'+file_name)
        if cnt < 20:
            cv2.imwrite(out_file_name, im)
            print('Image detection output saved to {}'.format(out_file_name))
        else:
            print('(Skip)Image detection output saved to {}'.format(out_file_name))

        # cv2.imwrite(os.path.join(FLAGS.out_dir, 'out_offset_'+file_name), im2)



def main(argv=None):
  if not tf.gfile.Exists(FLAGS.out_dir):
    tf.gfile.MakeDirs(FLAGS.out_dir)
  if FLAGS.mode == 'image':
    image_demo()
  else:
    video_demo()

if __name__ == '__main__':
    tf.app.run()
