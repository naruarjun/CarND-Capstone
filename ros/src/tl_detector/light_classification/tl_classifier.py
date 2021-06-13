import tensorflow as tf
import numpy as np
from styx_msgs.msg import TrafficLight
import cv2 
from datetime import datetime
import os


CLASS = ['None', 'Green', 'Yellow', 'Red']

class TLClassifier(object):
    def __init__(self):
        self.detection_graph = self.load_graph("light_classification/model/ssd_mobilenet_v1_frozen_graph.pb")
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Boxes, Scores and Classes
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.sess = tf.Session(graph = self.detection_graph)

    def load_graph(self,graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    def to_image_coords(self, boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1], so it converts back to the original coordinate.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width

        return box_coords

    def draw_boxes(self, image, boxes, classes, scores):
        """Draw bounding boxes on the image"""
        for i in range(len(boxes)):
            top, left, bot, right = boxes[i, ...]
            cv2.rectangle(image, (left, top), (right, bot), (255,0,0), 3)
            text = CLASS[classes[i]] + ': ' + str(int(scores[i]*100)) + '%'
            cv2.putText(image , text, (left, int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,0,0), 1, cv2.LINE_AA)

    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # Preprocess input
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        (im_width, im_height, _) = image_rgb.shape
        image_np = np.expand_dims(image_rgb, axis=0)

        # Detection
        with self.detection_graph.as_default():
            (boxes, scores, classes) = self.sess.run(
                [self.detection_boxes, self.detection_scores,
                 self.detection_classes],
                feed_dict={self.image_tensor: image_np})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        # Filter based on confidence
        conf_threshold = 0.5
        boxes, scores, classes = self.filter_boxes(conf_threshold, boxes, scores, classes)

        # Output the image
        output_images = False # make this True to output inference images
        if output_images:
            image = np.dstack((image[:, :, 2], image[:, :, 1], image[:, :, 0]))
            width, height = image.shape[1], image.shape[0]
            box_coords = self.to_image_coords(boxes, height, width)
            self.draw_boxes(image, box_coords, classes, scores)
            timestr = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            self.out_dir = '/home/naruarjun/udacity-dev/submission/CarND-Capstone/ros/src/tl_detector/light_classification/images/'
            filename = os.path.join(self.out_dir, 'image_' + timestr + '.jpg')
            im_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, im_bgr)

        if len(scores)>0:
            tl_class = int(classes[np.argmax(scores)])
        else:
            tl_class = 4 
        if tl_class==1:
            return TrafficLight.GREEN
        elif tl_class == 2:
             return TrafficLight.RED # Return RED for YELLOW as well
        elif tl_class == 3:
             return TrafficLight.RED

        return TrafficLight.GREEN # Return GREEN for UNKNOWN
