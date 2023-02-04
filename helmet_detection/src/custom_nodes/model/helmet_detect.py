"""
Casting classification model.
"""

from typing import Any, Dict

import cv2
import numpy as np
import tensorflow as tf

from peekingduck.pipeline.nodes.node import AbstractNode

IMG_HEIGHT = 180
IMG_WIDTH = 180

class Node(AbstractNode):
   """Initializes and uses a CNN to predict if an image frame shows a normal
   or defective casting.
   """

   def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
      super().__init__(config, node_path=__name__, **kwargs)
      self.model = tf.keras.models.load_model(self.weights_parent_dir)

   def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
      """Reads the image input and returns the predicted class label and
      confidence score.

      Args:
            inputs (dict): Dictionary with key "img".

      Returns:
            outputs (dict): Dictionary with keys "pred_label" and "pred_score".
      """
      bound = inputs["bound"]
      score = 0
      img = cv2.cvtColor(inputs["img"], cv2.COLOR_BGR2RGB)
      img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
      if len(bound) != 0:
            if len(bound) == 1:
                  try:
                        img0 = img[int(bound[0][1]*180/480):int(bound[0][3]*180/480),int(bound[0][0]*180/640):int(bound[0][2]*180/640)]
                        img0 = cv2.resize(img0, (IMG_WIDTH, IMG_HEIGHT))
                        #cv2.imshow('target1', img0)
                        img0 = np.expand_dims(img0, axis=0)
                        predictions0 = self.model.predict(img0)
                        score0 = tf.nn.softmax(predictions0[0])
                        print(self.class_label_map[np.argmax(score0)])
                        if self.class_label_map[np.argmax(score0)] == 'no_helmet':
                              cv2.putText(inputs["img"],'Dangerous(Missing Helmet)',(20,60),cv2.FONT_HERSHEY_PLAIN,2.5,(0,0,255),3)
                        else:
                              cv2.putText(inputs["img"],'Safe(Helmet Detected)',(20,60),cv2.FONT_HERSHEY_PLAIN,2.5,(0,255,0),3)

                  except Exception as e:
                        print(str(e))
            elif len(bound) == 2:
                  try:
                        img0 = img[int(bound[0][1]*180/480):int(bound[0][3]*180/480),int(bound[0][0]*180/640):int(bound[0][2]*180/640)]
                        img0 = cv2.resize(img0, (IMG_WIDTH, IMG_HEIGHT))
                        #cv2.imshow('target1', img0)
                        img0 = np.expand_dims(img0, axis=0)
                        predictions0 = self.model.predict(img0)
                        score0 = tf.nn.softmax(predictions0[0])
                        print(self.class_label_map[np.argmax(score0)])

                        img1 = img[int(bound[1][1]*180/480):int(bound[1][3]*180/480),int(bound[1][0]*180/640):int(bound[1][2]*180/640)]
                        img1 = cv2.resize(img1, (IMG_WIDTH, IMG_HEIGHT))
                        #cv2.imshow('target2', img1)
                        img1 = np.expand_dims(img1, axis=0)
                        predictions1 = self.model.predict(img1)
                        score1 = tf.nn.softmax(predictions1[0])
                        print(self.class_label_map[np.argmax(score1)])

                        if self.class_label_map[np.argmax(score0)] == 'no_helmet' or self.class_label_map[np.argmax(score1)] == 'no_helmet':
                              cv2.putText(inputs["img"],'Dangerous(Missing Helmet)',(20,60),cv2.FONT_HERSHEY_PLAIN,2.5,(0,0,255),3)
                        else:
                              cv2.putText(inputs["img"],'Safe(Helmet Detected)',(20,60),cv2.FONT_HERSHEY_PLAIN,2.5,(0,255,0),3)

                        
                  except Exception as e:
                        print(str(e))
            else:
                  try:
                        img0 = img[int(bound[0][1]*180/480):int(bound[0][3]*180/480),int(bound[0][0]*180/640):int(bound[0][2]*180/640)]
                        img0 = cv2.resize(img0, (IMG_WIDTH, IMG_HEIGHT))
                        #cv2.imshow('target1', img0)
                        img0 = np.expand_dims(img0, axis=0)
                        predictions0 = self.model.predict(img0)
                        score0 = tf.nn.softmax(predictions0[0])
                        print(self.class_label_map[np.argmax(score0)])

                        img1 = img[int(bound[1][1]*180/480):int(bound[1][3]*180/480),int(bound[1][0]*180/640):int(bound[1][2]*180/640)]
                        img1 = cv2.resize(img1, (IMG_WIDTH, IMG_HEIGHT))
                        #cv2.imshow('target2', img1)
                        img1 = np.expand_dims(img1, axis=0)
                        predictions1 = self.model.predict(img1)
                        score1 = tf.nn.softmax(predictions1[0])
                        print(self.class_label_map[np.argmax(score1)])

                        img2 = img[int(bound[2][1]*180/480):int(bound[2][3]*180/480),int(bound[2][0]*180/640):int(bound[2][2]*180/640)]
                        img2 = cv2.resize(img2, (IMG_WIDTH, IMG_HEIGHT))
                        #cv2.imshow('target3', img2)
                        img2 = np.expand_dims(img2, axis=0)
                        predictions2 = self.model.predict(img2)
                        score2 = tf.nn.softmax(predictions2[0])
                        print(self.class_label_map[np.argmax(score2)])

                        if self.class_label_map[np.argmax(score0)] == 'no_helmet' or self.class_label_map[np.argmax(score1)] == 'no_helmet' or self.class_label_map[np.argmax(score2)] == 'no_helmet':
                              cv2.putText(inputs["img"],'Dangerous(Missing Helmet)',(20,60),cv2.FONT_HERSHEY_PLAIN,2.5,(0,0,255),3)
                        else:
                              cv2.putText(inputs["img"],'Safe(Helmet Detected)',(20,60),cv2.FONT_HERSHEY_PLAIN,2.5,(0,255,0),3)

                        
                  except Exception as e:
                        print(str(e))
      else:
           return


      return {
            "pred_label": self.class_label_map[np.argmax(score)],     # node outputs
            "pred_score": 100.0 * np.max(score),
      }
      