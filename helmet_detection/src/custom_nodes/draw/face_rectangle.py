"""
Custom node to show object detection scores
"""

from typing import Any, Dict, List, Tuple
import cv2
from peekingduck.pipeline.nodes.abstract_node import AbstractNode

YELLOW = (0, 255, 255)        # in BGR format, per opencv's convention
  
def map_bbox_to_image_coords(
   bbox: List[float], image_size: Tuple[int, int]
) -> List[int]:
   """This is a helper function to map bounding box coords (relative) to
   image coords (absolute).
   Bounding box coords ranges from 0 to 1
   where (0, 0) = image top-left, (1, 1) = image bottom-right.

   Args:
      bbox (List[float]): List of 4 floats x1, y1, x2, y2
      image_size (Tuple[int, int]): Width, Height of image

   Returns:
      List[int]: x1, y1, x2, y2 in integer image coords
   """
   width, height = image_size[0], image_size[1]
   x1, y1, x2, y2 = bbox
   x1 *= width*0.9
   x2 *= width*1.1
   y1 *= height*0.2
   y2 *= height*1.2
   return int(x1), int(y1), int(x2), int(y2)


class Node(AbstractNode):
   """This is a template class of how to write a node for PeekingDuck,
      using AbstractNode as the parent class.
      This node draws scores on objects detected.

   Args:
      config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
   """

   def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
      """Node initializer

      Since we do not require any special setup, it only calls the __init__
      method of its parent class.
      """
      super().__init__(config, node_path=__name__, **kwargs)

   def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
      """This method implements the display score function.
      As PeekingDuck iterates through the CV pipeline, this 'run' method
      is called at each iteration.

      Args:
            inputs (dict): Dictionary with keys "img", "bboxes", "bbox_scores"

      Returns:
            outputs (dict): Empty dictionary
      """

      # extract pipeline inputs and compute image size in (width, height)
      img = inputs["img"]
      bboxes = inputs["bboxes"]
      img_size = (img.shape[1], img.shape[0])  # width, height
      bound = []

      for i, bbox in enumerate(bboxes):
         # for each bounding box:
         #   - compute (x1, y1) top-left, (x2, y2) bottom-right coordinates
         #   - convert score into a two decimal place numeric string
         #   - draw score string onto image using opencv's putText()
         #     (see opencv's API docs for more info)
         x1, y1, x2, y2 = map_bbox_to_image_coords(bbox, img_size)
         bound.append([x1, y1, x2, y2])
         start_point = (x1,y1)
         end_point = (x2,y2)
         thickness = 2
         color = (255, 0, 0)
         img = cv2.rectangle(img, start_point, end_point, color, thickness)
    
      if bound == []:
        bound = [0,0,1,1]
        
      return {"bound":bound}               # node outputs