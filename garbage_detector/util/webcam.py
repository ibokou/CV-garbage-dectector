import cv2
import numpy as np


class PredictionAnnotator():
    """
    A helper for displaying the predictions obtained by a Faster RCNN model on a camera live feed.

    Attributes
    --------
    Below mentioned parameters in constructer are directly passed to attributes.

    classes: list[str]
        The list of classes that are annotated.
    colors: list[tuple]
        A list of randomly generated colors for each class

    """

    def __init__(self, classes):
        """
        Parameters
        ----------
        classes: list[str]
            The list of classes that are annotated.
        """
        self.classes = classes
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def __draw_class_name_with_confidence(self, orig_img, pt1, pt2, class_name, score, thickness, offset=3):
        """Helper Function that draws the confidence score and the class name on the image)

        Parameters
        ----------
        orig_img: numpy.ndarray
            Image on which bounding box, confidence and label is drawn

        pt1: tuple[int, int]
            The x1, y1 coordinates of the bounding box.

        pt2: tuple[int, int]
            The x2, y2 coordinates of the bounding box.

        class_name: str
            Name of the class

        score: float
            Confidence score

        thickness: int
            The thickness of the lines

        offset: int
            Offset of text distance to bounding box

        Returns
        -------
        None

        """
        _, height = cv2.getTextSize(
            class_name,
            0,
            fontScale=thickness / 3,
            thickness=thickness
        )[0]

        lies_outside = pt1[1] - height >= 3
        cv2.putText(
            orig_img,
            f'{class_name}:  {score}%',
            (pt1[0], pt1[1] - 5 if lies_outside else + pt1[1] + height + offset-1),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=thickness / 3,
            color=(255, 255, 255),
            thickness=thickness,
            lineType=cv2.LINE_AA
        )

    def __draw_bounding_box(self, orig_img, p1, p2, color, thickness):
        """Helper Function that draws the bounding box obtained by predicition

        Parameters
        ----------
        orig_img: numpy.ndarray
            Image on which bounding box, confidence and label is drawn

        pt1: tuple[int, int]
            The x1, y1 coordinates of the bounding box.

        pt2: tuple[int, int]
            The x2, y2 coordinates of the bounding box.

        color: tuple
            Color of the bounding box.

        thickness: int
            The thickness of the lines

        Returns
        -------
        None
        """
        cv2.rectangle(
            orig_img,
            p1, p2,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA
        )

    def annotate(self, predictions, orig_image, detection_treshold):
        """Annotates the information obtained by model predicition
        on the frame received by the camera live feed.

        Parameters
        ----------
        predictions: dict
            Dictionary that contains coordinates of bounding boxes,
            the associated class (or label) for each bounding box
            and its confidence score.

        orig_img: numpy.ndarray

        detection_treshold: float
            Bounding boxes with a confidence score lower than the threshold
            are not considered and therefore not drawn

        Returns
        -------
        numpy.ndarray
            representing frame from live feed from camera with annotations
        None
        """
        boxes = predictions[0]['boxes'].data.numpy()
        scores = predictions[0]['scores'].data.numpy()

        line_thickness = max(round(sum(orig_image.shape) / 2 * 0.003), 2)
        boxes = boxes[scores >= detection_treshold].astype(np.int32)
        pred_classes = [self.classes[i]
                        for i in predictions[0]['labels'].cpu().numpy()]

        for i, box in enumerate(boxes):
            xy1, xy2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            class_name = pred_classes[i]

            color = self.colors[self.classes.index(class_name)]

            self.__draw_bounding_box(
                orig_image, xy1, xy2, color, line_thickness)
            self.__draw_class_name_with_confidence(
                orig_image, xy1, xy2, class_name, np.trunc(scores[i] * 100)/100, line_thickness)

        return orig_image
