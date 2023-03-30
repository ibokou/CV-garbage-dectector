import cv2
import numpy as np
import torch

from garbage_detector.util.colab import check_is_running_in_colab
from garbage_detector.util.colab.webcam import (bbox_to_bytes, js_to_image,
                                                int_colab_js,
                                                start_video_stream)
from garbage_detector.util.webcam import PredictionAnnotator


class Application:
    """
    A class that is responsible for setting up necessary resources, executing
    necessary code and cleaning up resources afterward, so that the user is
    able to test Faster RCNN detection models with his/her camera live feed.

    Attributes
    --------
    Below mentioned parameters in constructer are directly passed to attributes.

    runs_in_colab: bool
        It indicates whether the jupyter notebook and the code are running locally or 
        remotely in google colab through the browser. This flag is necessary, because accessing
        the camera feed through the browser from python needs to be handled differently.
    annotator:
        A helper object that is instantiated for annotation of the model predictions, meaning 
        displaying the bounding boxes, class names and scores.

    """

    def __init__(self, app_name, model, device, transform, classes, detection_threshold=0.5):
        """
        Parameters
        ----------
        app_name : str
            The name of the application window that is started by OpenCV.
        model : torchvision.models.detection.FasterRCNN
            The model that runs the detection and returns bounding boxes etc.
            of the detected objects.
        device : str
            The name of the device to which the model is attached. It represents the hardware 
            on which the model detection is executed. It is either CPU or GPU (cuda).
        transform: torchvision.transforms.Compose
            The composition of all transformations that need to be applied to the frames of the 
            camera feed before passing it to the model.
        classes: list[str]
            The list of all class as strings that can be detected by the model. It is for translating
            the output of the model to the user as text, so that it can be understood directly
            to which the detected object is classified.
        detection_treshold: float
            The threshold that dictiates the minimum score of a detection before it is shown as a
            bounding box on the screen.

        """

        self.name = app_name
        self.model = model
        self.device = device
        self.transform = transform
        self.classes = classes
        self.detection_treshold = detection_threshold
        self.runs_in_colab = check_is_running_in_colab()
        self.annotator = PredictionAnnotator(self.classes)

    def __open_camera(self, video_source):
        """Opens the camera

        Parameters
        ----------
        video_source: int
            ID of the video source that is accessed by OpenCV

        Returns
        -------
        None
        """
        self.camera = cv2.VideoCapture(video_source)

    def __transform_frame(self, frame, width, height):
        """Transforms the frame obtained by the camera so that it can be passed
        to the model

        Returns
        -------
        numpy.ndarray
            representing the transformed frame
        """

        frame = cv2.resize(frame, (width, height))
        image = frame.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(frame)
        image = torch.unsqueeze(image, 0)
        return image

    def __run_prediction(self, image):
        """Runs prediction by model

        Parameters
        ----------
        image: numpy.ndarray
            Image as ndarray

        Returns
        -------
        dict
            containing bounding boxes, scores, labels etc.
        """

        self.model.eval()
        with torch.no_grad():
            preds = self.model(image.to(self.device))
        preds = [{k: v.to('cpu') for k, v in t.items()}
                 for t in preds]

        return preds

    def __display_prediction(self, predictions, frame):
        """
        Shows the result of the model detection in a window application

        Parameters
        ----------
        frame: numpy.ndarray
            Image after transformation

        predictions: dict
            containing bounding boxes, scores, labels etc.

        Returns
        -------
        None
        """
        if len(predictions[0]['boxes']) != 0:
            frame = self.annotator.annotate(
                predictions, frame, self.detection_treshold)

        cv2.imshow(self.name, frame)

    def __clean_up_resources(self):
        """Clean up resources obtained before by OpenCV if not executed in google colab"""
        if not self.runs_in_colab:
            self.camera.release()
            cv2.destroyAllWindows()

    def run(self, camera=0, width=800, height=500):
        """Starts the entire process of video capture, model prediction and video annotation
        This process differs when it runs locally or in colab.

        Parameters
        ----------
        camera: int
            ID of the video source. Default is 0.
        width: int
            Width of the application window and subsequently the live feed obtained by the camera.
            Default is 800.
        height: int
            Height of the application window and subsequently the live feed obtained by the camera.
            Default is 500.

        Returns
        -------
        None
        """
        try:
            if self.runs_in_colab:
                label_html = 'Capture'
                frame_bytes = ''
                int_colab_js()
                while True:
                    self.browser_camera = start_video_stream(
                        label_html, frame_bytes)
                    transparent_image = np.zeros([480, 640, 4], dtype=np.uint8)
                    if not self.browser_camera:
                        break
                    frame = js_to_image(self.browser_camera['img'])
                    image = self.__transform_frame(frame, 480, 640)
                    preds = self.__run_prediction(image)

                    if len(preds[0]['boxes']) != 0:
                        frame = self.annotator.annotate(
                            preds, transparent_image, self.detection_treshold)
                        frame[:, :, 3] = (frame.max(axis=2) >
                                          0).astype(int) * 255
                    frame_bytes = bbox_to_bytes(frame)
            else:
                self.__open_camera(camera)

                while self.camera.isOpened():
                    ret, frame = self.camera.read()

                    if ret:
                        image = self.__transform_frame(frame, width, height)
                        preds = self.__run_prediction(image)
                        self.__display_prediction(preds, frame)

                        # If q is pressed, the window is closed.
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    else:
                        break

            self.__clean_up_resources()
        except Exception as e:
            print(e)
            self.__clean_up_resources()
