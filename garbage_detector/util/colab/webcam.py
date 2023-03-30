import io
import os
from base64 import b64decode, b64encode

import cv2
import numpy as np
import PIL
from IPython.display import Javascript, display

from garbage_detector.util.colab import check_is_running_in_colab


def js_to_image(js_reply):
    """
    Parameters
    ----------
    js_reply:
        JavaScript object containing image from webcam

    Returns
    -------
    numpy.ndarray
        representing OpenCV BGR image
    """
    # decode base64 image
    image_bytes = b64decode(js_reply.split(',')[1])
    # convert bytes to numpy array
    jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
    # decode numpy array into OpenCV BGR image
    img = cv2.imdecode(jpg_as_np, flags=1)

    return img


def bbox_to_bytes(bbox_array):
    """A function to convert bounding box on transparent image into base64 byte string to be overlayed on video stream

    Parameters
    ----------
    bbox_array: numpy.ndarray
        Represents transparent image with bounding box rectangle that is overlayed on the browser video stream.

    Returns
    -------
    bytes
        representing Base64 image byte string
    """
    # convert array into PIL image
    bbox_PIL = PIL.Image.fromarray(bbox_array, 'RGBA')
    iobuf = io.BytesIO()
    # format bbox into png for return
    bbox_PIL.save(iobuf, format='png')
    # format return string
    bbox_bytes = 'data:image/png;base64,{}'.format(
        (str(b64encode(iobuf.getvalue()), 'utf-8')))

    return bbox_bytes


def int_colab_js():
    """Function for loading javascript so that it is available for use in jupyter (IPython)

    Returns
    -------
    None
    """

    dir = os.path.dirname(__file__)
    path = os.path.join(dir, "js", "video.js")
    js = Javascript(filename=path)

    display(js)


def start_video_stream(label, bbox):
    """Starts video stream in browser by calling a javascript function
    The bounding box (bbox) that is passed to this image is inside a transparent image.
    The bounding box is placed over the video image of the browser, so that next to the 
    browser video frame, only the bounding box with label and score is visible.

    The call to eval_js returns the next frame for which new bounding boxes need to be
    calculated.

    Parameters
    ----------
    label: str
        Corresponding labe to bounding box

    bbox: byte
        Represents transparent image with bounding box rectangle

    Returns
    -------
    byte
        representing next video frame of the browser video feed.

    """
    try:
        from google.colab.output import eval_js
        data = eval_js('stream_frame("{}", "{}")'.format(label, bbox))
        return data
    except Exception as e:
        print(e)
