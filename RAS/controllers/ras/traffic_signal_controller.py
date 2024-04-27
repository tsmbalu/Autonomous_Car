import numpy as np
import cv2

from PIL import Image
from detecto import core


class TrafficSignalController:
    def __init__(self, debug=False):
        # state of traffic signal controller
        self.previous_mode = ''
        self.predictions = None
        self.is_traffic_signal_detected = False

        # Using Detecto package to train the custom model for traffic signal post detection
        # and slow down the car to detect the signal colour correctly.
        # Custom model training code is available in the Traffic_CNN_Model.ipynb file
        self.model = core.Model.load('traffic_signal_model_v7.pth', ['traffic_signal', 'sign_board'])

        """ 
            Counter for the traffic signal post detection. Using detecto model only once for every 50 simulator step.
            Since, the Detecto (i.e, object detection) model takes more time to predict the traffic signal post and 
            it slow down the Webots simulator.   
        """
        self.timer_count = 0

        # flag for debugging
        self.debug = debug

    """
        Display the camera image
    """

    def _display_image(self, image, name, scale=2):
        """
        if debug mode is active, show the image
        """
        if not self.debug:
            return

        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, image.shape[1] * scale, image.shape[0] * scale)
        cv2.imshow(name, image)
        cv2.waitKey(1)

    def _detect_traffic_signal(self, image):
        """
            Detect the traffic signal post using the custom detecto model
        """
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.predictions = self.model.predict(pil_image)
        labels, boxes, scores = self.predictions
        if 'traffic_signal' in labels:
            return True
        return False

    def get_traffic_signal(self, image):
        """
            Return the traffic signal color
        """
        self.timer_count += 1
        traffic_signal_images = []

        cropped_image = image[20:40, 10:128]
        display_image = cropped_image.copy()
        if self.timer_count >= 50:
            # Reset the counter for traffic signal post detection
            self.timer_count = 0
            self.is_traffic_signal_detected = self._detect_traffic_signal(cropped_image)
            labels, boxes, scores = self.predictions

            # Draw rectangle red box when traffic signal is detected
            if self.is_traffic_signal_detected:
                # Draw rectangles around the detected traffic signal on the image
                for box, label in zip(boxes, labels):
                    if label == 'traffic_signal':
                        coords = [int(coord) for coord in box]
                        cv2.rectangle(display_image, (coords[0], coords[1]), (coords[2], coords[3]), (0, 0, 255),
                                      1)  # Draw a green rectangle
                        cv2.putText(display_image, f'{label}', (coords[0], coords[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 0, 255), 1)

                        # Store the traffic light only on the right side
                        if coords[2] > cropped_image.shape[1]-30:
                            traffic_image = cropped_image[coords[1]:coords[3], coords[0]:coords[2]]
                            traffic_signal_images.append(traffic_image)

        # Display Camera Image either with red box or without red box
        self._display_image(display_image, 'Traffic Light Detection')

        if self.is_traffic_signal_detected:
            # Define color ranges for red, yellow and green
            # lower boundary RED color range values; Hue (0 - 10)
            red_lower1 = np.array([0, 100, 20], np.uint8)
            red_upper1 = np.array([10, 255, 255], np.uint8)

            # upper boundary RED color range values; Hue (160 - 180)
            red_lower2 = np.array([160, 100, 20], np.uint8)
            red_upper2 = np.array([179, 255, 255], np.uint8)

            green_lower = np.array([60, 100, 100], np.uint8)
            green_upper = np.array([80, 255, 255], np.uint8)

            yellow_lower = np.array([20, 100, 100], np.uint8)
            yellow_upper = np.array([30, 255, 255], np.uint8)

            for traffic_image in traffic_signal_images:
                self._display_image(traffic_image, 'Traffic Signal Alone')
                hsv_img = cv2.cvtColor(traffic_image, cv2.COLOR_BGR2HSV)

                # Threshold the image to get binary masks for each color
                red_lower_mask = cv2.inRange(hsv_img, red_lower1, red_upper1)
                red_upper_mask = cv2.inRange(hsv_img, red_lower2, red_upper2)
                red_mask = red_lower_mask + red_upper_mask
                green_mask = cv2.inRange(hsv_img, green_lower, green_upper)
                yellow_mask = cv2.inRange(hsv_img, yellow_lower, yellow_upper)

                # Count the number of non-zero pixels in each mask
                red_pixels = np.count_nonzero(red_mask)
                green_pixels = np.count_nonzero(green_mask)
                yellow_pixels = np.count_nonzero(yellow_mask)

                # Determine the color based on the mask with the most non-zero pixels
                if green_pixels > 0:
                    return 'Green'
                elif red_pixels > 0:
                    return 'Red'
                elif yellow_pixels > 0:
                    return 'Yellow'

        return 'NotAvailable'
