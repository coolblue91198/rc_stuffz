import cv2.cv2 as cv2
import numpy as np
import statistics as stats
from typing import Tuple


class ObjectTracking:

    def __init__(self, upper_color: Tuple[int, int, int] = (10, 10, 10), lower_color: Tuple[int, int, int] = (0, 0, 0)):
        self._cap = cv2.VideoCapture(0)
        self._ret = True
        self._xlist = [0]
        self._ylist = [0]
        self._wlist = [0]
        self._hlist = [0]
        self._upper = self.rgb2hsv(upper_color[0], upper_color[1], upper_color[2])[0, 0]
        self._lower = self.rgb2hsv(lower_color[0], lower_color[1], lower_color[2])[0, 0]
        self.filter_mask = None
        self.frame_size = (self._cap.get(cv2.CAP_PROP_FRAME_WIDTH), self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not self._start_stream():
            self._release_resources()

    @staticmethod
    def rgb2hsv(red, green, blue):
        """Converts from RGB to OPENCV HSV"""

        bgr = np.uint8([[[blue, green, red]]])

        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    def _start_stream(self):
        """Actual stream and it's logic"""

        while self._ret:
            self._ret, frame = self._cap.read()

            frame = cv2.flip(frame, 1)

            # Convert image to HSV
            frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Apply Color Filter
            lr, ur = self._apply_color_filter()

            # Get Mask
            mask = self._create_mask(frame_HSV, lr, ur)

            # Draw bounding boxes
            self._draw_boxes(mask, frame)

            # Draw Center of Area/Mass
            com_center = self._draw_com(frame)

            # Compute Distance from COM to Center of Frame
            self._get_distance(com_center)

            # Show Original Frame
            cv2.imshow("Original Video", frame)

            # Show Masked Frame
            cv2.imshow("Masked Video", mask)

            # Close stream if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return 0

    def _release_resources(self):
        self._cap.release()
        cv2.destroyAllWindows()

    def _apply_color_filter(self):
        lower_range = np.array(self._lower)
        upper_range = np.array(self._upper)

        return lower_range, upper_range

    @staticmethod
    def _create_mask(image, lower_range, upper_range):
        """Create mask and fix noise"""

        filter_mask = cv2.inRange(image, lower_range, upper_range)

        # Fix Noise
        mask_open = cv2.morphologyEx(filter_mask, cv2.MORPH_OPEN, np.ones((5, 5)))
        mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, np.ones((20, 20)))

        return mask_close

    def _draw_boxes(self, mask, frame, draw_contours: bool = False):
        """Draws bounding boxes to filtered objects"""

        # Find Contours
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw Contours
        if draw_contours:
            cv2.drawContours(frame, contours, -1, (255, 0, 0), 3)

        self._clear_lists(5)

        # Draw bounding boxes
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            self._add_to_lists(x, y, w, h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    def _draw_com(self, frame):
        """Draws center of area of frame"""

        # Compute average positions
        self._x_avg = int(stats.mean(self._xlist))
        self._y_avg = int(stats.mean(self._ylist))
        self._w_avg = int(stats.mean(self._wlist))
        self._h_avg = int(stats.mean(self._hlist))

        # Compute and draw center
        center = (int((2*self._x_avg + self._w_avg)/2), int((2*self._y_avg + self._h_avg)/2))
        cv2.circle(frame, center, 10, (255, 0, 0), 2)

        return center

    def _clear_lists(self, fps):
        """Clears lists every fps frames"""
        if len(self._xlist) > fps:
            self._xlist = [self._xlist[-1]]
        if len(self._ylist) > fps:
            self._ylist = [self._ylist[-1]]
        if len(self._wlist) > fps:
            self._wlist = [self._wlist[-1]]
        if len(self._hlist) > fps:
            self._hlist = [self._hlist[-1]]

    def _add_to_lists(self, x, y, w, h):
        """Adds values to list"""
        self._xlist.append(x)
        self._ylist.append(y)
        self._wlist.append(w)
        self._hlist.append(h)

    def _get_distance(self, com_center):
        """Gets distance from COM (blue circle) to center of camera"""
        width = self.frame_size[0]
        height = self.frame_size[1]

        # Find center of frame
        center = (width/2, height/2)

        # Get distance from center in x-y direction (in plane)
        x_dist = center[0] - com_center[0]
        y_dist = -center[1] + com_center[1]     # Switch signs since y-axis is flipped from normal cartesian coords

        #***NEED TO IMPLEMENT FOCAL LENGTH MEASUREMENTS TO GET REAL Z-DISTANCE***#
        # Get distance from center in z direction (out of plane of camera)
        z_dist = width - self._w_avg

        dist_cm = (2.54*x_dist/127, 2.54*y_dist/127, 2.54*z_dist/127)

        print('Distance: ({:.{}f}, {:.{}f}, {:.{}f}) cm'.format(dist_cm[0], 2, dist_cm[1], 2, dist_cm[2], 2))


if __name__ == '__main__':
    ot = ObjectTracking((255, 60, 120), (200, 130, 170))
