import cv2
import dlib
import numpy as np
from mtcnn.mtcnn import MTCNN


# our main class designed for face detection and processing
class FaceDetector:
    # local variables that don't change while program is running
    # the most important are refresh_rate (how often searching on full screen towards faces will be performed)
    # and radius (radius of searching for a local search)
    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([255, 173, 127], np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    refresh_rate = 60
    radius = 40
    small_radius = 40
    detector = MTCNN()
    tracker = dlib.correlation_tracker()

    def __init__(self):
        # local variables useful for continuous detection for one instance of the detector
        # most important are counter (at which frame we are at)
        # and Locations[] (list of approximated locations where faces are on this frame)
        self.counter = 0
        self.Locations = []
        self.trackers = []

    # method for getting mask of faces, using skin extraction based on our Locations[] and skin color values in YCrCb
    # color space
    def get_mask(self, frame):
        new_frame = np.zeros((1080, 1920, 3), np.uint8)
        for i in range(len(self.Locations)):
            if 2 * self.Locations[i][0] > 10 and 2 * self.Locations[i][2] < 1070 and 2 * self.Locations[i][1] < 1910 \
                    and 2 * self.Locations[i][3] > 10:
                face_cropped = frame[(2 * self.Locations[i][0] - 10):(2 * self.Locations[i][2] + 10),
                               (2 * self.Locations[i][3] - 10):(2 * self.Locations[i][1] + 10)]
                imageYCrCb = cv2.cvtColor(face_cropped, cv2.COLOR_BGR2YCR_CB)
                imageYCrCb = cv2.erode(imageYCrCb, self.kernel, iterations=6)
                imageYCrCb = cv2.dilate(imageYCrCb, self.kernel, iterations=6)
                skinRegionYCrCb = cv2.inRange(imageYCrCb, self.min_YCrCb, self.max_YCrCb)
                mask = np.zeros_like(face_cropped)
                ellipse_points = self.extract_ellipse_points(i)
                ellipse = cv2.fitEllipse(ellipse_points)
                mask = cv2.ellipse(mask, ellipse, color=(255, 255, 255), thickness=-1)
                face_cropped = cv2.bitwise_and(face_cropped, mask, mask=skinRegionYCrCb)
                new_frame[(2 * self.Locations[i][0] - 10):(2 * self.Locations[i][2] + 10),
                (2 * self.Locations[i][3] - 10):(2 * self.Locations[i][1] + 10)] = face_cropped
        return new_frame

    # method for performing whole frame search for faces using MTCNN face detector
    # Locations[] are filled with the coordinates of the faces
    def full_search(self, frame):
        detected_faces = self.detector.detect_faces(frame)
        self.Locations.clear()
        for result in detected_faces:
            x, y, width, height = result['box']
            left = x
            right = x + width
            top = y
            bottom = y + height
            self.Locations.append([top, right, bottom, left])


    def new_search(self, frame, head_list):
        self.Locations.clear()
        for element in head_list:
            left_margin = int(element[0]//2 - self.small_radius)
            right_margin = int(element[0]//2 + self.small_radius)
            top_margin = int(element[1]//2 - self.small_radius)
            bot_margin = int(element[1]//2 + self.small_radius)
            if left_margin < 0:
                left_margin = 0
            if right_margin >= 960:
                right_margin = 956
            if top_margin < 0:
                top_margin = 0
            if bot_margin >= 540:
                bot_margin = 539
            cropped = frame[top_margin:bot_margin, left_margin:right_margin]
            detected_faces = self.detector.detect_faces(cropped)
            for result in detected_faces:
                x, y, width, height = result['box']
                left = x
                right = x + width
                top = y
                bottom = y + height
                self.Locations.append([top_margin + top , left_margin + right, top_margin + bottom
                                            , left_margin + left])
        return frame


    # method for starting multiple dlib trackers for objects located in Locations[]
    def start_trackers(self, frame):
        new_trackers = []
        for result in self.Locations:
            maxArea = 0
            x = 0
            y = 0
            w = 0
            h = 0
            if (result[1] - result[3]) * (result[2] - result[0]) > maxArea:
                x = int(result[3])
                y = int(result[0])
                w = int(result[1] - result[3])
                h = int(result[2] - result[0])
                maxArea = w * h
            if maxArea > 0:
                t = dlib.correlation_tracker()
                t.start_track(frame, dlib.rectangle(x, y, x + w, y + h))
                new_trackers.append(t)
        return new_trackers

    # method of searching for a one face on a small area (performed when tracker has lost tracking object) based on the
    # last useful tracker location
    # Locations[] gets updated for specific index
    def small_search(self, frame, index):
        if (self.Locations[index][0] - self.radius > 0) and (self.Locations[index][3] - self.radius > 0) and (
                self.Locations[index][2] + self.radius < 540) and (self.Locations[index][1] + self.radius < 960):
            cropped = frame[(self.Locations[index][0] - self.radius):(self.Locations[index][2] + self.radius),
                      (self.Locations[index][3] - self.radius):(self.Locations[index][1] + self.radius)]
            detected_faces = self.detector.detect_faces(cropped)
            for result in detected_faces:
                x, y, width, height = result['box']
                left = x
                right = x + width
                top = y
                bottom = y + height
            for j in range(len(detected_faces)):
                self.Locations[index][1] = (self.Locations[index][3] + right - self.radius)
                self.Locations[index][2] = (self.Locations[index][0] + bottom - self.radius)
                self.Locations[index][0] = (self.Locations[index][0] + top - self.radius)
                self.Locations[index][3] = (self.Locations[index][3] + left - self.radius)
            return len(detected_faces)

    # writing current tracking location into Location[] assuming face is the written area
    def save_location(self, x, y, w, h, index):
        self.Locations[index][0] = y
        self.Locations[index][1] = x + w
        self.Locations[index][2] = y + h
        self.Locations[index][3] = x

    # method for unpacking position returned by tracker.getPosition() method (dlib.Rectangle)
    def unpack_position(self, box):
        x = int(box.left())
        y = int(box.top())
        w = int(box.width())
        h = int(box.height())
        return x, y, w, h

    # method for data preparation by translating our Location[] system into dlib.Rectangle used by trackers
    def extract_box(self, index):
        x = int(self.Locations[index][3])
        y = int(self.Locations[index][0])
        w = int(self.Locations[index][1] - self.Locations[index][3])
        h = int(self.Locations[index][2] - self.Locations[index][0])
        return x, y, w, h

    def extract_ellipse_points(self, index):
        _, _, w, h = self.extract_box(index)
        w *= 2
        h *= 2
        points = [[10, 10],
                  [10, h + 10],
                  [w + 10, (h + 10) / 2],
                  [10, (h + 10) / 2],
                  [(w + 10) / 2, h],
                  [(w + 10) / 2, 10],
                  [w + 10, h + 10],
                  [w + 10, 10]]
        return np.array(points, dtype=np.int32)

    # main method where all magic happens
    def face_processing(self, frame, heads):
        # Downsampled image used only for search algorithm
        small_frame = cv2.resize(frame, (960, 540), 0, 0)

        # every (refresh_rate) frames search based on head locations is performed
        # using MTCNN detector
        # and also trackers are being refreshed
        if self.counter % self.refresh_rate == 0:
            self.new_search(small_frame, heads)
        elif self.counter % self.refresh_rate == 1:
            self.trackers = self.start_trackers(small_frame)

        # in standard case scenario basic tracking is performed
        # trackers are being updated
        else:
            if self.counter % 3 != 2:
                if len(self.trackers)!=1:
                    mark = len(self.trackers)
                    half = int(mark/2)
                    j = 0
                    if(self.counter % 3 == 0):
                        for i in range(0, half):
                            trackingQuality = self.trackers[i-j].update(small_frame)
                            tracked_position = self.trackers[i-j].get_position()
                            t_x, t_y, t_w, t_h = self.unpack_position(tracked_position)
                            if trackingQuality >= 4.0:
                                self.save_location(t_x, t_y, t_w, t_h, i)

                            # in case of losing tracked object from a tracker window searching using MTCNN detector on a small
                            # area is performed using area of last known position of a tracker window
                            else:
                                self.trackers.pop(i)
                                check = self.small_search(small_frame, i)
                                j += 1
                                if check != 0:
                                    x, y, w, h = self.extract_box(i)
                                    t = dlib.correlation_tracker()
                                    t.start_track(small_frame, dlib.rectangle(x, y, x + w, y + h))
                                    self.trackers.insert(i, t)
                    else:
                        for i in range(half, mark):
                            trackingQuality = self.trackers[i-j].update(small_frame)
                            tracked_position = self.trackers[i-j].get_position()
                            t_x, t_y, t_w, t_h = self.unpack_position(tracked_position)
                            if trackingQuality >= 4.0:
                                self.save_location(t_x, t_y, t_w, t_h, i)

                            # in case of losing tracked object from a tracker window searching using MTCNN detector on a small
                            # area is performed using area of last known position of a tracker window
                            else:
                                self.trackers.pop(i)
                                check = self.small_search(small_frame, i)
                                j+=1
                                if check != 0:
                                    x, y, w, h = self.extract_box(i)
                                    t = dlib.correlation_tracker()
                                    t.start_track(small_frame, dlib.rectangle(x, y, x + w, y + h))
                                    self.trackers.insert(i, t)
                else:
                    for i in range(len(self.trackers)):
                        trackingQuality = self.trackers[i].update(small_frame)
                        tracked_position = self.trackers[i].get_position()
                        t_x, t_y, t_w, t_h = self.unpack_position(tracked_position)
                        if trackingQuality >= 4.0:
                            self.save_location(t_x, t_y, t_w, t_h, i)

                        # in case of losing tracked object from a tracker window searching using MTCNN detector on a small
                        # area is performed using area of last known position of a tracker window
                        else:
                            self.trackers.pop(i)
                            check = self.small_search(small_frame, i)
                            if check != 0:
                                x, y, w, h = self.extract_box(i)
                                t = dlib.correlation_tracker()
                                t.start_track(small_frame, dlib.rectangle(x, y, x + w, y + h))
                                self.trackers.insert(i, t)
        # next thing done is extracting the mask of the faces and hovering it into frame with converted background
        # mask is extracted from original frame
        new_frame = self.get_mask(frame)
        self.counter += 1
        return new_frame


if __name__ == '__main__':

    RGB = cv2.VideoCapture("body-all-simple-RGB.avi")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('outputFace-optimized.avi', fourcc, 20.0, (1920, 1080))
    time_passed = 0
    i = 0
    accuracy = 0
    face_detector = FaceDetector()
    if not RGB.isOpened():
        print("Error opening video")

    while RGB.isOpened():
        ret, frame = RGB.read()
        if not ret:
            break
        start = cv2.getTickCount()
        result = face_detector.face_processing(frame)
        time_passed += (cv2.getTickCount() - start)
        i += 1
        accuracy += np.count_nonzero(result)
        # cv2.namedWindow('RGB', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('RGB', 1366, 768)
        # cv2.imshow("RGB", result)
        out.write(result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    time_per_frame = time_passed / (i * cv2.getTickFrequency())
    print("%.3f fps" % (i * cv2.getTickFrequency() / time_passed))
    print('Time per frame: %.3fs' % time_per_frame)
    print("Accuracy: ", accuracy)

    RGB.release()
    out.release()
    cv2.destroyAllWindows()
