import cv2
import mediapipe as mp
import math
import numpy as np
import time

class YogaAnalyzer:
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.body_turned = False
        self.hands_gripped = False
        self.start_time_count = None
        self.calory_burned = 0
        self.elapsed_time = 0

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=False, min_detection_confidence=0.5)

        # This dictionary will hold the results for each analysis
        self.results_dict = {}

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img, draw=True):
        lmlist = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return lmlist

    def find_middle_point(self, x1, y1, x2, y2):
        middle_x = (x1 + x2) / 2
        middle_y = (y1 + y2) / 2
        return int(middle_x), int(middle_y)

    def calculate_distance_based_on_y(self, x1, y1, x2, y2):
        distance = abs(y2 - y1)
        return int(distance)

    def calculate_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def calculate_average_distance(self, x1, y1, x2, y2, x3, y3):
        distance1 = self.calculate_distance(x1, y1, x3, y3)
        distance2 = self.calculate_distance(x2, y2, x3, y3)
        average_distance = (distance1 + distance2) / 2
        return int(average_distance)

    def calculate_angle(self, x1, y1, x2, y2, x3, y3):
        vector1 = [x1 - x2, y1 - y2]
        vector2 = [x3 - x2, y3 - y2]
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
        magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
        cosine_angle = dot_product / (magnitude1 * magnitude2)
        cosine_angle = max(min(cosine_angle, 1.0), -1.0)  # Ensure cosine value is within the valid range
        angle_radians = math.acos(cosine_angle)
        angle_degrees = math.degrees(angle_radians)
        return angle_degrees

    def calculate_collinearity_percentage(self, x1, y1, x2, y2, x3, y3):
        vector1 = [x2 - x1, y2 - y1]
        vector2 = [x3 - x1, y3 - y1]
        cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]
        epsilon = 1e-6
        if abs(cross_product) < epsilon:
            return 100.0
        collinearity_percentage = (1 - abs(cross_product) / (abs(vector1[0] * vector2[1]) + abs(vector1[1] * vector2[0]))) * 100
        return collinearity_percentage

    def calculate_overall_progress(self, result1, result2, result3):
        threshold1 = 83
        threshold2 = 83
        threshold3 = 130
        progress1 = min(result1 / threshold1 * 100, 100)
        progress2 = min(result2 / threshold2 * 100, 100)
        progress3 = min(result3 / threshold3 * 100, 100)
        overall_progress = min((progress1 + progress2 + progress3) / 3, 100)
        return overall_progress

    def calculate_calories_burned(self, progress_score):
        calories_per_unit_progress = 0.001
        base_calories_burned = 0.005
        max_calories_burned = 0.009
        calories_burned = base_calories_burned + progress_score * calories_per_unit_progress
        calories_burned = min(calories_burned, max_calories_burned)
        return calories_burned

    def analyze_pose(self, img):
        img = self.findPose(img)
        lmlist = self.getPosition(img, draw=False)
        if len(lmlist) != 0:
            Head_Mid_point_x, Head_Mid_point_y = self.find_middle_point(lmlist[5][1], lmlist[5][2], lmlist[2][1], lmlist[2][2])
            Hand_Mid_point_x, Hand_Mid_point_y = self.find_middle_point(lmlist[16][1], lmlist[16][2], lmlist[15][1], lmlist[15][2])
            Body_Mid_point_x, Body_Mid_point_y = self.find_middle_point(lmlist[24][1], lmlist[24][2], lmlist[23][1], lmlist[23][2])
            lef_leg_distance = self.calculate_distance_based_on_y(Body_Mid_point_x, Body_Mid_point_y, lmlist[27][1], lmlist[27][2])
            right_leg_distance = self.calculate_distance_based_on_y(Body_Mid_point_x, Body_Mid_point_y, lmlist[28][1], lmlist[28][2])
            base_point_x = lmlist[27][1] if lef_leg_distance > right_leg_distance else lmlist[28][1]
            base_point_y = lmlist[27][2] if lef_leg_distance > right_leg_distance else lmlist[28][2]
            back_point_x = lmlist[28][1] if lef_leg_distance > right_leg_distance else lmlist[27][1]
            back_point_y = lmlist[28][2] if lef_leg_distance > right_leg_distance else lmlist[27][2]
            hands_close_distance = self.calculate_average_distance(lmlist[16][1], lmlist[16][2], lmlist[15][1], lmlist[15][2], Hand_Mid_point_x, Hand_Mid_point_y)
            self.hands_gripped = hands_close_distance < 30
            body_mid_points_distance = self.calculate_average_distance(lmlist[24][1], lmlist[24][2], lmlist[23][1], lmlist[23][2], Body_Mid_point_x, Body_Mid_point_y)
            self.body_turned = body_mid_points_distance < 30

            if self.body_turned:
                front_angle = self.calculate_angle(Head_Mid_point_x, Head_Mid_point_y, Body_Mid_point_x, Body_Mid_point_y, base_point_x, base_point_y)
                back_angle = self.calculate_angle(base_point_x, base_point_y, Body_Mid_point_x, Body_Mid_point_y, back_point_x, back_point_y)
                body_line_percentage = self.calculate_collinearity_percentage(Hand_Mid_point_x, Hand_Mid_point_y, Body_Mid_point_x, Body_Mid_point_y, back_point_x, back_point_y)
                hand_body_angle = self.calculate_angle(Hand_Mid_point_x, Hand_Mid_point_y, Head_Mid_point_x, Head_Mid_point_y, Body_Mid_point_x, Body_Mid_point_y)
                yoga_progress = self.calculate_overall_progress(front_angle, back_angle, hand_body_angle)
                if self.start_time_count is None and 50 < yoga_progress < 60:
                    self.start_time_count = time.time()
                elapsed_seconds = int(time.time() - self.start_time_count) if self.start_time_count else 0
                self.calory_burned += self.calculate_calories_burned(yoga_progress)

                hands_gripped_status = "Hands close to each other" if self.hands_gripped else "Hands not close to each other"
                body_turned_status = "Body turned" if self.body_turned else "Body not turned"

                self.results_dict = {
                    "front_angle": front_angle,
                    "back_angle": back_angle,
                    "body_line_percentage": body_line_percentage,
                    "hand_body_angle": hand_body_angle,
                    "yoga_progress": yoga_progress,
                    "calories_burned": self.calory_burned,
                    "elapsed_time": elapsed_seconds,
                    "hands_gripped_status": hands_gripped_status,
                    "body_turned_status": body_turned_status
                }

        return img

    def get_results(self):
        return self.results_dict

    def overlay_information(self, img, elapsed_time, calory_burned):
        elapsed_time_text = f"Elapsed Time: {elapsed_time:.2f}s"
        calory_burned_text = f"Calories Burned: {calory_burned:.3f}cal"
        cv2.putText(img, elapsed_time_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, calory_burned_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break
            img = self.analyze_pose(img)
            cv2.imshow("Yoga Analyzer", img)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
                break
        cap.release()
        cv2.destroyAllWindows()