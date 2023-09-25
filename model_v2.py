from ultralytics import YOLO
import cv2

import pandas as pd
import numpy as np


class ModelDetected:
    def __init__(self, model, type_cam: str = 'front', skip_frames: int = 2, conf_level: float = 0.3, verbose: bool = False,
                 device: str = 'cpu', dict_classes: dict = {0: 0}):
        # Тип камеры front, left, right
        self.type_cam = type_cam
        # Кол-во кадров которое пропускаем
        self.skip_frames = skip_frames
        ### Configurations
        # Verbose during prediction
        self.verbose = False
        # Scaling percentage of original frame
        self.scale_percent = 50
        self.model = model
        # model confidence level
        # conf_level = 0.5
        self.conf_level = conf_level
        self.iou = 0.5
        # Threshold of centers ( old\new)
        # thr_centers = 20
        self.thr_centers = 20
        # Number of max frames to consider a object lost
        self.frame_max = 5
        # Number of max tracked centers stored
        self.patience = 100
        # ROI area color transparency
        self.alpha = 0.2
        # Objects to detect Yolo
        # self.class_IDS = [0]
        self.dict_classes = dict_classes
        self.class_IDS = list(self.dict_classes.keys())
        self.verbose = verbose
        self.device = device
        self.input_file = None

    def load_video(self, filename: str):
        self.dataset_path = "/".join(filename.split("/")[:-1]).strip()
        if self.dataset_path is None:
            self.dataset_path = ''
        if len(self.dataset_path) != 0:
            self.dataset_path += '/'
        self.filename = filename.split("/")[-1]
        self.input_file = self.dataset_path + self.filename
        self.output_file = self.dataset_path + "rep_" + self.filename
        if self.verbose:
            print(f"self.input_file: {self.input_file}")
            print(f"self.output_file: {self.output_file}")
        # -------------------------------------------------------
        # Reading video with cv2
        self.video = cv2.VideoCapture(self.input_file)
        if self.verbose:
            print(f'[INFO] - Verbose during Prediction: {self.verbose}')
        # Original informations of video
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))

        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.duration = self.frame_count / self.fps
        if self.verbose:
            print('[INFO] - Original Dim: ', (self.width, self.height))
        # Scaling Video for better performance
        if self.scale_percent != 100:
            if self.verbose:
                print('[INFO] - Scaling change may cause errors in pixels lines ')
            self.width = int(self.width * self.scale_percent / 100)
            self.height = int(self.height * self.scale_percent / 100)
            if self.verbose:
                print('[INFO] - Dim Scaled: ', (self.width, self.height))

        # -------------------------------------------------------
        ### Video output ####
        self.VIDEO_CODEC = "mp4v"
        if self.verbose:
            print(self.output_file)

        self.output_video = cv2.VideoWriter(self.output_file,
                                            cv2.VideoWriter_fourcc(*self.VIDEO_CODEC),
                                            self.fps, (self.width, self.height))

        if self.type_cam == 'right':
            region_of_interest_vertices = [
                    (0, self.height),
                    (0 , self.height*3/5),
                    (0 + self.width/9, self.height/10),
                    (0 + self.width/5, self.height/10),
                    (self.width - self.width/5, self.height)
                ]
        elif self.type_cam == 'left':
            region_of_interest_vertices = [
                    (self.width, self.height),
                    (self.width , self.height*3/5),
                    (self.width - self.width/4, self.height/10),
                    (self.width - self.width/3, self.height/10),
                    (0 + self.width/5, self.height)
                ]
        else:
            region_of_interest_vertices = [
                (0 + self.width/5, self.height),
                (self.width/2 - self.width/20, self.height/2.5),
                (self.width/2 + self.width/20, self.height/2.5),
                (self.width - self.width/5, self.height)
            ]

        self.area_roi = [np.array(region_of_interest_vertices, np.int32)]

        self.width_slice = slice(max(min(self.area_roi[0][:, 0]) - 1, 0),
                                 min(max(self.area_roi[0][:, 0]) + 1, self.width))
        self.height_slice = slice(max(min(self.area_roi[0][:, 1]) - 1, 0),
                                  min(max(self.area_roi[0][:, 1]) + 1, self.height))

    # Auxiliary functions
    def risize_frame(self, frame, scale_percent):
        """Function to resize an image in a percent scale"""
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        return resized

    def filter_tracks(self, centers, patience):
        """Function to filter track history"""
        filter_dict = {}
        for k, i in centers.items():
            d_frames = i.items()
            filter_dict[k] = dict(list(d_frames)[-patience:])

        return filter_dict

    def update_tracking(self, centers_old, obj_center, thr_centers, lastKey, frame, frame_max):
        """Function to update track of objects"""
        is_new = 0
        lastpos = [(k, list(center.keys())[-1], list(center.values())[-1]) for k, center in centers_old.items()]
        lastpos = [(i[0], i[2]) for i in lastpos if abs(i[1] - frame) <= frame_max]
        # Calculating distance from existing centers points
        previous_pos = [(k, obj_center) for k, centers in lastpos if
                        (np.linalg.norm(np.array(centers) - np.array(obj_center)) < thr_centers)]
        # if distance less than a threshold, it will update its positions
        if previous_pos:
            id_obj = previous_pos[0][0]
            centers_old[id_obj][frame] = obj_center

        # Else a new ID will be set to the given object
        else:
            if lastKey:
                last = lastKey.split('D')[1]
                id_obj = 'ID' + str(int(last) + 1)
            else:
                id_obj = 'ID0'

            is_new = 1
            centers_old[id_obj] = {frame: obj_center}
            lastKey = list(centers_old.keys())[-1]

        return centers_old, id_obj, is_new, lastKey

    def canny_edges(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ex, threshold = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # print(threshold)
        kernel = np.ones((2, 3), np.uint8)
        opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=2)
        edges = cv2.Canny(opening, 3, 3)
        return edges
        ex, threshold = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((2, 3), np.uint8)
        opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=2)
        edges = cv2.Canny(opening, 3, 3)
        return edges

    def region_of_interest(self, img, vertices):
        mask = np.zeros_like(img)
        channel_count = img.shape[2]
        match_mask_color = (255,) * channel_count
        cv2.fillPoly(mask, vertices, match_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def detected(self):
        # Auxiliary variables
        centers_old = {}
        frames_list = []
        count_p = 0
        lastKey = ''

        if self.input_file is None:
            raise "Need load video load_video(filename)"

        timestamps = [self.video.get(cv2.CAP_PROP_POS_MSEC)]
        calc_timestamps = [0.0]

        detected_report = {}

        current_skipframe = 0
        for i in range(int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))):
            # reading frame from video
            # _, frame = self.video.read()
            frame_exists, frame = self.video.read()

            if frame_exists:
                timestamps.append(self.video.get(cv2.CAP_PROP_POS_MSEC))
                calc_timestamps.append(calc_timestamps[-1] + 1000 / self.fps)
            else:
                break

            current_skipframe += 1
            if current_skipframe < self.skip_frames:
                continue
            current_skipframe = 0

            # Applying resizing of read frame
            frame = self.risize_frame(frame, self.scale_percent)
            ROI = frame[self.width_slice, self.height_slice]

            # Getting predictions
            y_hat = self.model.predict(ROI, conf=self.conf_level, iou=self.iou, classes=self.class_IDS, device=self.device,
                                  verbose=False)

            # Getting the bounding boxes, confidence and classes of the recognize objects in the current frame.
            boxes = y_hat[0].boxes.xyxy.cpu().numpy()
            conf = y_hat[0].boxes.conf.cpu().numpy()
            classes = y_hat[0].boxes.cls.cpu().numpy()

            # Storing the above information in a dataframe
            positions_frame = pd.DataFrame(y_hat[0].cpu().numpy().boxes.data,
                                           columns=['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class'])

            if len(positions_frame) > 0:
                positions_frame = positions_frame.sort_values(by='conf', ascending=False)
                detected_report[int(timestamps[-1])] = [int(timestamps[-1]) / 1000, positions_frame.iloc[0]['conf'],
                                                        positions_frame.iloc[0]['class']]

            # Translating the numeric class labels to text
            labels = [self.dict_classes[i] for i in classes]

            # For each people, draw the bounding-box and counting each one the pass thought the ROI area
            for ix, row in enumerate(positions_frame.iterrows()):
                # Getting the coordinates of each vehicle (row)
                xmin, ymin, xmax, ymax, confidence, category, = row[1].astype('int')

                # Calculating the center of the bounding-box
                center_x, center_y = int(((xmax + xmin)) / 2), int((ymax + ymin) / 2)

                # Updating the tracking for each object
                centers_old, id_obj, is_new, lastKey = self.update_tracking(centers_old, (center_x, center_y),
                                                                            self.thr_centers, lastKey, i,
                                                                            self.frame_max)

                # Updating people in roi
                count_p += is_new

                # drawing center and bounding-box in the given frame
                cv2.rectangle(ROI, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # box
                for center_x, center_y in centers_old[id_obj].values():
                    cv2.circle(ROI, (center_x, center_y), 5, (0, 0, 255), -1)  # center of box

                # Drawing above the bounding-box the name of class recognized.
                cv2.putText(img=ROI, text=id_obj + ':' + str(np.round(conf[ix], 2)),
                            org=(xmin, ymin - 10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(0, 0, 255),
                            thickness=1)

            # Filtering tracks history
            centers_old = self.filter_tracks(centers_old, self.patience)

            # Drawing the ROI area
            overlay = frame.copy()

            # Рисуем трапецию
            cv2.polylines(overlay, pts=self.area_roi, isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.fillPoly(overlay, self.area_roi, (0, 255, 0))
            frame = cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0)

            # Saving frames in a list
            frames_list.append(frame)
            # saving transformed frames in a output video formaat
            self.output_video.write(frame)

        # Releasing the video
        self.output_video.release()
        self.video.release()
        self.history = detected_report
        self.create_history_df(detected_report)
        return detected_report

    def create_history_df(self, history):
        self.history_df = pd.DataFrame.from_dict(history).T.rename(columns={0:"time", 1:'accuracy', 2:'class'})
        if len(self.history_df) == 0:
            self.history_df = pd.DataFrame(columns=['time', 'accuracy', 'class'])
        print(len(self.history_df))
        display(self.history_df)
        self.history_df['time'] = self.history_df['time'].astype('int32')
        self.history_df['time'] = self.history_df['time'].apply(lambda x: '\"{0:02d}:{1:02d}\"'.format(int(x // 60), int(x - int(x // 60) * 60)))
        self.history_df['class'] = self.history_df['class'].apply(lambda x: self.dict_classes[x])
        self.history_df['accuracy'] = self.history_df['accuracy'].apply(lambda x: round(x,2))
        return self.history_df

    def calc_events(self, time_by_events: int = 20):
        """
        param: time_by_events - минимальное кол-во секунд между событиями, чтобы они считалсиь различными
        """
        events = sorted(list(self.history.keys()))
        if len(events) == 0:
            return []
        filter_events = [int((events[0]-1)/1000)]
        if len(events) > 0:
            current_event = events[0]
            for i in range(1, len(events)):
                if (events[i] - current_event) >= (time_by_events * 1000):
                    filter_events.append(int((events[i]-1)/1000))
                current_event = events[i]
        return filter_events

    def report(self, events):
        # report = "filename,cases_count,timestamps\n"
        report = f'{self.filename},'
        event_report = []
        events = list(map(int, sorted(events)))
        for evnt in events:
            minutes = int(evnt // 60)
            seconds = int(evnt - minutes * 60)
            event_report.append('\"{0:02d}:{1:02d}\"'.format(minutes, seconds))
        report += str(len(event_report)) + ','
        report += f'[{", ".join(event_report)}]'
        return report



if __name__ == "__main__":
    # device = 'cpu'
    device = 'cuda'
    model = YOLO('yolov8n.pt')

    DATASET_PATH = 'datasets/'
    input_file = DATASET_PATH + "01_48_31.mp4"

    dict_classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 16: 'dog', 17: 'horse', 18: 'sheep',
                    19: 'cow', 21: 'bear', }

    detector = ModelDetected(model=model, type_cam='front', skip_frames=10, conf_level=0.7, verbose=False, dict_classes=dict_classes, device=device)

    detector.load_video(input_file)
    history = detector.detected()
    events = detector.calc_events(time_by_events=7)
    report = detector.report(events)

    # ссылка на сгенерированное видео
    print(detector.output_file)
    # подробный отчет в формате pd.DataFrame
    print(detector.history_df)
    # отчет для submita
    print(report)