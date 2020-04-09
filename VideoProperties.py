import cv2

class VideoProperties():
    def __init__(self, ini):
        self.camera_rows = int(ini['video']['camera_rows'])
        self.camera_cols = int(ini['video']['camera_cols'])

        self.resized_width = int(ini['video']['resized_width'])
        self.resized_height = 0

        self.modified_fps = int(ini['video']['fps'])
        self.start_frame = int(ini['video']['start_frame'])

        self.scene_width = 0
        self.scene_height = 0

        self.video_path = ini['video']['path']
        self.start_frame = int(ini['video']['start_frame'])

        self.camera_rows = int(ini['video']['camera_rows'])
        self.camera_cols = int(ini['video']['camera_cols'])

        self.counter = 0

    def set_video_info(self, width, height, resized_width):

        self.resized_width = resized_width
        self.resized_height = int(height * (self.resized_width / width))

        self.scene_width = int(self.resized_width / self.camera_cols)
        self.scene_height = int(self.resized_height / self.camera_rows)

        # self.rois = self.initialize_rois(self.init_rois, self.scene_width, self.scene_height)
        # self.matrices = self.get_matrices(self.rois, self.roi_width, self.roi_height)

    def play_video(self, video_capture):
        ret = False
        frame = None
        original_fps = video_capture.get(cv2.CAP_PROP_FPS)
        step = int(round(original_fps/self.modified_fps))
        for i in range(step):
            ret, frame = video_capture.read()
            self.counter += 1
        if ret:
            frame = cv2.resize(frame, dsize=(self.resized_width, self.resized_height))
            return frame
        return ret
