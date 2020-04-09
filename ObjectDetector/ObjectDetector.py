from __future__ import division

import configparser
import pickle as pkl
import random

from ObjectDetector.util import *
from ObjectDetector.darknet import Darknet

class ObjectBBox:
    def __init__(self, label, x, y, w, h, frame_index):
        self.label = label
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.frame_index = frame_index

class ObjectDetector():

    def __init__(self, ini):

        self.model = set_model(ini['object_detector']['cfg_path'],
                               ini['object_detector']['weights_path'],
                               int(ini['object_detector']['resolution']))
        self.inp_dim = self.model.net_info["height"]

        self.confidence = float(ini['object_detector']['confidence'])
        self.nms_thresh = float(ini['object_detector']['nms_thresh'])

        self.classes = load_classes(ini['object_detector']['names_path'])

        self.colors = pkl.load(open(ini['object_detector']['pallete_path'], "rb"))

        self.whitelist = [item.strip() for item in ini['object_detector']['whitelist'].split(',')]
        self.blacklist = [item.strip() for item in ini['object_detector']['blacklist'].split(',')]

        self.objects = {}

    def detect(self, org_img, frame_index, show_img=False):
        img, dim = prep_image(org_img, self.inp_dim)

        im_dim = torch.FloatTensor(dim).repeat(1, 2)

        if torch.cuda.is_available():
            im_dim = im_dim.cuda()
            img = img.cuda()

        with torch.no_grad():
            output = self.model(Variable(img), torch.cuda.is_available())

        output = write_results(output, self.confidence, len(self.classes), nms=True, nms_conf=self.nms_thresh)

        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(self.inp_dim / im_dim, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (self.inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (self.inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

        # self.delete_old_objects(frames)
        result = list(map(lambda x:self.get_obj_bbox(x, org_img, frame_index, show_img), output))
        result = list(filter(lambda x:x is not None, result))

        if show_img:
            cv2.imshow("frame", org_img)
            cv2.waitKey(1)
        return result

    def get_obj_bbox(self, x, img, frame_index, show_img):
        cls = int(x[-1])
        label = "{0}".format(self.classes[cls])

        if label not in self.whitelist:
            return

        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())

        if show_img:
            self.draw_bounding_boxes(img, c1, c2, label)
        x, y, w, h = int(c1[0]), int(c1[1]), int(c2[0])-int(c1[0]), int(c2[1])-int(c1[1])
        if w == 0 or h == 0:
            return None
        return ObjectBBox(label, x, y, w, h, frame_index)

    def delete_old_objects(self, frames):
        for label, obj_list in self.objects.items():
            self.objects[label] = [obj for obj in obj_list if frames - obj.frames < self.queue_frames]

    def draw_bounding_boxes(self, img, c1, c2, label):
        color = random.choice(self.colors)
        cv2.rectangle(img, c1, c2, color, 5)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    dim = img.shape[1], img.shape[0]
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, dim


def set_model(cfg_file, weights_file, resolution):
    print("Loading network.....")
    model = Darknet(cfg_file)
    model.load_weights(weights_file)
    print("Network successfully loaded")

    model.net_info["height"] = resolution

    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    return model


if __name__ == '__main__':

    ini = configparser.ConfigParser()
    ini.read('ObjectDetector.ini', encoding='utf-8')

    detector = ObjectDetector(ini)

    assert detector.inp_dim % 32 == 0
    assert detector.inp_dim > 32

    video_path = ini['general']['video_path']
    cap = cv2.VideoCapture(video_path)

    assert cap.isOpened()
    count = 0
    while cap.isOpened():

        ret, frame = cap.read()
        count += 1

        if not ret:
            break
        if count % 5 != 0:
            continue

        output = detector.detect(frame, count, show_img=True)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
    # for est in output:
    #     c1 = tuple(est[1:3].int())
    #     c2 = tuple(est[3:5].int())
    #     cls = int(est[-1])
    #     label = detector.classes[cls]
    #     print(int(c1[0]), int(c1[1]), int(c2[0]), int(c2[1]))
    #     print(label)
