# -*- coding: utf-8 -*-
import configparser
import collections
import cv2

from ObjectDetector.ObjectDetector import ObjectDetector
from VideoProperties import VideoProperties


def draw_object_bounding_boxes(img, c1, c2, has_mask):
    if has_mask:
        color = (26, 177, 0)
        thickness = 6
    else:
        color = (10, 10, 255)
        thickness = 6
    cv2.rectangle(img, c1, c2, color, thickness)
    cv2.imshow("frame", img)
    cv2.waitKey(1)


def get_init_status_que():
    return collections.deque([True, True, True, True])


def put_status(status_que, status):
    status_que.append(status)
    status_que.popleft()


def check_status(status_que):
    cnt = 0
    for st in status_que:
        if st:
            cnt += 1
    if len(status_que) - 2 <= cnt:
        return True
    else:
        return False



if __name__ == '__main__':
    ini = configparser.ConfigParser()
    ini.read('config.ini', encoding='utf-8')

    video_properties = VideoProperties(ini)
    ini.read('person-detector.ini')
    person_detector = ObjectDetector(ini)
    ini.read('mask-detector.ini')
    mask_detector = ObjectDetector(ini)

    # video_capture = cv2.VideoCapture(video_properties.video_path)
    video_capture = cv2.VideoCapture(0)
    width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_properties.set_video_info(width, height, video_properties.resized_width)

    video_capture.set(1, video_properties.start_frame)

    resized_frame = video_properties.play_video(video_capture)

    status_que = get_init_status_que()

    while resized_frame is not None:
        persons = person_detector.detect(resized_frame, video_properties.counter, show_img=False)
        masks = mask_detector.detect(resized_frame, video_properties.counter, show_img=False)
        for person in persons:
            has_mask = False
            for mask in masks:
                # check if mask is in person bbox
                if person.x < mask.x < person.x + person.w and person.x < mask.x + mask.w < person.x + person.w and person.y < mask.y < person.y + person.h and person.y < mask.y + mask.h < person.y + person.h:
                    has_mask = True
            if not has_mask:
                print('No mask')

            # put_status(status_que, has_mask)
            draw_object_bounding_boxes(resized_frame, (person.x, person.y), (person.x + person.w, person.y + person.h), has_mask)

        cv2.imshow('frame', resized_frame)
        cv2.waitKey(1)
        print(video_properties.counter)
        resized_frame = video_properties.play_video(video_capture)
        if cv2.waitKey(1) == 27:
            break