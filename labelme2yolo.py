from pathlib import Path
import sys
import os
from datetime import datetime
import json
import yaml
from shutil import copyfile, move

import numpy as np
import cv2
import imagesize
import exifread

from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description='Convert Labelme to YOLO')
parser.add_argument('--data', default='config.yaml', type=str, required=True, help='yaml file')
args = parser.parse_args()

with open(args.data) as f:
    data = yaml.safe_load(f)  # data dict
base_dir = data['path']
dir_name = data['dirname']
resize_value = data['resize_value']
labels = data['labels']

# set directory for saving results
save_dir = data['save_dir'] if data['save_dir'] else datetime.now().strftime('%b%d_%H-%M-%S')
save_dir = save_dir if save_dir[0]=='/' else f"results/{save_dir}"
os.makedirs(f"{save_dir}/images", exist_ok=True)
os.makedirs(f"{save_dir}/error", exist_ok=True)
os.makedirs(f"{save_dir}/labels", exist_ok=True)
copyfile(args.data, os.path.join(save_dir, f"{dir_name}.yaml")) # save data file (.yaml)


def copy_img(base_path, dirname, target_path):
    print(f"\n>> copy_img! ")
    target_dir_list = os.listdir(f"{target_path}/images")
    print(f"[{dirname}] 작업 시작")
    if dirname in target_dir_list:
        print(f"- 이미 복사됨!")
        return
    
    # img 복사
    img_target_path = f"{target_path}/images"
    if not Path(f"{img_target_path}/{dirname}").exists():
        os.mkdir(f"{img_target_path}/{dirname}")
        os.system(f"find {base_path}/{dirname} -iname '*.jpg' -exec cp {{}} {img_target_path}/{dirname} \;")
        # os.system(f"cp {base_path}/{dirname}/*.jpg {img_target_path}/{dirname}")
        print(f"- image 복사 완료!")

def convert_yolo_txt(base_path, dirname, target_path):
    print(f"\n>> convert_yolo_txt! ")
    target_dir_list = os.listdir(f"{target_path}/labels")
    print(f"[{dirname}] 작업 시작")
    if dirname in target_dir_list:
        print(f"- 이미 복사됨!")
        return

    error_file = ''

    # labels 복사
    json_target_path = f"{target_path}/labels"
    if not Path(f"{json_target_path}/{dirname}").exists():
        os.mkdir(f"{json_target_path}/{dirname}")

    dict_label = {} # for label check

    json_list = list(Path(f"{base_path}/{dirname}").rglob("*.json"))
    # json 읽기
    for jsonfile in tqdm(json_list):
        with open(jsonfile) as f:
            try:
                data = json.load(f)
            except:
                error_file+=f"\n--  crack json : {jsonfile}"
                if os.path.exists(str(jsonfile)[:-4]+'jpg'):
                    move(str(jsonfile)[:-4]+'jpg', f"{save_dir}/error/{jsonfile.name[:-4]+'jpg'}")
                    # os.remove(f"{str(jsonfile)[:-4]+'jpg'}")
                if os.path.exists(str(jsonfile)):
                    move(str(jsonfile), f"{save_dir}/error/{jsonfile.name}")
                    # os.remove(f"{str(jsonfile)}")
                # print(f"\n--  crack json : {jsonfile}")
                continue

        # image_path = f"{target_path}/images/{dirname}/{jsonfile.name[:-4]+'jpg'}"
        image_path = str(jsonfile)[:-4]+'jpg'

        try: 
            w0, h0 = imagesize.get(image_path)
        except:
            error_file+=f"\n--    no image : {image_path}"
            if os.path.exists(image_path):
                move(str(jsonfile), f"{save_dir}/error/{jsonfile.name[:-4]+'jpg'}")
                # os.remove(f"{image_path}")
            if os.path.exists(str(jsonfile)):
                move(str(jsonfile), f"{save_dir}/error/{jsonfile.name}")
                # os.remove(f"{str(jsonfile)}")
            # print(f"\n--    no image : {image_path}")
            continue
        if w0 == -1 or h0 == -1:
            error_file+=f"\n-- crack image : {image_path}"
            if os.path.exists(image_path):
                move(str(jsonfile), f"{save_dir}/error/{jsonfile.name[:-4]+'jpg'}")
                # os.remove(f"{image_path}")
            if os.path.exists(str(jsonfile)):
                move(str(jsonfile), f"{save_dir}/error/{jsonfile.name}")
                # os.remove(f"{str(jsonfile)}")
            # print(f"\n-- crack image : {image_path}")
            continue

        with open(image_path, 'rb') as f:
            try:
                tags = exifread.process_file(f)
                ori = tags['Image Orientation'].values[0]
                if ori in [6,8]:
                    w0, h0 = h0, w0
            except:
                # print(f"-- EXIF error : {image_path}")
                pass

        # cv2
        # img = cv2.imread(str(jsonfile)[:-4]+'jpg')
        # h, w = img.shape[:2]

        # json 
        # w = data['imageWidth']
        # h = data['imageHeight']

        if len(data['shapes']) == 0:
            error_file+=f"\n--     no bbox : {jsonfile}"
            if os.path.exists(image_path):
                move(str(jsonfile), f"{save_dir}/error/{jsonfile.name[:-4]+'jpg'}")
                # os.remove(f"{image_path}")
            if os.path.exists(str(jsonfile)):
                move(str(jsonfile), f"{save_dir}/error/{jsonfile.name}")
                # os.remove(f"{str(jsonfile)}")
            continue

        # txt 생성
        with open(f"{json_target_path}/{dirname}/{jsonfile.name[:-4]}txt", 'w') as f:
            count = 0

            for bbox in data['shapes']:
                try:
                    if bbox['label'] == 'gbg':
                        continue
                    l_class = labels[bbox['label']]
                    points = np.array(bbox['points'])

                    if len(points) == 0:
                        error_file+=f"\n--    no coord : {jsonfile}"
                        continue

                    min_p = (max(np.min(points[:,0]),0), max(np.min(points[:,1]),0))
                    max_p = (min(np.max(points[:,0]),w0), min(np.max(points[:,1]),h0))
                    width = max_p[0]-min_p[0]
                    height = max_p[1]-min_p[1]

                    if width*height < 50:
                        error_file+=f"\n--  small bbox : {jsonfile}"
                        continue

                    l_bbox = [(max_p[0]+min_p[0])/2/w0, (max_p[1]+min_p[1])/2/h0, width/w0, height/h0]
                except:
                    error_file+=f"\n-- label error : {jsonfile}"
                    if os.path.exists(image_path):
                        move(str(jsonfile), f"{save_dir}/error/{jsonfile.name[:-4]+'jpg'}")
                        # os.remove(f"{image_path}")
                    if os.path.exists(str(jsonfile)):
                        move(str(jsonfile), f"{save_dir}/error/{jsonfile.name}")
                        # os.remove(f"{str(jsonfile)}")
                    break

                # bbox 복사
                f.write(f"{l_class} {l_bbox[0]} {l_bbox[1]} {l_bbox[2]} {l_bbox[3]}\n")

                # label check
                if l_class not in dict_label.keys():
                    dict_label[l_class] = 1
                else:
                    dict_label[l_class] += 1
                count += 1
    print(error_file)
    print(f"- [{dirname}] label 출력 : {dict_label.keys()}")

def check_num(dirname, target_path): # 개수 check
    print(f"\n>> check_num! ")
    img_list = list(Path(f"{target_path}/images/{dirname}").rglob("*.jpg"))
    img_dict = {i.name[:-4] : i for i in img_list}

    txt_list = list(Path(f"{target_path}/labels/{dirname}").rglob("*.txt"))
    txt_dict = {i.name[:-4] : i for i in txt_list}

    for txt_file in txt_list:
        if txt_file.name[:-4] not in img_dict:
            if os.path.exists(txt_file):
                os.remove(txt_file)

    for img_file in img_list:
        if img_file.name[:-4] not in txt_dict:
            if os.path.exists(img_file):
                os.remove(img_file)
        
    img_list = list(Path(f"{target_path}/images/{dirname}").rglob("*.jpg"))
    txt_list = list(Path(f"{target_path}/labels/{dirname}").rglob("*.txt"))

    if len(img_list) == len(txt_list):
        print(f"[{dirname}] - OK!")
    else :
        print(f"[{dirname}] - Fail! ( img: {len(img_list)}, txt: {len(txt_list)} )")

def check_img(dirname, target_path):
    print(f"\n>> check_img! ")
    print(f"[{dirname}] 시작!")
    img_list = list(Path(f"{target_path}/images/{dirname}").rglob("*.jpg"))

    for imgfile in tqdm(img_list):
        try:
            img = cv2.imread(str(imgfile))
        except Exception as e:
            print(f"error -> {imgfile}")
    print(f"- 끝!")

def resize_img(dirname, target_path, resize_size=None):
    print(f"\n>> resize_img! ")
    if resize_size:
        print(f"[{dirname}] 작업 시작")
        img_target_path = f"{target_path}/images/{dirname}"
        img_list = list(Path(img_target_path).glob('*.jpg'))

        error_file = ''

        resize_count = 0
        pass_count = 0
        for img_path in tqdm(img_list):
            try:
                img = cv2.imread(str(img_path))
            except:
                error_file+=f"\n--    no image : {str(img_path)}"
                if os.path.exists(str(img_path)):
                    os.remove(f"{str(img_path)}")
                # print(f"\n--    no image : {str(img_path)}")
                continue

            if img is None:
                error_file+=f"\n-- crack image : {str(img_path)}"
                if os.path.exists(str(img_path)):
                    os.remove(f"{str(img_path)}")
                # print(f"\n-- crack image : {str(img_path)}")
                continue

            h0, w0 = img.shape[:2]
            if h0 == -1 or w0 == -1:
                error_file+=f"\n-- crack image : {str(img_path)}"
                if os.path.exists(str(img_path)):
                    os.remove(f"{str(img_path)}")
                # print(f"\n-- crack image : {str(img_path)}")
                continue


            if h0 == resize_size or w0 == resize_size:
                continue

            r = resize_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                img_resized = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                                interpolation=cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR)
                cv2.imwrite(str(img_path), img_resized)
                resize_count +=1
            else :
                pass_count += 1
                
        print(error_file)
        print(f"- [{dirname}] image resize 완료! (resize:{resize_count}, pass:{pass_count})")
    else:
        print(f"- resize value is wrong!")

def add_dataset(base_path, dirname, target_path, resize_v):
    convert_yolo_txt(base_path, dirname, target_path)
    copy_img(base_path, dirname, target_path)
    resize_img(dirname, target_path, resize_size=resize_v)
    check_num(dirname, target_path)
    # check_img(target_path, new_dir_list)
    print(f">> {dirname} - added! ")

if __name__ == "__main__":   
    add_dataset(base_dir, dir_name, save_dir, resize_value)
