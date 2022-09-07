import json
import csv
import os
from glob import glob
import shutil
from pycocotools import mask as cocomask

class PrepareGotDataset:
    def __init__(self):
        self.images_dir = None
        self.number_of_im = None
        self.out_of_view = 0

    def number_of_images(self, folder_parent, folder_child):

        all_rendered_images_parent = [os.path.join(folder_parent, p) for p in os.listdir(folder_parent) if p.lower().endswith('png')]
        all_rendered_images_child = [os.path.join(folder_child, p) for p in os.listdir(folder_child) if p.lower().endswith('png')]
        if len(all_rendered_images_parent) > len(all_rendered_images_child):
            self.images_dir = folder_parent
            self.number_of_im = len(all_rendered_images_parent)
        else:
            self.images_dir = folder_child
            self.number_of_im = len(all_rendered_images_child)


    def prepare(self):
        # read all the folders
        c = glob("*/", recursive=True)
        file_list = list()
        for file_ in c:
            print(file_)
            cut_by_image_list = list()
            cover_list = list()
            meta_info_list = list()
            meta_info_list.append('[METAINFO]')
            origin = file_
            source = file_ + 'images/'

            # if folder has images
            try:
                self.number_of_images(origin, source)
                int(str(file_).split('_')[-1][:-1])
                if self.number_of_im > 0:
                    file_list.append(str(file_)[:-1])
            except:
                continue
            try:

                # opens JSON file
                data = json.load(open(origin + "/coco_annotations.json"))
                # create meta_info.ini
                all_objects = data['categories']
                distractors = list()
                for q, objects in enumerate(all_objects):
                    if q == 0:
                        new_str = 'object_class: ' + str(objects['name'])
                        meta_info_list.append(new_str)
                    else:
                        distractors.append(objects['name'])
                distractor_str = ' ,'.join(distractors)
                meta_info_list.append('distractor_class: ' + distractor_str)
                meta_info_list.append('resolution: (1280, 720)')
                # empty list
                different_objects = list()
                visible_objects_id = list()
                annotation_list = list()
                # how many objects we have
                for k_ in data['annotations']:
                    # append every id
                    different_objects.append(k_['category_id'])
                    visible_objects_id.append(k_['image_id'])
                # set of the list - all different objects
                different_objects = list(set(different_objects))
                # create a list with distractors id's
                distractor_numbers = [x for x in different_objects if x >= 10]
                # go through each annotation
                last_known_area = 0
                last_valid_image = 0
                for frame_ in data['annotations']:
                    # id
                    id_ = frame_['category_id']
                    # object to track
                    if id_ not in distractor_numbers:
                        last_valid_image += 1
                        # add bounding box
                        annotation_list.append(frame_['bbox'])
                        # bounding box positions
                        top_right_x, top_right_y, width_, height_ = frame_['bbox']
                        # segmentation mask
                        segmentation_ = frame_['segmentation']
                        # convert to area to calculate cover label
                        RLEs = cocomask.frPyObjects(segmentation_, 1280, 720)
                        area = cocomask.area(RLEs)
                        # calculate cut by image
                        if (top_right_x + width_ >= 1280) or (top_right_y + height_ >= 720):
                            cut_by_image_list.append(1)
                            self.out_of_view += 1
                            if self.out_of_view >= 5:
                                cover_list.append(cover_list[-1])
                                #break
                        else:
                            cut_by_image_list.append(0)
                            last_known_area = area
                            self.out_of_view = 0
                        perc_visibility = area / last_known_area
                        # calculate cover label
                        if perc_visibility == 1:
                            cover_list.append(8)
                        elif ((perc_visibility > 0.9) and (perc_visibility < 1)) or (perc_visibility > 1): # object gets bigger
                            cover_list.append(7)
                        elif (perc_visibility > 0.75) and (perc_visibility <= 0.9):
                            cover_list.append(6)
                        elif (perc_visibility > 0.6) and (perc_visibility <= 0.75):
                            cover_list.append(5)
                        elif (perc_visibility > 0.45) and (perc_visibility <= 0.6):
                            cover_list.append(4)
                        elif (perc_visibility > 0.3) and (perc_visibility <= 0.45):
                            cover_list.append(3)
                        elif (perc_visibility > 0.15) and (perc_visibility <= 0.3):
                            cover_list.append(2)
                        elif (perc_visibility > 0) and (perc_visibility <= 0.15):
                            cover_list.append(1)
                        else:
                            cover_list.append(0)
                # save groundtruth
                absence_list = [0] * len(annotation_list)
                with open(file_ + 'groundtruth.txt', "w") as f:
                    wr = csv.writer(f)
                    wr.writerows(annotation_list[:min(self.number_of_im, last_valid_image)])
                # save absence.label
                with open(file_ + 'absence.label', "w") as g:
                    for item in absence_list[:min(self.number_of_im, last_valid_image)]:
                        g.write("%s\n" % item)
                # save cut_by_image.label
                with open(file_ + 'cut_by_image.label', "w") as g2:
                    for item in cut_by_image_list[:min(self.number_of_im, last_valid_image)]:
                        g2.write("%s\n" % item)
                # save cover.label
                with open(file_ + 'cover.label', "w") as g3:
                    for item in cover_list[:min(self.number_of_im, last_valid_image)]:
                        g3.write("%s\n" % item)
                # write meta_info.ini
                with open(file_ + 'meta_info.ini', "w") as g4:
                    for item in meta_info_list:
                        g4.write("%s\n" % item)
                # place images in parent folder
                new_names = [visible_objects_id]
                if self.images_dir == source:
                    files = os.listdir(source)
                    files = list(sorted(files))
                    c = 0
                    #copy = False
                    for file in files:
                        #print(int(str(file).split('_')[-1][:-4]))
                        file_new = str(c).zfill(6) + '.png'

                        if int(str(file).split('_')[-1][:-4]) in visible_objects_id:
                            try:
                                file_name = os.path.join(source, file)
                                #print(file_name)
                                shutil.move(file_name, origin)
                                new_file_name = file_name.split('/')
                                if new_file_name.pop(-2) == 'images':
                                    os.rename('/'.join(new_file_name), new_file_name[-2] + '/' + file_new)
                                    c += 1
                                else:
                                    AssertionError
                            except Exception as p:
                                print(p)
            except Exception as e:
                print(e)
                pass
        file_list = list(sorted(file_list))

        all_files_split = len(file_list)

        train_num = [str(x) for x in range(all_files_split)]

        with open("got10k_train_full_split.txt", "w") as t:
            for item in train_num:
                t.write("%s\n" % item)

        with open('list.txt', "w") as g:
            for item in file_list:
                g.write("%s\n" % item)


if __name__ == "__main__":
    p = PrepareGotDataset()
    p.prepare()
