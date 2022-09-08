import numpy as np

import sys
sys.path.append('..')
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.utils.load_text import load_text


class TOTBDataset(BaseDataset):
    """
    TOTB consisting of 225 videos for transparent objects

    Publication:
        Transparent Object Tracking Benchmark
        H. Fan, H. A. Miththanthaya, Harshit, S. R. Rajan, X. Liu, Z. Zou, Y. Lin, and H. Ling
        ICCV, 2021
        https://arxiv.org/abs/2011.10875

    Download the dataset from https://hengfan2010.github.io/projects/TOTB/
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.totb_path
        self.sequence_list = self._get_sequence_list()
        self.clean_list = self.clean_seq_list()

    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)):
            cls, _ = self.sequence_list[i].split('_')
            clean_lst.append(cls)
        return  clean_lst

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        class_name = sequence_name.split('_')[0]
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        occlusion_label_path = '{}/{}/full_occlusion.txt'.format(self.base_path, sequence_name)

        # NOTE: pandas backed seems super super slow for loading occlusion/oov masks
        full_occlusion = load_text(str(occlusion_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        out_of_view_label_path = '{}/{}/out_of_view.txt'.format(self.base_path, sequence_name)
        out_of_view = load_text(str(out_of_view_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)

        frames_path = '{}/{}/img'.format(self.base_path, sequence_name)

        frames_list = ['{}/img{:05d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        target_class = class_name
        return Sequence(sequence_name, frames_list, 'lasot', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=target_visible)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = ['Beaker_1','Beaker_10','Beaker_11','Beaker_12','Beaker_13','Beaker_14','Beaker_15','Beaker_2','Beaker_3','Beaker_4','Beaker_5','Beaker_6','Beaker_7','Beaker_8','Beaker_9', 
                         'BubbleBalloon_1','BubbleBalloon_10','BubbleBalloon_11','BubbleBalloon_12','BubbleBalloon_13','BubbleBalloon_14','BubbleBalloon_15','BubbleBalloon_2','BubbleBalloon_3','BubbleBalloon_4','BubbleBalloon_5','BubbleBalloon_6','BubbleBalloon_7','BubbleBalloon_8','BubbleBalloon_9',
                         'Bulb_1','Bulb_10','Bulb_11','Bulb_12','Bulb_13','Bulb_14','Bulb_15','Bulb_2','Bulb_3','Bulb_4','Bulb_5','Bulb_6','Bulb_7','Bulb_8','Bulb_9',
                         'Flask_1','Flask_10','Flask_11','Flask_12','Flask_13','Flask_14','Flask_15','Flask_2','Flask_3','Flask_4','Flask_5','Flask_6','Flask_7','Flask_8','Flask_9',
                         'GlassBall_1','GlassBall_10','GlassBall_11','GlassBall_12','GlassBall_13','GlassBall_14','GlassBall_15','GlassBall_2','GlassBall_3','GlassBall_4','GlassBall_5','GlassBall_6','GlassBall_7','GlassBall_8','GlassBall_9',
                         'GlassBottle_1','GlassBottle_10','GlassBottle_11','GlassBottle_12','GlassBottle_13','GlassBottle_14','GlassBottle_15','GlassBottle_2','GlassBottle_3','GlassBottle_4','GlassBottle_5','GlassBottle_6','GlassBottle_7','GlassBottle_8','GlassBottle_9',
                         'GlassCup_1','GlassCup_10','GlassCup_11','GlassCup_12','GlassCup_13','GlassCup_14','GlassCup_15','GlassCup_2','GlassCup_3','GlassCup_4','GlassCup_5','GlassCup_6','GlassCup_7','GlassCup_8','GlassCup_9',
                         'GlassJar_1','GlassJar_10','GlassJar_11','GlassJar_12','GlassJar_13','GlassJar_14','GlassJar_15','GlassJar_2','GlassJar_3','GlassJar_4','GlassJar_5','GlassJar_6','GlassJar_7','GlassJar_8','GlassJar_9',
                         'GlassSlab_1','GlassSlab_10','GlassSlab_11','GlassSlab_12','GlassSlab_13','GlassSlab_14','GlassSlab_15','GlassSlab_2','GlassSlab_3','GlassSlab_4','GlassSlab_5','GlassSlab_6','GlassSlab_7','GlassSlab_8','GlassSlab_9',
                         'JuggleBubble_1','JuggleBubble_10','JuggleBubble_11','JuggleBubble_12','JuggleBubble_13','JuggleBubble_14','JuggleBubble_15','JuggleBubble_2','JuggleBubble_3','JuggleBubble_4','JuggleBubble_5','JuggleBubble_6','JuggleBubble_7','JuggleBubble_8','JuggleBubble_9',
                         'MagnifyingGlass_1','MagnifyingGlass_10','MagnifyingGlass_11','MagnifyingGlass_12','MagnifyingGlass_13','MagnifyingGlass_14','MagnifyingGlass_15','MagnifyingGlass_2','MagnifyingGlass_3','MagnifyingGlass_4','MagnifyingGlass_5','MagnifyingGlass_6','MagnifyingGlass_7','MagnifyingGlass_8','MagnifyingGlass_9',
                         'ShotGlass_1','ShotGlass_10','ShotGlass_11','ShotGlass_12','ShotGlass_13','ShotGlass_14','ShotGlass_15','ShotGlass_2','ShotGlass_3','ShotGlass_4','ShotGlass_5','ShotGlass_6','ShotGlass_7','ShotGlass_8','ShotGlass_9',
                         'TransparentAnimal_1','TransparentAnimal_10','TransparentAnimal_11','TransparentAnimal_12','TransparentAnimal_13','TransparentAnimal_14','TransparentAnimal_15','TransparentAnimal_2','TransparentAnimal_3','TransparentAnimal_4','TransparentAnimal_5','TransparentAnimal_6','TransparentAnimal_7','TransparentAnimal_8','TransparentAnimal_9',
                         'WineGlass_1','WineGlass_10','WineGlass_11','WineGlass_12','WineGlass_13','WineGlass_14','WineGlass_15','WineGlass_2','WineGlass_3','WineGlass_4','WineGlass_5','WineGlass_6','WineGlass_7','WineGlass_8','WineGlass_9',
                         'WubbleBubble_1','WubbleBubble_10','WubbleBubble_11','WubbleBubble_12','WubbleBubble_13','WubbleBubble_14','WubbleBubble_15','WubbleBubble_2','WubbleBubble_3','WubbleBubble_4','WubbleBubble_5','WubbleBubble_6','WubbleBubble_7','WubbleBubble_8','WubbleBubble_9']

        return sequence_list
