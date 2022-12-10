import os
import numpy as np


def process_cal_mat(text):
    """Return intrinsic calibration matrix from text string

    Args:
        text (str): calibration text line in calib.txt

    Returns:
        np.array: 3-by-3 intrinsic matrix
    """
    raveled = np.fromstring(text[6:-2].replace(';', ''), dtype=float, sep=' ')
    return raveled.reshape((3,3))

def process_middlebury_calib_txt(path):
    """Return calib matrices given file path to calibration text file

    Args:
        path (str): path to stereo folder

    Returns:
        (np.array, np.array): intrinsic matrices corresp. to cam 1 and 2.
    """
    with open(os.path.join(path, 'calib.txt'), "r") as f:
        calib_1 = process_cal_mat(f.readline())
        calib_2 = process_cal_mat(f.readline())
    return calib_1, calib_2

def process_img_pair_paths(path):
    """Return concatenated image pair paths

    Args:
        path (str): path to stereo directory
    
    Returns:
        (str, str): paths corresp. to im0 and im1.
    """
    
    im0_path = os.path.join(path, 'im0.png')
    im1_path = os.path.join(path, 'im1.png')
    
    return im0_path, im1_path