import yaml

from utils import process_middlebury_calib_txt, process_img_pair_paths
from three_dim_rec import Reconstruction3D


if __name__ == "__main__":

    # process config file
    with open("confs/config.yml", "r") as f:
        config = yaml.safe_load(f)

    # parse params
    stereo_dir_path = config['input_path']['stereo_folder_path']
    method = config['keypt_match']['method']  # method for keypt matching
    method_show = config['keypt_match']['display']  # whether to display matches or not

    # load calib matrices
    K1, K2 = process_middlebury_calib_txt(stereo_dir_path)
    # process image pair paths
    path1, path2 = process_img_pair_paths(stereo_dir_path)

    # instantiate recons object
    rec = Reconstruction3D(K1, K2, None, None)

    # apply SIFT to find set of keypt pairs from stereo pair
    rec.process_img_pair_keypts(path1, path2, method, method_show)

    # normalize keypoints
    rec.normalize_keypts()

    # compute projection matrices
    rec.compute_proj_matrices()

    # triangulate
    rec.triangulate()

    # compute reprojection error
    error = rec.compute_reproj_error()
    print(f'Error: {error}')

    # show 3D plot
    rec.plot_3d_points(config)
