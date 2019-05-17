import os
import cv2
import numpy as np
from scipy import signal


def make_out_dir(dir_path):
    """
    创造一个文件夹
    :param dir_path:文件夹目录
    :return:
    """
    try:
        os.makedirs(dir_path)
    except OSError:
        pass


def calculate_spatial_frequency(last_image, next_image, block_size=5):
    right_shift_kernel = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
    bottom_shift_kernel = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    last_right_shift = signal.correlate2d(last_image, right_shift_kernel, boundary='symm', mode='same')
    last_bottom_shift = signal.correlate2d(last_image, bottom_shift_kernel, boundary='symm', mode='same')
    next_right_shift = signal.correlate2d(next_image, right_shift_kernel, boundary='symm', mode='same')
    next_bottom_shift = signal.correlate2d(next_image, bottom_shift_kernel, boundary='symm', mode='same')
    last_sf = np.power(last_right_shift - last_image, 2) + np.power(last_bottom_shift - last_image, 2)
    next_sf = np.power(next_right_shift - next_image, 2) + np.power(next_bottom_shift - next_image, 2)
    add_kernel = np.ones((block_size, block_size))
    last_sf_convolve = signal.correlate2d(last_sf, add_kernel, boundary='symm', mode='same')
    next_sf_convolve = signal.correlate2d(next_sf, add_kernel, boundary='symm', mode='same')
    sf_compare = np.where(last_sf_convolve > next_sf_convolve, 1, 0)
    return sf_compare


def create_sf_in_dataset(input_dir):
    sub_folders = [os.path.join(input_dir, item) for item in sorted(os.listdir(input_dir))]
    for sub_folder in sub_folders:
        file_name = os.path.basename(sub_folder)
        last_image = cv2.imread(os.path.join(sub_folder, file_name + "_1.png"), 0)
        next_image = cv2.imread(os.path.join(sub_folder, file_name + "_2.png"), 0)
        sf_image = calculate_spatial_frequency(last_image, next_image, block_size=5) * 255
        print(os.path.join(sub_folder, file_name + "_sf.png"))
        cv2.imwrite(os.path.join(sub_folder, file_name + "_sf.png"), sf_image)

def crop_image(input_image, crop_size = 256):
    sub_images_list = []
    rows, cols = input_image.shape
    row_num = rows // crop_size
    col_num = cols // crop_size
    row_have_remain = True
    col_have_remain = True
    if rows % crop_size== 0:
        row_have_remain = False
    if cols % crop_size == 0:
        col_have_remain = False
    for i in range(row_num + 1):
        for j in range(col_num + 1):
            row_start = i * crop_size
            row_end = row_start + crop_size
            col_start = j * crop_size
            col_end = col_start + crop_size
            if i == row_num:
                if row_have_remain:
                    row_start = rows - crop_size
                    row_end = rows
                else:
                    break
            if j == col_num:
                if col_have_remain:
                    col_start = cols - crop_size
                    col_end = cols
                else:
                    continue
            sub_images_list.append(input_image[row_start: row_end, col_start:col_end])
    return sub_images_list

def crop_dataset(input_dir, output_dir):
    sub_folders = [os.path.join(input_dir, item) for item in sorted(os.listdir(input_dir))]
    for sub_folder in sub_folders:
        file_name = os.path.basename(sub_folder)
        last_image = cv2.imread(os.path.join(sub_folder, file_name + "_1.png"), 0)
        next_image = cv2.imread(os.path.join(sub_folder, file_name + "_2.png"), 0)
        sf_image = cv2.imread(os.path.join(sub_folder, file_name + "_sf.png"), 0)
        last_list = crop_image(last_image)
        next_list = crop_image(next_image)
        sf_list = crop_image(sf_image)


        for i in range(0, len(last_list)):
            output_dir_address = os.path.join(output_dir, file_name + "_" + str(i).zfill(5))
            make_out_dir(output_dir_address)
            print(os.path.join(output_dir_address, file_name + "_" + str(i).zfill(5) + "_1.png"))
            cv2.imwrite(os.path.join(output_dir_address, file_name + "_" + str(i).zfill(5) + "_1.png"), last_list[i])
            cv2.imwrite(os.path.join(output_dir_address, file_name + "_" + str(i).zfill(5) + "_2.png"), next_list[i])
            cv2.imwrite(os.path.join(output_dir_address, file_name + "_" + str(i).zfill(5) + "_sf.png"), sf_list[i])


project_address = os.getcwd()
input_dir = os.path.join(os.path.join(os.path.join(project_address, "datasets"), "used_for_nets"), "old_val")
output_dir = os.path.join(os.path.join(os.path.join(project_address, "datasets"), "used_for_nets"), "val")
crop_dataset(input_dir, output_dir)