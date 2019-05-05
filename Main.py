import cv2
# from old_video_stitcher import VideoStitch
from video_stitcher import VideoStitch
from images_stitcher import ImagesStitch
import os
import time
import numpy as np
from utility import Method

def register_multi_focus_images():
    """
    Register the images with multi-foucs, it is prepare for network training
    """
    image_stitcher = ImagesStitch()
    image_stitcher.feature_method = "surf"     # "sift","surf" or "orb"
    image_stitcher.is_gpu_available = False     # use or not use gpu
    image_stitcher.search_ratio = 0.75          # 0.75 is common value for matches
    image_stitcher.offset_calculate = "mode"   # "mode" or "ransac"
    image_stitcher.offset_evaluate = 10          # 3 menas nums of matches for mode, 3.0 menas  of matches for ransac
    image_stitcher.roi_ratio = 0.2              # roi length for stitching in first direction
    project_address = os.getcwd()
    origin_address = os.path.join(os.path.join(os.path.join(project_address, "datasets"), "mufoc"), "origin")
    register_address = os.path.join(os.path.join(os.path.join(project_address, "datasets"), "mufoc"), "register")
    sub_folders = os.listdir(origin_address)
    select_list = [6, 5, 5, 4, 6, 5, 5, 5, 5, 4, 4, 4, 3, 4]
    for index, sub_folder in enumerate(sub_folders):
        input_address = os.path.join(origin_address, sub_folder)
        output_address = os.path.join(register_address, sub_folder)
        input_images_list = [os.path.join(input_address, item) for item in sorted(os.listdir(input_address))]
        select_index = select_list[index]
        image_stitcher.register_multi_focus_images(input_images_list, output_address, select_index)


def stitch_images():
    """
    Stitching all the patches in "./datasets/patch/"
    """
    image_stitcher = ImagesStitch()
    image_stitcher.feature_method = "surf"     # "sift","surf" or "orb"
    image_stitcher.is_gpu_available = False     # use or not use gpu
    image_stitcher.search_ratio = 0.75          # 0.75 is common value for matches
    image_stitcher.offset_calculate = "mode"   # "mode" or "ransac"
    image_stitcher.offset_evaluate = 10          # 3 menas nums of matches for mode, 3.0 menas  of matches for ransac
    image_stitcher.roi_ratio = 1              # roi length for stitching in first direction
    # "not_fuse", "average", "maximum", "minimum", "fade_in_fade_out", "trigonometric", "multi_band_blending"
    image_stitcher.fuse_method = "trigonometric"

    project_address = os.getcwd()
    patch_address = os.path.join(os.path.join(project_address, "datasets"), "patch")
    result_address = os.path.join(os.path.join(os.path.join(project_address, "datasets"),
                                               "result"), "origin_images_stitch_result")
    sub_folders = sorted(os.listdir(patch_address))
    time_arrays = np.zeros((15, 1))
    count_index = 0
    # Searching all sub-folders in patch_address and stitch images in each sub-folder
    for sub_folder in sub_folders:
        sub_folder_address = os.path.join(patch_address, sub_folder)
        images_name = sorted(os.listdir(sub_folder_address))
        if len(images_name) == 0:
            continue
        start_time = time.time()

        status, stitch_image = image_stitcher.start_stitching(sub_folder_address)

        end_time = time.time()
        print("The total duration of image stitching is {}'s".format(end_time-start_time))
        time_arrays[count_index, 0] = end_time-start_time
        count_index = count_index + 1
        if status:
            output_address = os.path.join(result_address,
                                          "origin_images_stitch_result" + sub_folder[5:] + ".png")
            cv2.imwrite(output_address, stitch_image)

    print("Conclusion:")
    time_arrays = time_arrays.reshape((3, 5))
    time_mean = np.average(time_arrays, axis=1)
    for index in range(0, time_mean.shape[0]):
        print("The mean duration for image stitching in {}th material is {:.2f}'s".format(index + 1, time_mean[index]))


def stitch_videos(fuse_method="trigonometric", description="", use_pre_calculate=True):
    """
    Stitching all the videos in "./datasets/video/"
    """
    video_stitcher = VideoStitch()
    video_stitcher.feature_method = "surf"      # "sift","surf" or "orb"
    video_stitcher.is_gpu_available = False     # use or not use gpu
    video_stitcher.search_ratio = 0.75          # 0.75 is common value for matches
    video_stitcher.offset_calculate = "mode"    # "mode" or "ransac"
    video_stitcher.offset_evaluate = 10        # 3 menas nums of matches for mode
    video_stitcher.roi_ratio = 0.2              # roi length for stitching in first direction
    # "not_fuse", "average", "maximum", "minimum", "fade_in_fade_out",
    # "trigonometric", "multi_band_blending", "spatial_frequency" ,"deep_fuse", "", "DSIFT"
    video_stitcher.fuse_method = fuse_method
    pre_available = None
    pre_offsets = None
    pre_regis_times = None
    project_address = os.getcwd()
    if use_pre_calculate:  # 是否使用预先计算的偏移量，以此来减少配准时间
        record_file_address = os.path.join(os.path.join(os.path.join(project_address, "datasets"),
                                                        "result"), "video_stitch_record.txt")
        pre_available, pre_offsets, pre_regis_times = video_stitcher.read_video_stitch_parameters(record_file_address)

    video_folder = os.path.join(os.path.join(project_address, "datasets"), "video")
    temp_address = sorted(os.listdir(video_folder))
    # 过滤操作
    if '.ipynb_checkpoints' in temp_address:
        temp_address.remove('.ipynb_checkpoints')
    for item in temp_address:
        if "temp" in item:
            temp_address.remove(item)
            video_stitcher.delete_folder(os.path.join(video_folder, item))
    videos_address = [os.path.join(video_folder, item) for item in temp_address]
    result_address = os.path.join(os.path.join(os.path.join(project_address, "datasets"), "result"),
                                  "origin_videos_stitch_result")
    time_arrays = np.zeros((15, 1))

    output_dir = os.path.join(os.path.join(project_address, "datasets"), "result")
    if use_pre_calculate is False:
        file_address = os.path.join(output_dir, "video_stitch_record.txt")
        f = open(file_address, 'w')
        f.close()

    count_index = 0
    # Searching all sub-folders in patch_address and stitch images in each sub-folder
    for index, video_address in enumerate(videos_address):

        video_name = os.path.basename(video_address).split(".")[0]
        start_time = time.time()
        if use_pre_calculate:
            stitch_image, record_available_list, record_offset_list = video_stitcher.start_stitching(video_address,
                                                          sample_rate=1,
                                                          use_pre_calculate=use_pre_calculate,
                                                          pre_calculate_available=pre_available[index],
                                                          pre_calculate_offset=pre_offsets[index],
                                                          pre_register_time=pre_regis_times[index])
        else:
            stitch_image, record_available_list, record_offset_list = video_stitcher.start_stitching(video_address,
                                                          sample_rate=1,
                                                          use_pre_calculate=use_pre_calculate)

        end_time = time.time()
        if use_pre_calculate:
            time_arrays[count_index, 0] = end_time - start_time + pre_regis_times[index]
            print("The total duration of video stitching is {}'s"
                  .format(end_time - start_time + pre_regis_times[index]))
        else:
            time_arrays[count_index, 0] = end_time - start_time
            print("The total duration of video stitching is {}'s".format(end_time - start_time))
            # Record the video stitch parameters
            video_stitcher.record_video_stitch_parameters(output_dir,
                                                          video_name[5:],
                                                          record_available_list,
                                                          record_offset_list,
                                                          time_arrays[count_index, 0])
        count_index = count_index + 1
        output_address = os.path.join(result_address,
                                      "origin_videos_stitch_result" + str(video_name[5:]) + ".png")
        print(output_address)
        cv2.imwrite(output_address, stitch_image)
        # only use for comparison in the paper
        out_dir = os.path.join(os.path.join(project_address, "back_up_different_methods"), video_stitcher.fuse_method + description)
        method = Method()
        method.make_out_dir(out_dir)
        print(out_dir + os.path.basename(output_address))
        cv2.imwrite(out_dir + os.path.basename(output_address), stitch_image)
        del stitch_image
    print("Conclusion:")
    print("Duration:{}".format(time_arrays.T.tolist()))
    time_arrays = time_arrays.reshape((3, 5))
    time_mean = np.average(time_arrays, axis=1)
    for index in range(0, time_mean.shape[0]):
        print("The mean duration for video stitching in {:.2f}th material is {}'s".format(index + 1, time_mean[index]))


def register_results_and_compare(description, use_pre_calculate=True):
    """
    Compare the stitching result of video and images
    The first is to register two images
    """
    image_stitcher = ImagesStitch()
    image_stitcher.feature_method = "surf"      # "sift","surf" or "orb"
    image_stitcher.is_gpu_available = False     # use or not use gpu
    image_stitcher.search_ratio = 0.95          # 0.75 is common value for matches
    image_stitcher.offset_calculate = "mode"    # "mode" or "ransac"
    image_stitcher.offset_evaluate = 3          # 3 menas nums of matches for mode, 3.0 menas  of matches for ransac
    image_stitcher.roi_ratio = 1.0              # roi length for stitching in first direction
    project_address = os.getcwd()
    images_stitch_folder = os.path.join(os.path.join(os.path.join(project_address, "datasets"),
                                                     "result"), "origin_images_stitch_result")
    videos_stitch_folder = os.path.join(os.path.join(os.path.join(project_address, "datasets"),
                                                     "result"), "origin_videos_stitch_result")
    images_address = sorted(os.listdir(images_stitch_folder))
    total_mse = []
    total_psnr = []
    total_ssim = []
    pre_offsets = None
    project_address = os.getcwd()
    if use_pre_calculate:  # 是否使用预先计算的偏移量，以此来减少配准时间
        record_file_address = os.path.join(os.path.join(os.path.join(project_address, "datasets"),
                                                        "result"), "register_record.txt")
        pre_offsets = image_stitcher.read_register_parameters(record_file_address)

    output_dir = os.path.join(os.path.join(project_address, "datasets"), "result")
    if use_pre_calculate is False:
        file_address = os.path.join(output_dir, "register_record.txt")
        f = open(file_address, 'w')
        f.close()
    for index, image_stitch_address in enumerate(images_address):
        image_index = os.path.basename(image_stitch_address)[-7:]
        print("Analyzing {}".format(image_index.split(".")[0]))
        video_stitch_address = os.path.join(videos_stitch_folder, "origin_videos_stitch_result_" + image_index)
        origin_image_stitch = cv2.imread(os.path.join(images_stitch_folder, image_stitch_address), 0)
        origin_video_stitch = cv2.imread(video_stitch_address, 0)
        if use_pre_calculate:
            status, register_video_stitch, register_image_stitch, record_offset_list = \
                image_stitcher.register_result_shape(origin_video_stitch, origin_image_stitch,
                                                     use_pre_calculate=use_pre_calculate,
                                                     pre_calculate_offset=pre_offsets[index])
        else:
            status, register_video_stitch, register_image_stitch, record_offset_list = \
                image_stitcher.register_result_shape(origin_video_stitch, origin_image_stitch,
                                                     use_pre_calculate=use_pre_calculate)
        if status:
            cv2.imwrite(os.path.join(images_stitch_folder.replace("origin", "register"),
                                     "register_images_stitch_result_" + image_index), register_image_stitch)
            cv2.imwrite(os.path.join(videos_stitch_folder.replace("origin", "register"),
                                     "register_videos_stitch_result_" + image_index), register_video_stitch)
            mse_score, psnr_score, ssim_score = \
                image_stitcher.compare_result_gt(register_video_stitch, register_image_stitch)
            total_mse.append(mse_score)
            total_psnr.append(psnr_score)
            total_ssim.append(ssim_score)
            print("  The mse is {:.4f}, psnr is {:.4f}, ssim is {:.4f}".format(mse_score, psnr_score, ssim_score))
            if use_pre_calculate is False:
                # Record the register parameters
                image_stitcher.record_register_parameters(output_dir,
                                                          image_index,
                                                          record_offset_list)
    if len(total_mse) > 0:
        print("The mse is: {}".format(total_mse))
        print("The average mse is {:.4f}".format(np.average(total_mse)))
        print("The psnr is: {}".format(total_psnr))
        print("The average psnr is {:.4f}".format(np.average(total_psnr)))
        print("The ssim is: {}".format(total_ssim))
        print("The average ssim is {:.4f}".format(np.average(total_ssim)))
        output_dir = os.path.join(os.path.join(os.path.join(project_address, "datasets"), "result"), "record")
        image_stitcher.record_evalution(output_dir, description, total_mse, total_psnr, total_ssim)

if __name__ == "__main__":
    # register_multi_focus_images()
    stitch_images()
    # "not_fuse", "average", "maximum", "minimum", "fade_in_fade_out",
    # "trigonometric", "multi_band_blending", "spatial_frequency"
    fuse_method = "trigonometric"
    description = ""
    # stitch_videos(fuse_method, description, use_pre_calculate=True)
    register_results_and_compare(fuse_method + description, use_pre_calculate=True)
