import cv2
# from old_video_stitcher import VideoStitch
from video_stitcher import VideoStitch
from images_stitcher import ImagesStitch
import os
import time
import numpy as np


def stitch_images():
    """
    Stitching all the patches in ".\datasets\patch\"
    """
    image_stitcher = ImagesStitch()
    image_stitcher.feature_method = "surf"     # "sift","surf" or "orb"
    image_stitcher.is_gpu_available = True     # use or not use gpu
    image_stitcher.search_ratio = 0.75          # 0.75 is common value for matches
    image_stitcher.offset_calculate = "mode"   # "mode" or "ransac"
    image_stitcher.offset_evaluate = 3          # 3 menas nums of matches for mode, 3.0 menas  of matches for ransac
    image_stitcher.roi_ratio = 0.2              # roi length for stitching in first direction
    # "not_fuse", "average", "maximum", "minimum", "fade_in_fade_out", "trigonometric", "multi_band_blending"
    image_stitcher.fuse_method = "fade_in_fade_out"

    project_address = os.getcwd()
    patch_address = os.path.join(os.path.join(project_address, "datasets"), "patch")
    result_address = os.path.join(os.path.join(os.path.join(project_address, "datasets"), "result"),
                                  "origin_images_stitch_result")
    sub_folders = os.listdir(patch_address)
    time_arrays = np.zeros((20, 1))
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
                                          "origin_images_stitch_result" + sub_folder[5:] + ".tif")
            cv2.imwrite(output_address, stitch_image)

    print("Conclusion:")
    time_arrays = time_arrays.reshape((4, 5))
    time_mean = np.average(time_arrays, axis=1)
    for index in range(0, time_mean.shape[0]):
        print("The mean duration for image stitching in {}th material is {}'s".format(index + 1, time_mean[index]))


def stitch_videos():
    """
    Stitching all the videos in ".\datasets\video\"
    """
    video_stitcher = VideoStitch()
    video_stitcher.feature_method = "surf"      # "sift","surf" or "orb"
    video_stitcher.is_gpu_available = False     # use or not use gpu
    video_stitcher.search_ratio = 0.75          # 0.75 is common value for matches
    video_stitcher.offset_calculate = "mode"    # "mode" or "ransac"
    video_stitcher.offset_evaluate = 3          # 3 menas nums of matches for mode
    video_stitcher.roi_ratio = 0.2              # roi length for stitching in first direction
    # "not_fuse", "average", "maximum", "minimum", "fade_in_fade_out", "trigonometric", "multi_band_blending"
    video_stitcher.fuse_method = "fade_in_fade_out"

    project_address = os.getcwd()
    video_folder = os.path.join(os.path.join(project_address, "datasets"), "video")
    temp_address = sorted(os.listdir(video_folder))
    videos_address = [os.path.join(video_folder, item) for item in temp_address]
    result_address = os.path.join(os.path.join(os.path.join(project_address, "datasets"), "result"),
                                  "origin_videos_stitch_result")
    time_arrays = np.zeros((20, 1))
    count_index = 0
    # Searching all sub-folders in patch_address and stitch images in each sub-folder
    for video_address in videos_address:

        video_name = os.path.basename(video_address).split(".")[0]
        start_time = time.time()

        status, stitch_image = video_stitcher.start_stitching(video_address, sample_rate=1)

        end_time = time.time()
        print("The total duration of video stitching is {}'s".format(end_time-start_time))
        time_arrays[count_index, 0] = end_time-start_time
        count_index = count_index + 1
        if status:
            output_address = os.path.join(result_address,
                                          "origin_videos_stitch_result" + str(video_name[5:]) + ".tif")
            cv2.imwrite(output_address, stitch_image)

    print("Conclusion:")
    time_arrays = time_arrays.reshape((4, 5))
    time_mean = np.average(time_arrays, axis=1)
    for index in range(0, time_mean.shape[0]):
        print("The mean duration for video stitching in {}th material is {}'s".format(index + 1, time_mean[index]))


def register_and_compare():
    """
    Compare the stitching result of video and images
    The first is to register two images
    """
    video_stitcher = VideoStitch()
    video_stitcher.feature_method = "surf"     # "sift","surf" or "orb"
    video_stitcher.is_gpu_available = True     # use or not use gpu
    video_stitcher.search_ratio = 0.75          # 0.75 is common value for matches
    video_stitcher.offset_calculate = "mode"   # "mode" or "ransac"
    video_stitcher.offset_evaluate = 10          # 3 menas nums of matches for mode, 3.0 menas  of matches for ransac
    video_stitcher.roi_ratio = 1.0              # roi length for stitching in first direction
    # use_const_offset = True
    # conset_offset = []
    project_address = os.getcwd()
    images_stitch_folder = os.path.join(os.path.join(os.path.join(project_address, "datasets"), "result"),
                                "origin_images_stitch_result")
    videos_stitch_folder = os.path.join(os.path.join(os.path.join(project_address, "datasets"), "result"),
                                "origin_videos_stitch_result")
    images_address = sorted(os.listdir(images_stitch_folder))
    total_mse = []
    total_psnr = []
    total_ssim = []
    for image_stitch_address in images_address:
        image_index = os.path.basename(image_stitch_address)[-7:]
        print("Analyzing {}".format(image_index.split(".")[0]))
        video_stitch_address = os.path.join(videos_stitch_folder, "origin_videos_stitch_result_" + image_index)
        origin_image_stitch = cv2.imread(os.path.join(images_stitch_folder, image_stitch_address), 0)
        origin_video_stitch = cv2.imread(video_stitch_address, 0)
        status, register_video_stitch, register_image_stitch = \
            video_stitcher.justify_result_shape(origin_video_stitch, origin_image_stitch)
        if status:
            cv2.imwrite(os.path.join(images_stitch_folder.replace("origin", "register"),
                                     "register_images_stitch_result_" + image_index), register_image_stitch)
            cv2.imwrite(os.path.join(videos_stitch_folder.replace("origin", "register"),
                                     "register_videos_stitch_result_" + image_index), register_video_stitch)
            mse_score, psnr_score, ssim_score = \
                video_stitcher.compare_result_gt(register_video_stitch, register_image_stitch)
            total_mse.append(mse_score)
            total_psnr.append(psnr_score)
            total_ssim.append(ssim_score)
            print("  The mse is {}, psnr is {}, ssim is {}".format(mse_score, psnr_score, ssim_score))
    print("The average mse is {}".format(np.average(total_mse)))
    print("The average psnr is {}".format(np.average(total_psnr)))
    print("The average ssim is {}".format(np.average(total_ssim)))

if __name__ == "__main__":
    # stitch_images()
    stitch_videos()
    # register_and_compare()
