import cv2
import Stitcher
import os
import time

if __name__ == "__main__":
    project_address = os.getcwd()
    gt_image = cv2.imread("D:\\Coding_Test\\Python\\MicroscopicVideoStitch\\stitching_by_human.jpg", 0)
    video_address = os.path.join(os.path.join(project_address, "videos"), "test_video.avi")

    # fuse_method = "notFuse"
    # fuse_method = "average"
    # fuse_method = "maximum"
    # fuse_method = "minimum"
    # fuse_method = "fadeInAndFadeOut"
    # fuse_method = "trigonometric"
    fuse_method = "multiBandBlending"

    stitcher = Stitcher.VideoStitch(video_address, fuse_method=fuse_method)
    start_time = time.time()
    stitch_image = stitcher.start_stitching()
    end_time = time.time()
    stitcher.print_and_log("The total time of video stitching is {:.3f} \'s".format(end_time - start_time))
    cv2.imwrite("stitching_by_video.jpg", stitch_image)
    stitcher.compare_result_gt(stitch_image, gt_image)
