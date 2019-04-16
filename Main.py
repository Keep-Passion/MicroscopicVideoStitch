import cv2
import Stitcher
import os
import time

if __name__ == "__main__":
    project_address = os.getcwd()
    gt_image = cv2.imread("stitching_by_human.png", 0)
    video_address = os.path.join(os.path.join(project_address, "videos"), "FeC2-15.avi")

    # fuse_method = "notFuse"
    # fuse_method = "average"
    # fuse_method = "maximum"
    # fuse_method = "minimum"
    fuse_method = "fadeInAndFadeOut"
    # fuse_method = "trigonometric"
    # fuse_method = "multiBandBlending"

    stitcher = Stitcher.VideoStitch(video_address, roi_ratio=1.0, fuse_method=fuse_method)
    start_time = time.time()
    stitch_image = stitcher.start_stitching()
    end_time = time.time()
    stitcher.print_and_log("The total time of video stitching is {:.3f} \'s".format(end_time - start_time))
    cv2.imwrite("stitching_by_video_Fec_15.png", stitch_image)
    # justified_stitch_image = stitcher.justify_result_shape(stitch_image, gt_image)
    # stitcher.compare_result_gt(stitch_image, gt_image)
