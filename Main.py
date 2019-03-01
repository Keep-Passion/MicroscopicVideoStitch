import cv2
import glob
import Stitcher
import os


if __name__=="__main__":
    gt_image = cv2.imread("D:\\Coding_Test\\Python\\MicroscopicVideoStitch\\stitching_result.jpg")
    stitcher = Stitcher.Stitcher()
    project_address = os.getcwd()
    video_address = os.path.join(os.path.join(project_address, "videos"), "test_video.avi")
    stitch_image = stitcher.videoStitch(video_address)
    cv2.imwrite("result.jpg", stitch_image)
    stitcher.compare_result_gt(stitch_image, gt_image)