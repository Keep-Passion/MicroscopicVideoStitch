import numpy as np
import cv2
from scipy.stats import mode
import time
import os
import glob
import skimage.measure
from numba import jit
import ImageUtility as Utility
import ImageFusion
import time

class ImageFeature():
    # 用来保存串行全局拼接中的第二张图像的特征点和描述子，为后续加速拼接使用
    kps = None
    features = None
    is_break = False


class Stitcher(Utility.Method):
    '''
	    图像拼接类，包括所有跟材料显微组织图像配准相关函数
	'''
    fuse_method = "notFuse"
    last_image_feature = ImageFeature()
    sample_rate = 3

    # isIncreInFrameSearch = True         # 判断是否通过增长区域搜索
    # direction = 1   # 1： 第一张图像在上，第二张图像在下；   2： 第一张图像在左，第二张图像在右；
    #                 # 3： 第一张图像在下，第二张图像在上；   4： 第一张图像在右，第二张图像在左；
    # directIncre = 1
    # def directionIncrease(self, direction):
    #     direction += self.directIncre
    #     if direction == 5:
    #         direction = 1
    #     if direction == 0:
    #         direction = 4
    #     return direction


    def videoStitch(self, video_address):

        # *********** 对视频采样，将采样的所有图像输出到与视频文件同目录的temp文件夹 ***********
        # 建立 temp 文件夹
        file_dir = os.path.dirname(video_address)
        sample_dir = os.path.join(file_dir, "temp")
        self.make_out_dir(sample_dir)

        # 将 video 采样到 temp 文件夹
        self.print_and_log("Video name:" + video_address)
        self.print_and_log("Sampling rate:" + str(self.sample_rate))
        self.print_and_log("We save sampling images in " + sample_dir)
        self.print_and_log("Sampling images ...")
        cap = cv2.VideoCapture(video_address)
        frame_num = 0
        save_num = 0
        start_time = time.time()
        while True:
            ret, origin_frame = cap.read()
            if ret is False:
                break
            frame_num = frame_num + 1
            if frame_num % self.sample_rate == 0:
                gray_frame = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2GRAY)
                save_num = save_num + 1
                cv2.imwrite(os.path.join(sample_dir, str(save_num).zfill(10) + ".png"), gray_frame)
        cap.release()
        end_time = time.time()
        self.print_and_log("Sampled done, The time of sampling is {:.3f} \'s".format(end_time - start_time))

        # **************************** 对采样后的文件进行拼接 ****************************
        # 开始拼接文件夹下的图片
        images_address_list = glob.glob(os.path.join(sample_dir, "*.png"))
        offset_list = []
        is_image_available = []
        start_time = time.time()
        imageA = cv2.imdecode(np.fromfile(images_address_list[0], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        kpsA, featuresA = self.calculate_feature(imageA)
        self.last_image_feature.kps = kpsA
        self.last_image_feature.features = featuresA
        is_image_available.append(True)
        for file_index in range(1, len(images_address_list)):
            self.print_and_log("    Analyzing {}th frame and the name is {}".format(file_index, os.path.basename(images_address_list[file_index])))
            status, offset = self.calculate_offset_by_feature(images_address_list[file_index])
            if status is False:
                self.print_and_log("    {}th frame can not be stitched".format(file_index))
                is_image_available.append(False)
            else:
                self.print_and_log("    {}th frame can be stitched, the offset is {}".format(file_index, offset))
                is_image_available.append(True)
                offset_list.append(offset)
        end_time = time.time()
        self.print_and_log("The time of registering is {:.3f} \'s".format(end_time - start_time))

        # *************************** stitching and fusing ***************************
        self.print_and_log("start stitching")
        stitch_image = self.get_stitch_by_offset(images_address_list, is_image_available, offset_list)
        end_time = time.time()
        self.print_and_log("The time of fusing is {:.3f} \'s".format(end_time - start_time))
        return stitch_image


    def calculate_feature(self, origin_image):
        if self.is_enhance == True:
            if self.is_clahe == True:
                clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=(self.tileSize, self.tileSize))
                origin_image = clahe.apply(origin_image)
            elif self.is_clahe == False:
                origin_image = cv2.equalizeHist(origin_image)
        kps, features = self.detect_and_describe(origin_image, featureMethod=self.featureMethod)
        return kps, features


    def calculate_offset_by_feature(self, image_address):
        '''
        Stitch two images
        :param images: [imageA, imageB]
        :param registrateMethod: list:
        :param fuse_method:
        :param direction: stitching direction
        :return:
        '''
        offset = [0, 0]
        status = False

        # get the feature points
        kpsA = self.last_image_feature.kps
        featuresA = self.last_image_feature.features
        imageB = cv2.imdecode(np.fromfile(image_address, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        kpsB, featuresB = self.calculate_feature(imageB)
        if featuresA is not None and featuresB is not None:
            matches = self.match_descriptors(featuresA, featuresB)
            # match all the feature points
            if self.offsetCaculate == "mode":
                (status, offset) = self.getOffsetByMode(kpsA, kpsB, matches, offsetEvaluate = self.offsetEvaluate)
            elif self.offsetCaculate == "ransac":
                (status, offset, adjustH) = self.getOffsetByRansac(kpsA, kpsB, matches, offsetEvaluate = self.offsetEvaluate)
        if status is True:
            self.last_image_feature.kps = kpsB
            self.last_image_feature.features = featuresB
        return (status, offset)


    # def calculateFrameOffsetForFeatureSearchIncre(self, images):
    #     '''
    #     Stitch two images
    #     :param images: [imageA, imageB]
    #     :param registrateMethod: list:
    #     :param fuse_method:
    #     :param direction: stitching direction
    #     :return:
    #     '''
    #
    #     (imageA, imageB) = images
    #     offset = [0, 0]
    #     status = False
    #     maxI = (np.floor(0.5 / self.roiRatio) + 1).astype(int)+ 1
    #     iniDirection = self.direction
    #     localDirection = iniDirection
    #     for i in range(1, maxI):
    #         # self.printAndWrite("  i=" + str(i) + " and maxI="+str(maxI))
    #         while(True):
    #             # get the roi region of images
    #             # self.printAndWrite("  localDirection=" + str(localDirection))
    #             roiImageA = self.getROIRegionForIncreMethod(imageA, direction=localDirection, order="first", searchRatio = i * self.roiRatio)
    #             roiImageB = self.getROIRegionForIncreMethod(imageB, direction=localDirection, order="second", searchRatio = i * self.roiRatio)
    #
    #             if self.isEnhance == True:
    #                 if self.isClahe == True:
    #                     clahe = cv2.createCLAHE(clipLimit=self.clipLimit,tileGridSize=(self.tileSize, self.tileSize))
    #                     roiImageA = clahe.apply(roiImageA)
    #                     roiImageB = clahe.apply(roiImageB)
    #                 elif self.isClahe == False:
    #                     roiImageA = cv2.equalizeHist(roiImageA)
    #                     roiImageB = cv2.equalizeHist(roiImageB)
    #             # get the feature points
    #             kpsA, featuresA = self.detectAndDescribe(roiImageA, featureMethod=self.featureMethod)
    #             kpsB, featuresB = self.detectAndDescribe(roiImageB, featureMethod=self.featureMethod)
    #             if featuresA is not None and featuresB is not None:
    #                 matches = self.matchDescriptors(featuresA, featuresB)
    #                 # match all the feature points
    #                 if self.offsetCaculate == "mode":
    #                     (status, offset) = self.getOffsetByMode(kpsA, kpsB, matches, offsetEvaluate = self.offsetEvaluate)
    #                 elif self.offsetCaculate == "ransac":
    #                     (status, offset, adjustH) = self.getOffsetByRansac(kpsA, kpsB, matches, offsetEvaluate = self.offsetEvaluate)
    #             if status == True:
    #                 break
    #             else:
    #                 localDirection = self.directionIncrease(localDirection)
    #             if localDirection == iniDirection:
    #                 break
    #         if status == True:
    #             if localDirection == 1:
    #                 offset[0] = offset[0] + imageA.shape[0] - int(i * self.roiRatio * imageA.shape[0])
    #             elif localDirection == 2:
    #                 offset[1] = offset[1] + imageA.shape[1] - int(i * self.roiRatio * imageA.shape[1])
    #             elif localDirection == 3:
    #                 offset[0] = offset[0] - (imageB.shape[0] - int(i * self.roiRatio * imageB.shape[0]))
    #             elif localDirection == 4:
    #                 offset[1] = offset[1] - (imageB.shape[1] - int(i * self.roiRatio * imageB.shape[1]))
    #             self.direction = localDirection
    #             break
    #     if status == False:
    #         return (status, "  The two images can not match")
    #     elif status == True:
    #         self.printAndWrite("  The offset of stitching: dx is " + str(offset[0]) + " dy is " + str(offset[1]))
    #         return (status, offset)


    def get_stitch_by_offset(self, fileList, is_image_available, offsetListOrigin):
        '''
        通过偏移量列表和文件列表得到最终的拼接结果
        :param fileList: 图像列表
        :param offsetListOrigin: 偏移量列表
        :return: ndaarry，图像
        '''
        # 如果你不细心，不要碰这段代码
        # 已优化到根据指针来控制拼接，CPU下最快了
        dxSum = dySum = 0
        imageList = []
        # imageList.append(cv2.imread(fileList[0], 0))
        imageList.append(cv2.imdecode(np.fromfile(fileList[0], dtype=np.uint8), cv2.IMREAD_GRAYSCALE))
        resultRow = imageList[0].shape[0]         # 拼接最终结果的横轴长度,先赋值第一个图像的横轴
        resultCol = imageList[0].shape[1]         # 拼接最终结果的纵轴长度,先赋值第一个图像的纵轴
        offsetListOrigin.insert(0, [0, 0])        # 增加第一张图像相对于最终结果的原点的偏移量

        rangeX = [[0,0] for x in range(len(offsetListOrigin))]  # 主要用于记录X方向最大最小边界
        rangeY = [[0, 0] for x in range(len(offsetListOrigin))] # 主要用于记录Y方向最大最小边界
        offsetList = offsetListOrigin.copy()
        rangeX[0][1] = imageList[0].shape[0]
        rangeY[0][1] = imageList[0].shape[1]

        for i in range(1, len(offsetList)):
            if is_image_available[i] is False:
                continue
            # self.printAndWrite("  stitching " + str(fileList[i]))
            # 适用于流形拼接的校正,并更新最终图像大小
            # tempImage = cv2.imread(fileList[i], 0)
            tempImage = cv2.imdecode(np.fromfile(fileList[i], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            dxSum = dxSum + offsetList[i][0]
            dySum = dySum + offsetList[i][1]
            # self.printAndWrite("  The dxSum is " + str(dxSum) + " and the dySum is " + str(dySum))
            if dxSum <= 0:
                for j in range(0, i):
                    offsetList[j][0] = offsetList[j][0] + abs(dxSum)
                    rangeX[j][0] = rangeX[j][0] + abs(dxSum)
                    rangeX[j][1] = rangeX[j][1] + abs(dxSum)
                resultRow = resultRow + abs(dxSum)
                rangeX[i][1] = resultRow
                dxSum = rangeX[i][0] = offsetList[i][0] = 0
            else:
                offsetList[i][0] = dxSum
                resultRow = max(resultRow, dxSum + tempImage.shape[0])
                rangeX[i][1] = resultRow
            if dySum <= 0:
                for j in range(0, i):
                    offsetList[j][1] = offsetList[j][1] + abs(dySum)
                    rangeY[j][0] = rangeY[j][0] + abs(dySum)
                    rangeY[j][1] = rangeY[j][1] + abs(dySum)
                resultCol = resultCol + abs(dySum)
                rangeY[i][1] = resultCol
                dySum = rangeY[i][0] = offsetList[i][1] = 0
            else:
                offsetList[i][1] = dySum
                resultCol = max(resultCol, dySum + tempImage.shape[1])
                rangeY[i][1] = resultCol
            imageList.append(tempImage)
        stitchResult = np.zeros((resultRow, resultCol), np.int) - 1
        self.print_and_log("  The rectified offsetList is " + str(offsetList))

        # 如上算出各个图像相对于原点偏移量，并最终计算出输出图像大小，并构造矩阵，如下开始赋值
        for i in range(0, len(offsetList)):
            self.print_and_log("  stitching " + str(fileList[i]))
            if i == 0:
                stitchResult[offsetList[0][0]: offsetList[0][0] + imageList[0].shape[0], offsetList[0][1]: offsetList[0][1] + imageList[0].shape[1]] = imageList[0]
            else:
                if self.fuse_method == "notFuse":
                    # 适用于无图像融合，直接覆盖
                    # self.printAndWrite("Stitch " + str(i+1) + "th, the roi_ltx is " + str(offsetList[i][0]) + " and the roi_lty is " + str(offsetList[i][1]))
                    stitchResult[offsetList[i][0]: offsetList[i][0] + imageList[i].shape[0], offsetList[i][1]: offsetList[i][1] + imageList[i].shape[1]] = imageList[i]
                else:
                    # 适用于图像融合算法，切出 roiA 和 roiB 供图像融合
                    minOccupyX = rangeX[i-1][0]
                    maxOccupyX = rangeX[i-1][1]
                    minOccupyY = rangeY[i-1][0]
                    maxOccupyY = rangeY[i-1][1]
                    # self.printAndWrite("Stitch " + str(i + 1) + "th, the offsetList[i][0] is " + str(
                    #     offsetList[i][0]) + " and the offsetList[i][1] is " + str(offsetList[i][1]))
                    # self.printAndWrite("Stitch " + str(i + 1) + "th, the minOccupyX is " + str(
                    #     minOccupyX) + " and the maxOccupyX is " + str(maxOccupyX) + " and the minOccupyY is " + str(
                    #     minOccupyY) + " and the maxOccupyY is " + str(maxOccupyY))
                    roi_ltx = max(offsetList[i][0], minOccupyX)
                    roi_lty = max(offsetList[i][1], minOccupyY)
                    roi_rbx = min(offsetList[i][0] + imageList[i].shape[0], maxOccupyX)
                    roi_rby = min(offsetList[i][1] + imageList[i].shape[1], maxOccupyY)
                    # self.printAndWrite("Stitch " + str(i + 1) + "th, the roi_ltx is " + str(
                    #     roi_ltx) + " and the roi_lty is " + str(roi_lty) + " and the roi_rbx is " + str(
                    #     roi_rbx) + " and the roi_rby is " + str(roi_rby))
                    roiImageRegionA = stitchResult[roi_ltx:roi_rbx, roi_lty:roi_rby].copy()
                    stitchResult[offsetList[i][0]: offsetList[i][0] + imageList[i].shape[0], offsetList[i][1]: offsetList[i][1] + imageList[i].shape[1]] = imageList[i]
                    roiImageRegionB = stitchResult[roi_ltx:roi_rbx, roi_lty:roi_rby].copy()
                    stitchResult[roi_ltx:roi_rbx, roi_lty:roi_rby] = self.fuseImage([roiImageRegionA, roiImageRegionB], offsetListOrigin[i][0], offsetListOrigin[i][1])
        stitchResult[stitchResult == -1] = 0
        return stitchResult.astype(np.uint8)


    def fuseImage(self, images, dx, dy):
        (imageA, imageB) = images
        # cv2.namedWindow("A", 0)
        # cv2.namedWindow("B", 0)
        # cv2.imshow("A", imageA.astype(np.uint8))
        # cv2.imshow("B", imageB.astype(np.uint8))
        fuseRegion = np.zeros(imageA.shape, np.uint8)
        # imageA[imageA == 0] = imageB[imageA == 0]
        # imageB[imageB == 0] = imageA[imageB == 0]
        imageFusion = ImageFusion.ImageFusion()
        if self.fuse_method == "notFuse":
            imageB[imageA == -1] = imageB[imageA == -1]
            imageA[imageB == -1] = imageA[imageB == -1]
            fuseRegion = imageB
        elif self.fuse_method == "average":
            imageA[imageA == 0] = imageB[imageA == 0]
            imageB[imageB == 0] = imageA[imageB == 0]
            imageA[imageA == -1] = 0
            imageB[imageB == -1] = 0
            fuseRegion = imageFusion.fuseByAverage([imageA, imageB])
        elif self.fuse_method == "maximum":
            imageA[imageA == 0] = imageB[imageA == 0]
            imageB[imageB == 0] = imageA[imageB == 0]
            imageA[imageA == -1] = 0
            imageB[imageB == -1] = 0
            fuseRegion = imageFusion.fuseByMaximum([imageA, imageB])
        elif self.fuse_method == "minimum":
            imageA[imageA == 0] = imageB[imageA == 0]
            imageB[imageB == 0] = imageA[imageB == 0]
            imageA[imageA == -1] = 0
            imageB[imageB == -1] = 0
            fuseRegion = imageFusion.fuseByMinimum([imageA, imageB])
        elif self.fuse_method == "fadeInAndFadeOut":
            fuseRegion = imageFusion.fuseByFadeInAndFadeOut(images, dx, dy)
        elif self.fuse_method == "trigonometric":
            fuseRegion = imageFusion.fuseByTrigonometric(images, dx, dy)
        elif self.fuse_method == "multiBandBlending":
            imageA[imageA == 0] = imageB[imageA == 0]
            imageB[imageB == 0] = imageA[imageB == 0]
            imageA[imageA == -1] = 0
            imageB[imageB == -1] = 0
            # imageA = imageA.astye(np.uint8);  imageB = imageB.astye(np.uint8);
            fuseRegion = imageFusion.fuseByMultiBandBlending([imageA, imageB])
        elif self.fuse_method == "optimalSeamLine":
            fuseRegion = imageFusion.fuseByOptimalSeamLine(images, self.direction)
        return fuseRegion