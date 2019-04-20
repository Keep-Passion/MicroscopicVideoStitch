import cv2
from video_stitcher import VideoStitch
import os
import time
import glob
from image_utility import Method
import numpy as np
import myGpuFeatures

class RectifyImages(Method):

    def __init__(self, input_images_list, output_address, select_index=0, feature_method="surf", search_ratio=0.75):
        self.input_images_list = input_images_list
        self.output_address = output_address
        self.select_index = select_index
        self.feature_method = feature_method
        self.search_ratio = search_ratio

        # 关于特征配准的设置
        self.offset_calculate = "mode"  # "mode" or "ransac"
        self.offset_evaluate = 5

        # 关于 GPU-SURF 的设置
        self.surf_hessian_threshold = 100.0
        self.surf_n_octaves = 4
        self.surf_n_octave_layers = 3
        self.surf_is_extended = True
        self.surf_key_points_ratio = 0.01
        self.surf_is_upright = False

        # 关于 GPU-ORB 的设置
        self.orb_n_features = 5000
        self.orb_scale_factor = 1.2
        self.orb_n_levels = 8
        self.orb_edge_threshold = 31
        self.orb_first_level = 0
        self.orb_wta_k = 2
        self.orb_patch_size = 31
        self.orb_fast_threshold = 20
        self.orb_blur_for_descriptor = False
        self.orb_max_distance = 30

    def start_rectifying(self):
        rectify_images = []
        images_name = []
        reference_image = cv2.imdecode(np.fromfile(self.input_images_list[self.select_index], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        reference_image = reference_image[0: 800, :]
        image_name = os.path.basename(self.input_images_list[self.select_index])
        images_name.append(image_name)
        rectify_images.append(reference_image)
        reference_kps, reference_features = self.calculate_feature(reference_image)
        for i in range(len(self.input_images_list)):
            if i == self.select_index:
                continue
            image_name = os.path.basename(self.input_images_list[i])
            self.print_and_log("Analysing {}".format(image_name))
            target_image = cv2.imdecode(np.fromfile(self.input_images_list[i], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            target_image = target_image[0: 800, :]
            target_kps, target_features = self.calculate_feature(target_image)
            offset = [0, 0]
            status = False
            if target_features is not None:
                matches = self.match_descriptors(reference_features, target_features)
                # match all the feature points
                if self.offset_calculate == "mode":
                    (status, offset) = self.get_offset_by_mode(reference_kps, target_kps, matches)
                elif self.offset_calculate == "ransac":
                    (status, offset, adjustH) = self.get_offset_by_ransac(reference_kps, target_kps, matches)
            else:
                self.print_and_log("   {} can't be justified because no features".format(image_name))
                continue
            if status is False:
                self.print_and_log("   {} can't be justified because have less common features".format(image_name))
                continue
            rectify_image = self.rectify_image_by_offset(target_image, offset)
            rectify_images.append(rectify_image)
            images_name.append(image_name)
            self.print_and_log("Analysing done")

        for index in range(0 , len(images_name)):
            temp_image = rectify_images[index][20:-20, 20:-20]
            cv2.imwrite(self.output_address + images_name[index], temp_image)


    def rectify_image_by_offset(self, image, offset):
        rectified = np.zeros(image.shape)
        h, w = image.shape
        if offset[0] >= 0 and offset[1] >= 0:
            rectified[offset[0]: h, offset[1]: w] = image[0: h - offset[0], 0: w - offset[1]]
        elif offset[0] < 0 and offset[1] >= 0:
            rectified[0: h + offset[0], offset[1]: w] = image[-offset[0]: h, 0: w - offset[1]]
        elif offset[0] >= 0 and offset[1] < 0:
            rectified[offset[0]: h, 0: w + offset[1]] = image[0: h - offset[0], -offset[1]: w]
        elif offset[0] < 0 and offset[1] < 0:
            rectified[0: h + offset[0], 0: w + offset[1]] = image[-offset[0]: h, -offset[1]: w]
        return rectified

    def calculate_feature(self, input_image):
        """
        计算图像特征点
        :param input_image: 输入图像
        :return: 返回特征点(kps)，及其相应特征描述符
        """
        kps, features = self.detect_and_describe(input_image)
        return kps, features

    def get_offset_by_mode(self, last_kps, next_kps, matches):
        """
        通过众数的方法求取位移
        :param last_kps: 上一张图像的特征点
        :param next_kps: 下一张图像的特征点
        :param matches: 匹配矩阵
        :return: 返回拼接结果图像
        """
        print("matches:", len(matches))
        total_status = True
        if len(matches) == 0:
            total_status = False
            return total_status, "the two images have no matches"
        dx_list = []
        dy_list = []
        for trainIdx, queryIdx in matches:
            last_pt = (last_kps[queryIdx][1], last_kps[queryIdx][0])
            next_pt = (next_kps[trainIdx][1], next_kps[trainIdx][0])
            if int(last_pt[0] - next_pt[0]) == 0 and int(last_pt[1] - next_pt[1]) == 0:
                continue
            # dx_list.append(int(round(last_pt[0] - next_pt[0])))
            # dy_list.append(int(round(last_pt[1] - next_pt[1])))
            dx_list.append(int(last_pt[0] - next_pt[0]))
            dy_list.append(int(last_pt[1] - next_pt[1]))
        if len(dx_list) == 0:
            dx_list.append(0)
            dy_list.append(0)
        # Get Mode offset in [dxList, dyList], thanks for clovermini
        zipped = zip(dx_list, dy_list)
        zip_list = list(zipped)
        zip_dict = dict((a, zip_list.count(a)) for a in zip_list)
        zip_dict_sorted = dict(sorted(zip_dict.items(), key=lambda x: x[1], reverse=True))
        dx = list(zip_dict_sorted)[0][0]
        dy = list(zip_dict_sorted)[0][1]
        num = zip_dict_sorted[list(zip_dict_sorted)[0]]
        print("dx:", dx, " dy:", dy)
        print("num:", num)
        if num < self.offset_evaluate:
            total_status = False
            return total_status, "the two images have less common offset"
        else:
            return total_status, [dx, dy]

    def get_offset_by_ransac(self, last_kps, next_kps, matches):
        """
        通过ransac方法求取位移
        :param last_kps: 上一张图像的特征点
        :param next_kps: 下一张图像的特征点
        :param matches: 匹配矩阵
        :return: 返回拼接结果图像
        """
        total_status = False
        last_pts = np.float32([last_kps[i] for (_, i) in matches])
        next_pts = np.float32([next_kps[i] for (i, _) in matches])
        if len(matches) == 0:
            return total_status, [0, 0], 0
        (H, status) = cv2.findHomography(last_pts, next_pts, cv2.RANSAC, 3, 0.9)
        true_count = 0
        for i in range(0, len(status)):
            if status[i]:
                true_count = true_count + 1
        if true_count >= self.offset_evaluate:
            total_status = True
            adjust_h = H.copy()
            adjust_h[0, 2] = 0
            adjust_h[1, 2] = 0
            adjust_h[2, 0] = 0
            adjust_h[2, 1] = 0
            return total_status, [np.round(np.array(H).astype(np.int)[1, 2]) * (-1),
                                  np.round(np.array(H).astype(np.int)[0, 2]) * (-1)], adjust_h
        else:
            return total_status, [0, 0], 0

    @staticmethod
    def np_to_list_for_keypoints(array):
        """
        GPU返回numpy形式的特征点，转成list形式
        :param array:
        :return:
        """
        kps = []
        row, col = array.shape
        for i in range(row):
            kps.append([array[i, 0], array[i, 1]])
        return kps

    @staticmethod
    def np_to_list_for_matches(array):
        """
        GPU返回numpy形式的匹配对，转成list形式
        :param array:
        :return:
        """
        descriptors = []
        row, col = array.shape
        for i in range(row):
            descriptors.append((array[i, 0], array[i, 1]))
        return descriptors

    @staticmethod
    def np_to_kps_and_descriptors(array):
        """
        GPU返回numpy形式的kps，descripotrs，转成list形式
        :param array:
        :return:
        """
        kps = []
        descriptors = array[:, :, 1]
        for i in range(array.shape[0]):
            kps.append([array[i, 0, 0], array[i, 1, 0]])
        return kps, descriptors

    def detect_and_describe(self, image):
        """
        给定一张图像，求取特征点和特征描述符
        :param image: 输入图像
        :return: kps，features
        """
        descriptor = None
        kps = None
        features = None
        if self.is_gpu_available is False:  # CPU mode
            if self.feature_method == "sift":
                descriptor = cv2.xfeatures2d.SIFT_create()
            elif self.feature_method == "surf":
                descriptor = cv2.xfeatures2d.SURF_create()
            elif self.feature_method == "orb":
                descriptor = cv2.ORB_create(self.orb_n_features, self.orb_scale_factor, self.orb_n_levels,
                                            self.orb_edge_threshold, self.orb_first_level, self.orb_wta_k, 0,
                                            self.orb_patch_size, self.orb_fast_threshold)
            # 检测SIFT特征点，并计算描述子
            kps, features = descriptor.detectAndCompute(image, None)
            # 将结果转换成NumPy数组
            kps = np.float32([kp.pt for kp in kps])
        else:  # GPU mode
            if self.feature_method == "sift":
                # 目前GPU-SIFT尚未开发，先采用CPU版本的替代
                descriptor = cv2.xfeatures2d.SIFT_create()
                kps, features = descriptor.detectAndCompute(image, None)
                kps = np.float32([kp.pt for kp in kps])
            elif self.feature_method == "surf":
                kps, features = self.np_to_kps_and_descriptors(
                    myGpuFeatures.detectAndDescribeBySurf(image, self.surf_hessian_threshold,
                                                          self.surf_n_octaves, self.surf_n_octave_layers,
                                                          self.surf_is_extended, self.surf_key_points_ratio,
                                                          self.surf_is_upright))
            elif self.feature_method == "orb":
                kps, features = self.np_to_kps_and_descriptors(
                    myGpuFeatures.detectAndDescribeByOrb(image, self.orb_n_features, self.orb_scale_factor,
                                                         self.orb_n_levels, self.orb_edge_threshold,
                                                         self.orb_first_level, self.orb_wta_k, 0,
                                                         self.orb_patch_size, self.orb_fast_threshold,
                                                         self.orb_blur_for_descriptor))
        # 返回特征点集，及对应的描述特征
        return kps, features

    def match_descriptors(self, last_features, next_features):
        """
        根据两张图像的特征描述符，找到相应匹配对
        :param last_features: 上一张图像特征描述符
        :param next_features: 下一张图像特征描述符
        :return: matches
        """
        matches = None
        if self.feature_method == "surf" or self.feature_method == "sift":
            matcher = cv2.DescriptorMatcher_create("BruteForce")
            # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2，返回一个列表
            raw_matches = matcher.knnMatch(last_features, next_features, 2)
            matches = []
            for m in raw_matches:
                # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
                if len(m) == 2 and m[0].distance < m[1].distance * self.search_ratio:
                    # 存储两个点在featuresA, featuresB中的索引值
                    matches.append((m[0].trainIdx, m[0].queryIdx))
        elif self.feature_method == "orb":
            matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
            raw_matches = matcher.match(last_features, next_features)
            matches = []
            for m in raw_matches:
                matches.append((m.trainIdx, m.queryIdx))
        # if self.isGPUAvailable == False:        # CPU Mode
        #     # 建立暴力匹配器
        #     if self.featureMethod == "surf" or self.featureMethod == "sift":
        #         matcher = cv2.DescriptorMatcher_create("BruteForce")
        #         # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2，返回一个列表
        #         rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        #         matches = []hu
        #         for m in rawMatches:
        #         # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
        #             if len(m) == 2 and m[0].distance < m[1].distance * self.searchRatio:
        #                 # 存储两个点在featuresA, featuresB中的索引值
        #                 matches.append((m[0].trainIdx, m[0].queryIdx))
        #     elif self.featureMethod == "orb":
        #         matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
        #         rawMatches = matcher.match(featuresA, featuresB)
        #         matches = []
        #         for m in rawMatches:
        #             matches.append((m.trainIdx, m.queryIdx))
        #     # self.printAndWrite("  The number of matches is " + str(len(matches)))
        # else:                                   # GPU Mode
        #     if self.featureMethod == "surf":
        #         matches = self.npToListForMatches(myGpuFeatures.matchDescriptors(np.array(featuresA),
        # np.array(featuresB), 2, self.searchRatio))
        #     elif self.featureMethod == "orb":
        #         matches = self.npToListForMatches(myGpuFeatures.matchDescriptors(np.array(featuresA),
        # np.array(featuresB), 3, self.orbMaxDistance))
        return matches

if __name__ == "__main__":
    input_address = ".\\datasets\\mufoc\\origin\\mufoc-1-2\\"
    output_address = ".\\datasets\\mufoc\\rectify\\mufoc-1-2\\"
    input_images_list = glob.glob(input_address + "*.tif")
    select_index = 4
    input_address = ".\\datasets\\mufoc\\origin\\mufoc-1-1\\"
    output_address = ".\\datasets\\mufoc\\rectify\\mufoc-1-1\\"
    input_images_list = glob.glob(input_address + "*.jpg")
    select_index = 0
    rectify = RectifyImages(input_images_list, output_address, select_index=select_index)
    rectify.start_rectifying()