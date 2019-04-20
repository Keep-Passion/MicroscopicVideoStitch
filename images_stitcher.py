import numpy as np
import cv2
import os
from image_utility import Method
from image_fusion import ImageFusion
import time
import myGpuFeatures
import skimage.measure


class ImagesStitch(Method):
    """
    The class of image stitcher
    """

    # 关于录入文件的设置
    image_shape = None
    offset = [0, 0]
    images_address_list = None

    # 关于图像增强的操作
    is_enhance = False
    is_clahe = False
    clip_limit = 20
    tile_size = 5

    # 关于特征搜索的设置
    roi_ratio = 0.2
    feature_method = "surf"  # "sift","surf" or "orb"
    search_ratio = 0.75       # 0.75 is common value for matches

    # 关于特征配准的设置
    offset_calculate = "mode"  # "mode" or "ransac"
    offset_evaluate = 5

    # 关于 GPU-SURF 的设置
    surf_hessian_threshold = 100.0
    surf_n_octaves = 4
    surf_n_octave_layers = 3
    surf_is_extended = True
    surf_key_points_ratio = 0.01
    surf_is_upright = False

    # 关于 GPU-ORB 的设置
    orb_n_features = 5000
    orb_scale_factor = 1.2
    orb_n_levels = 8
    orb_edge_threshold = 31
    orb_first_level = 0
    orb_wta_k = 2
    orb_patch_size = 31
    orb_fast_threshold = 20
    orb_blur_for_descriptor = False
    orb_max_distance = 30

    # 关于融合方法的设置
    fuse_method = "not_fuse"

    def start_stitching(self, input_address):
        """
        功能：序列拼接，拼接第一张和第二张图像
        :param input_address: 图像所属文件夹地址
        :return: status, stitch_image, （拼接状态， 拼接结果）
        """
        self.print_and_log("Stitching the directory: " + input_address)
        dirs = sorted(os.listdir(input_address))
        self.images_address_list = [os.path.join(input_address, item) for item in dirs]

        self.print_and_log("  stitching " + os.path.basename(self.images_address_list[0]) + " and " + os.path.basename(self.images_address_list[1]))

        self.print_and_log("  start matching")
        start_time = time.time()

        last_image = cv2.imdecode(np.fromfile(self.images_address_list[0], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        next_image = cv2.imdecode(np.fromfile(self.images_address_list[1], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        self.image_shape = last_image.shape
        status, temp_offset = self.calculate_offset_by_feature_in_roi([last_image, next_image])

        if status is False:
            self.print_and_log("  the two images can not be stitched, the reason is: {}".format(temp_offset))
        else:
            self.offset = temp_offset

        end_time = time.time()
        self.print_and_log("  the time of registering is " + str(end_time - start_time) + "s")

        self.print_and_log("  start stitching")
        start_time = time.time()

        stitch_image = self.get_stitch_by_offset()

        end_time = time.time()
        self.print_and_log("  the time of fusing is " + str(end_time - start_time) + "s")
        return status, stitch_image

    def calculate_offset_by_feature_in_roi(self, images):
        """
        采用ROI增长的方式进行特征搜索
        :param images: [last_image, next_image]
        :return: status, temp_offset, （拼接状态， 偏移量）
        """
        status = False
        temp_offset = [0, 0]
        last_image, next_image = images
        max_iteration = int(1 / self.roi_ratio)
        for i in range(0, max_iteration):
            temp_ratio = self.roi_ratio * (i + 1)
            roi_last_image = self.get_roi_region(last_image, temp_ratio, order=0)
            roi_next_image = self.get_roi_region(next_image, temp_ratio, order=1)
            status, temp_offset = self.calculate_offset_by_feature(roi_last_image, roi_next_image)
            if status is False:
                continue
            else:
                temp_offset[0] = temp_offset[0] + last_image.shape[0] - int(temp_ratio * last_image.shape[0])
                break
        return status, temp_offset

    def get_roi_region(self, input_image, ratio, order=0):
        """
        裁剪ROI区域
        :param input_image:输入图像
        :param ratio:裁剪比例
        :param order:为0代表第1张，为1代表第2张
        :return:裁剪图像
        """
        if order == 0:
            return input_image[int((-1) * self.image_shape[0] * ratio):, :]
        elif order == 1:
            return input_image[0: int(self.image_shape[0] * ratio), :]

    def calculate_offset_by_feature(self, last_image, next_image):
        """
        通过全局特征匹配计算偏移量
        :param last_image: 上一张图像
        :param next_image: 下一张图像
        :return: status, offset, （拼接状态， 偏移量）
        """
        offset = [0, 0]
        status = False
        last_kps, last_features = self.calculate_feature(last_image)
        next_kps, next_features = self.calculate_feature(next_image)
        if last_features is not None and next_features is not None:
            matches = self.match_descriptors(last_features, next_features)
            # match all the feature points
            if self.offset_calculate == "mode":
                (status, offset) = self.get_offset_by_mode(last_kps, next_kps, matches)
            elif self.offset_calculate == "ransac":
                (status, offset, adjustH) = self.get_offset_by_ransac(last_kps, next_kps, matches)
        else:
            return status, "there are one image have no features"
        return status, offset

    def calculate_feature(self, input_image):
        """
        计算图像特征点
        :param input_image: 输入图像
        :return: kps, features， 返回特征点及其相应特征描述符
        """
        # 判断是否有增强
        if self.is_enhance:
            if self.is_clahe:
                clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.tile_size, self.tile_size))
                input_image = clahe.apply(input_image)
            elif self.is_clahe is False:
                input_image = cv2.equalizeHist(input_image)
        kps, features = self.detect_and_describe(input_image)
        return kps, features

    def get_offset_by_mode(self, last_kps, next_kps, matches, use_round=True):
        """
        通过众数的方法求取位移
        :param last_kps: 上一张图像的特征点
        :param next_kps: 下一张图像的特征点
        :param matches: 匹配矩阵
        :param use_round: 计算坐标偏移量时是否要四舍五入
        :return: 返回拼接结果图像
        """
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
            if use_round:
                dx_list.append(int(round(last_pt[0] - next_pt[0])))
                dy_list.append(int(round(last_pt[1] - next_pt[1])))
            else:
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

    def get_stitch_by_offset(self):
        """
        根据偏移量计算返回拼接结果
        :return: 拼接结果图像
        """
        dx = self.offset[0]
        dy = self.offset[1]
        rectified_offset = []
        rectified_offset.append([0, 0])
        rectified_offset.append(self.offset)
        result_row = self.image_shape[0]    # 拼接最终结果的横轴长度,先赋值第一个图像的横轴
        result_col = self.image_shape[1]    # 拼接最终结果的纵轴长度,先赋值第一个图像的纵轴
        roi_ltx, roi_lty, roi_rbx, roi_rby = 0, 0, 0, 0
        if dx >= 0 and dy >= 0:
            rectified_offset[0][0] = 0
            rectified_offset[0][1] = 0
            rectified_offset[1][0] = dx
            rectified_offset[1][1] = dy
            roi_ltx = dx
            roi_lty = dy
            roi_rbx = self.image_shape[0]
            roi_rby = self.image_shape[1]
        elif dx >= 0 and dy < 0:
            rectified_offset[0][0] = 0
            rectified_offset[0][1] = -dy
            rectified_offset[1][0] = dx
            rectified_offset[1][1] = 0
            roi_ltx = dx
            roi_lty = -dy
            roi_rbx = self.image_shape[0]
            roi_rby = self.image_shape[1]
        elif dx < 0 and dy >= 0:
            rectified_offset[0][0] = -dx
            rectified_offset[0][1] = 0
            rectified_offset[1][0] = 0
            rectified_offset[1][1] = dy
            roi_ltx = -dx
            roi_lty = dy
            roi_rbx = self.image_shape[0]
            roi_rby = self.image_shape[1]
        elif dx < 0 and dy < 0:
            rectified_offset[0][0] = -dx
            rectified_offset[0][1] = -dy
            rectified_offset[1][0] = 0
            rectified_offset[1][1] = 0
            roi_ltx = -dx
            roi_lty = -dy
            roi_rbx = self.image_shape[0]
            roi_rby = self.image_shape[1]
        result_row = result_row + abs(dx)
        result_col = result_col + abs(dy)
        crop_result_ltx = 0
        crop_result_lty = abs(dy)
        crop_result_rbx = self.image_shape[0] + abs(dx)
        crop_result_rby = self.image_shape[1]
        stitch_result = np.zeros((result_row, result_col)) - 1
        last_image = cv2.imdecode(np.fromfile(self.images_address_list[0], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        next_image = cv2.imdecode(np.fromfile(self.images_address_list[1], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        stitch_result[rectified_offset[0][0]: rectified_offset[0][0] + self.image_shape[0],
            rectified_offset[0][1]: rectified_offset[0][1] + self.image_shape[1]] = last_image
        last_roi_fuse_region = stitch_result[roi_ltx: roi_rbx, roi_lty: roi_rby].copy()
        stitch_result[rectified_offset[1][0]: rectified_offset[1][0] + self.image_shape[0],
            rectified_offset[1][1]: rectified_offset[1][1] + self.image_shape[1]] = next_image
        next_roi_fuse_region = stitch_result[roi_ltx:roi_rbx, roi_lty:roi_rby].copy()
        stitch_result[roi_ltx:roi_rbx, roi_lty:roi_rby] = \
            self.fuse_image([last_roi_fuse_region, next_roi_fuse_region], [dx, dy])
        # return stitch_result
        return stitch_result[crop_result_ltx: crop_result_rbx, crop_result_lty: crop_result_rby]

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
        :return: kps，features 返回特征点集，及对应的描述特征
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
        :return: matches， 匹配矩阵
        """
        # matches = None
        # if self.feature_method == "surf" or self.feature_method == "sift":
        #     matcher = cv2.DescriptorMatcher_create("BruteForce")
        #     # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2，返回一个列表
        #     raw_matches = matcher.knnMatch(last_features, next_features, 2)
        #     matches = []
        #     for m in raw_matches:
        #         # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
        #         if len(m) == 2 and m[0].distance < m[1].distance * self.search_ratio:
        #             # 存储两个点在featuresA, featuresB中的索引值
        #             matches.append((m[0].trainIdx, m[0].queryIdx))
        # elif self.feature_method == "orb":
        #     matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
        #     raw_matches = matcher.match(last_features, next_features)
        #     matches = []
        #     for m in raw_matches:
        #         matches.append((m.trainIdx, m.queryIdx))
        matches = None
        if self.is_gpu_available is False:        # CPU Mode
            # 建立暴力匹配器
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
        else:                                   # GPU Mode
            if self.feature_method == "surf":
                matches = self.np_to_list_for_matches(myGpuFeatures.matchDescriptors(np.array(last_features),
                                                                                     np.array(next_features),
                                                                                     2, self.search_ratio))
            elif self.feature_method == "orb":
                matches = self.np_to_list_for_matches(myGpuFeatures.matchDescriptors(np.array(last_features),
                                                                                     np.array(next_features),
                                                                                     3, self.orb_max_distance))
        return matches

    def fuse_image(self, overlap_rfrs, offset):
        """
        融合两个重合区域,其中rfr代表（roi_fuse_region）
        :param overlap_rfrs:重合区域
        :param offset: 原本两图像的位移
        :return:返回融合结果
        """
        (last_rfr, next_rfr) = overlap_rfrs
        (dx, dy) = offset
        if self.fuse_method != "fade_in_fade_out" and self.fuse_method != "trigonometric":
            # 将各自区域中为背景的部分用另一区域填充，目的是消除背景
            # 权值为-1是为了方便渐入检出融合和三角融合计算
            last_rfr[last_rfr == -1] = 0
            next_rfr[next_rfr == -1] = 0
            last_rfr[last_rfr == 0] = next_rfr[last_rfr == 0]
            next_rfr[next_rfr == 0] = last_rfr[next_rfr == 0]
        fuse_region = np.zeros(last_rfr.shape, np.uint8)
        image_fusion = ImageFusion()
        if self.fuse_method == "not_fuse":
            fuse_region = next_rfr
        elif self.fuse_method == "average":
            fuse_region = image_fusion.fuse_by_average([last_rfr, next_rfr])
        elif self.fuse_method == "maximum":
            fuse_region = image_fusion.fuse_by_maximum([last_rfr, next_rfr])
        elif self.fuse_method == "minimum":
            fuse_region = image_fusion.fuse_by_minimum([last_rfr, next_rfr])
        elif self.fuse_method == "fade_in_fade_out":
            fuse_region = image_fusion.fuse_by_fade_in_and_fade_out(overlap_rfrs, dx, dy)
        elif self.fuse_method == "trigonometric":
            fuse_region = image_fusion.fuse_by_trigonometric(overlap_rfrs, dx, dy)
        elif self.fuse_method == "multi_band_blending":
            fuse_region = image_fusion.fuse_by_multi_band_blending([last_rfr, next_rfr])
        return fuse_region

    def register_result_shape(self, pre_image, gt_image):
        """
        根据真实图像校准视频拼接的结果
        由于视频拼接和图像拼接属于两次拍摄，起始位置可能不同，因此需要校准后才能对比
        默认图像拼接结果没有黑色区域，需要识别pre_image中所占据的最小方格
        :param pre_image: 视频拼接结果图像
        :param gt_image: 图像拼接真实图像
        :return: 校准后的视频拼接图像
        """
        # 首先去掉pre_image中黑色的部分
        roi_ltx, roi_lty, roi_rbx, roi_rby = 0, 0, 0, 0
        threshold = 5
        for index in range(pre_image.shape[1]//2, 0, -1):
            if np.count_nonzero(pre_image[:, index] == 0) > threshold:
                roi_lty = index
                break
        for index in range(pre_image.shape[1]//2, pre_image.shape[1]):
            if np.count_nonzero(pre_image[:, index] == 0) > threshold:
                roi_rby = index
                break
        pre_image = pre_image[:, roi_lty: roi_rby]
        # 获取两幅图像的形状
        pre_h, pre_w = pre_image.shape
        gt_h, gt_w = gt_image.shape

        # 裁剪两幅图像的ROI区域
        pre_roi = pre_image[0: int(pre_h * self.roi_ratio), :]
        gt_roi = gt_image[0: int(gt_h * self.roi_ratio), :]

        # 求取两张图像的偏移量
        pre_kps, pre_features = self.detect_and_describe(pre_roi)
        gt_kps, gt_features = self.detect_and_describe(gt_roi)

        status = False
        offset = [0, 0]
        if pre_features is not None and gt_features is not None:
            matches = self.match_descriptors(gt_features, pre_features)
            if self.offset_calculate == "mode":
                (status, offset) = self.get_offset_by_mode(gt_kps, pre_kps, matches)

        self.print_and_log("  Justifying two images, the offset is {}".format(offset))
        if status is False:
            self.print_and_log("  Matching error in justifying, please check, the reason is {}".format(offset))
            return status, 0, 0
        else:  # 对齐
            pre_justify_image = np.zeros(pre_image.shape, dtype=np.uint8)
            dx, dy = offset
            roi_ltx_gt, roi_lty_gt, roi_rbx_gt, roi_rby_gt, roi_ltx_pre, roi_lty_pre, roi_rbx_pre, roi_rby_pre \
                = 0, 0, 0, 0, 0, 0, 0, 0
            if dx >= 0 and dy >= 0:
                roi_ltx_gt = dx
                roi_lty_gt = dy
                roi_rbx_gt = min(pre_h + abs(dx), gt_h)
                roi_rby_gt = min(pre_w + abs(dy), gt_w)
                roi_ltx_pre = 0
                roi_lty_pre = 0
                roi_rbx_pre = min(gt_h - abs(dx), pre_h)
                roi_rby_pre = min(gt_w - abs(dy), pre_w)
            elif dx >= 0 > dy:
                roi_ltx_gt = dx
                roi_lty_gt = 0
                roi_rbx_gt = min(pre_h + abs(dx), gt_h)
                roi_rby_gt = min(pre_w - abs(dy), gt_w)
                roi_ltx_pre = 0
                roi_lty_pre = abs(dy)
                roi_rbx_pre = min(gt_h - abs(dx), pre_h)
                roi_rby_pre = min(gt_w + abs(dy), pre_w)
            elif dx < 0 <= dy:
                roi_ltx_gt = 0
                roi_lty_gt = dy
                roi_rbx_gt = min(pre_h - abs(dx), gt_h)
                roi_rby_gt = min(pre_w + abs(dy), gt_w)
                roi_ltx_pre = abs(dx)
                roi_lty_pre = 0
                roi_rbx_pre = min(gt_h + abs(dx), pre_h)
                roi_rby_pre = min(gt_w - abs(dy), pre_w)
            elif dx < 0 and dy < 0:
                roi_ltx_gt = 0
                roi_lty_gt = 0
                roi_rbx_gt = min(pre_h - abs(dx), gt_h)
                roi_rby_gt = min(pre_w - abs(dy), gt_w)
                roi_ltx_pre = abs(dx)
                roi_lty_pre = abs(dy)
                roi_rbx_pre = min(gt_h + abs(dx), pre_h)
                roi_rby_pre = min(gt_w + abs(dy), pre_w)
            register_pre_image = pre_image[roi_ltx_pre: roi_rbx_pre, roi_lty_pre: roi_rby_pre]
            register_gt_image = gt_image[roi_ltx_gt: roi_rbx_gt, roi_lty_gt: roi_rby_gt]
            return status, register_pre_image, register_gt_image

    def compare_result_gt(self, stitch_image, gt_image):
        """
        对比拼接图像和真实图像，MSE，pnsr,ssim
        :param stitch_image:拼接图像
        :param gt_image:结果图像
        :return:
        """
        assert stitch_image.shape == gt_image.shape, "The shape of two image is not same"
        mse_score = skimage.measure.compare_mse(stitch_image, gt_image)
        psnr_score = skimage.measure.compare_psnr(stitch_image, gt_image)
        ssim_score = skimage.measure.compare_ssim(stitch_image, gt_image)
        return mse_score, psnr_score, ssim_score

    def register_multi_focus_images(self, input_images_list, output_address, select_index):
        """
        对齐所有的多聚焦图像，为网路训练做准备
        :param input_images_list: 输入图像list
        :param output_address: 输出地址
        :param select_index: 参考图像索引
        :return:
        """
        register_images = []
        images_name = []
        # 先选取参考图像和其特征
        reference_image = cv2.imdecode(np.fromfile(input_images_list[select_index], dtype=np.uint8),
                                       cv2.IMREAD_GRAYSCALE)
        reference_image = reference_image[0: 800, :]
        image_name = os.path.basename(input_images_list[select_index])
        images_name.append(image_name)
        register_images.append(reference_image)
        reference_kps, reference_features = self.calculate_feature(reference_image)
        for i in range(len(input_images_list)):
            if i == select_index:
                continue
            image_name = os.path.basename(input_images_list[i])
            self.print_and_log("Analyzing {}".format(image_name))
            target_image = cv2.imdecode(np.fromfile(input_images_list[i], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
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
            register_image = np.zeros(target_image.shape)
            h, w = target_image.shape
            dx, dy = offset[0], offset[1]
            if dx >= 0 and dy >= 0:
                register_image[dx: h, dy: w] = target_image[0: h - dx, 0: w - dy]
            elif dx < 0 and dy >= 0:
                register_image[0: h + dx, dy: w] = target_image[-dx: h, 0: w - dy]
            elif dx >= 0 and dy < 0:
                register_image[dx: h, 0: w + dy] = target_image[0: h - dx, -dy: w]
            elif dx < 0 and dy < 0:
                register_image[0: h + dx, 0: w + dy] = target_image[-dx: h, -dy: w]
            register_images.append(register_image)
            images_name.append(image_name)
            self.print_and_log("Analyzing done")

        for index in range(0, len(images_name)):
            temp_image = register_images[index][20:-20, 20:-20]
            # print(os.path.join(output_address, images_name[index]))
            cv2.imwrite(os.path.join(output_address, images_name[index]), temp_image)
