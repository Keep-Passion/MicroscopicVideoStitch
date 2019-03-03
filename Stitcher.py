import numpy as np
import cv2
import os
import ImageUtility as Utility
import ImageFusion
import time
import glob
import myGpuFeatures


class ImageFeature:
    """
    用来保存串行全局拼接中的第二张图像的特征点和描述子，为后续加速拼接使用
    """
    kps = None
    features = None


class VideoStitch(Utility.Method):
    """
    视频拼接类
    """
    def __init__(self, video_address, fuse_method="notFuse", sample_rate=1,
                 roi_ratio=1, feature_method="sift", search_ratio=0.75):
        # 关于录入文件的设置
        self.video_address = video_address
        self.input_dir = os.path.dirname(self.video_address)
        self.sample_rate = sample_rate
        self.image_shape = None
        self.offset_list = []
        self.is_available_list = []
        self.images_address_list = None

        # 关于图像增强的操作
        self.is_enhance = False
        self.is_clahe = False
        self.clip_limit = 20
        self.tile_size = 5

        # 关于特征搜索的设置
        self.roi_ratio = roi_ratio  # Incre method
        self.feature_method = feature_method  # "sift","surf" or "orb"
        self.search_ratio = search_ratio  # 0.75 is common value for matches
        self.last_image_feature = ImageFeature()  # 保存上一张图像特征，方便使用

        # 关于特征配准的设置
        self.offset_calculate = "mode"  # "mode" or "ransac"
        self.offset_evaluate = 5  # 40 menas nums of matches for mode, 3.0 menas  of matches for ransac

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

        # 关于融合方法的设置
        self.fuse_method = fuse_method

    def start_stitching(self):
        """
        对视频进行拼接
        :return: 返回拼接后的图像，ndarry
        """
        # *********** 对视频采样，将采样的所有图像输出到与视频文件同目录的temp文件夹 ***********
        # 建立 temp 文件夹
        sample_dir = os.path.join(self.input_dir, "temp")
        self.make_out_dir(sample_dir)

        # 将 video 采样到 temp 文件夹
        self.print_and_log("Video name:" + self.video_address)
        self.print_and_log("Sampling rate:" + str(self.sample_rate))
        self.print_and_log("We save sampling images in " + sample_dir)
        self.print_and_log("Sampling images ...")
        cap = cv2.VideoCapture(self.video_address)
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

        # **************************** 配准 ****************************
        # 开始拼接文件夹下的图片
        self.images_address_list = glob.glob(os.path.join(sample_dir, "*.png"))
        self.print_and_log("start matching")
        start_time = time.time()
        last_image = cv2.imdecode(np.fromfile(self.images_address_list[0], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        last_roi = self.get_roi_region_for_incre(last_image)
        last_kps, last_features = self.calculate_feature(last_roi)
        self.last_image_feature.kps = last_kps
        self.last_image_feature.features = last_features
        self.is_available_list.append(True)
        self.image_shape = last_image.shape
        for file_index in range(1, len(self.images_address_list)):
            self.print_and_log("    Analyzing {}th frame and the name is {}".format(file_index, os.path.basename(
                self.images_address_list[file_index])))
            next_image = cv2.imdecode(np.fromfile(self.images_address_list[file_index], dtype=np.uint8),
                                      cv2.IMREAD_GRAYSCALE)
            status, offset = self.calculate_offset_by_feature_in_roi(next_image)
            if status is False:
                self.print_and_log("    {}th frame can not be stitched, the reason is {}".format(file_index, offset))
                self.is_available_list.append(False)
            else:
                self.print_and_log("    {}th frame can be stitched, the offset is {}".format(file_index, offset))
                self.is_available_list.append(True)
                self.offset_list.append(offset)
        end_time = time.time()
        self.print_and_log("The time of registering is {:.3f} \'s".format(end_time - start_time))

        # *************************** 融合及拼接 ***************************
        self.print_and_log("start fusing")
        start_time = time.time()
        stitch_image = self.get_stitch_by_offset()
        end_time = time.time()
        self.print_and_log("The time of fusing is {:.3f} \'s".format(end_time - start_time))
        # self.delete_folder(sample_dir)
        return stitch_image

    def calculate_feature(self, input_image):
        """
        计算图像特征点
        :param input_image: 输入图像
        :return: 返回特征点(kps)，及其相应特征描述符
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

    def calculate_offset_by_feature(self, next_image):
        """
        通过全局特征匹配计算偏移量
        :param next_image: 下一张图像
        :return: 返回配准结果和偏移量(offset = [dx,dy])
        """
        offset = [0, 0]
        status = False

        # get the feature points
        last_kps = self.last_image_feature.kps
        last_features = self.last_image_feature.features

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
        if status is True:
            self.last_image_feature.kps = next_kps
            self.last_image_feature.features = next_features
        return status, offset

    def calculate_offset_by_feature_in_roi(self, next_image):
        """
        通过局部特征匹配计算偏移量
        :param next_image: 下一张图像
        :return: 返回配准结果和偏移量(offset = [dx,dy])
        """
        offset = [0, 0]
        status = False
        # get the feature points
        last_kps = self.last_image_feature.kps
        last_features = self.last_image_feature.features
        next_roi = self.get_roi_region_for_incre(next_image)
        next_kps, next_features = self.detect_and_describe(next_roi)
        if last_features is not None and next_features is not None:
            matches = self.match_descriptors(last_features, next_features)
            # match all the feature points
            if self.offset_calculate == "mode":
                (status, offset) = self.get_offset_by_mode(last_kps, next_kps, matches)
            elif self.offset_calculate == "ransac":
                (status, offset, adjustH) = self.get_offset_by_ransac(last_kps, next_kps, matches)
        else:
            return status, "there are one image have no features"
        if status:
            self.last_image_feature.kps = next_kps
            self.last_image_feature.features = next_features
        return status, offset

    def get_stitch_by_offset(self):
        """
        根据偏移量计算返回拼接结果
        :return: 拼接结果图像
        """
        # 如果你不细心，不要碰这段代码
        # 已优化到根据指针来控制拼接，CPU下最快了
        dx_sum = dy_sum = 0
        result_row = self.image_shape[0]  # 拼接最终结果的横轴长度,先赋值第一个图像的横轴
        result_col = self.image_shape[1]  # 拼接最终结果的纵轴长度,先赋值第一个图像的纵轴
        self.offset_list.insert(0, [0, 0])  # 增加第一张图像相对于最终结果的原点的偏移量
        range_x = [[0, 0] for _ in range(len(self.offset_list))]  # 主要用于记录X方向最大最小边界
        range_y = [[0, 0] for _ in range(len(self.offset_list))]  # 主要用于记录Y方向最大最小边界
        temp_offset_list = self.offset_list.copy()
        offset_list_origin = self.offset_list.copy()
        range_x[0][1] = self.image_shape[0]
        range_y[0][1] = self.image_shape[1]

        for i in range(1, len(temp_offset_list)):
            if self.is_available_list[i] is False:
                continue
            dx_sum = dx_sum + self.offset_list[i][0]
            dy_sum = dy_sum + self.offset_list[i][1]
            if dx_sum <= 0:
                for j in range(0, i):
                    temp_offset_list[j][0] = temp_offset_list[j][0] + abs(dx_sum)
                    range_x[j][0] = range_x[j][0] + abs(dx_sum)
                    range_x[j][1] = range_x[j][1] + abs(dx_sum)
                result_row = result_row + abs(dx_sum)
                range_x[i][1] = result_row
                dx_sum = range_x[i][0] = temp_offset_list[i][0] = 0
            else:
                temp_offset_list[i][0] = dx_sum
                result_row = max(result_row, dx_sum + self.image_shape[0])
                range_x[i][1] = result_row
            if dy_sum <= 0:
                for j in range(0, i):
                    temp_offset_list[j][1] = temp_offset_list[j][1] + abs(dy_sum)
                    range_y[j][0] = range_y[j][0] + abs(dy_sum)
                    range_y[j][1] = range_y[j][1] + abs(dy_sum)
                result_col = result_col + abs(dy_sum)
                range_y[i][1] = result_col
                dy_sum = range_y[i][0] = temp_offset_list[i][1] = 0
            else:
                temp_offset_list[i][1] = dy_sum
                result_col = max(result_col, dy_sum + self.image_shape[1])
                range_y[i][1] = result_col
        stitch_result = np.zeros((result_row, result_col), np.int) - 1
        self.offset_list = temp_offset_list
        self.print_and_log("  The rectified offsetList is " + str(self.offset_list))
        # 如上算出各个图像相对于原点偏移量，并最终计算出输出图像大小，并构造矩阵，如下开始赋值
        for i in range(0, len(self.offset_list)):
            image = cv2.imdecode(np.fromfile(self.images_address_list[i], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            self.print_and_log("  stitching " + str(self.images_address_list[i]))
            if i == 0:
                stitch_result[self.offset_list[0][0]: self.offset_list[0][0] + image.shape[0],
                              self.offset_list[0][1]: self.offset_list[0][1] + image.shape[1]] = image
            else:
                if self.fuse_method == "notFuse":
                    # 适用于无图像融合，直接覆盖
                    stitch_result[
                        self.offset_list[i][0]: self.offset_list[i][0] + image.shape[0],
                        self.offset_list[i][1]: self.offset_list[i][1] + image.shape[1]] = image
                else:
                    # 适用于图像融合算法，切出 roiA 和 roiB 供图像融合
                    min_occupy_x = range_x[i - 1][0]
                    max_occupy_x = range_x[i - 1][1]
                    min_occupy_y = range_y[i - 1][0]
                    max_occupy_y = range_y[i - 1][1]
                    # self.printAndWrite("Stitch " + str(i + 1) + "th, the offsetList[i][0] is " + str(
                    #     offsetList[i][0]) + " and the offsetList[i][1] is " + str(offsetList[i][1]))
                    # self.printAndWrite("Stitch " + str(i + 1) + "th, the minOccupyX is " + str(
                    #     minOccupyX) + " and the maxOccupyX is " + str(maxOccupyX) + " and the minOccupyY is " + str(
                    #     minOccupyY) + " and the maxOccupyY is " + str(maxOccupyY))
                    roi_ltx = max(self.offset_list[i][0], min_occupy_x)
                    roi_lty = max(self.offset_list[i][1], min_occupy_y)
                    roi_rbx = min(self.offset_list[i][0] + image.shape[0], max_occupy_x)
                    roi_rby = min(self.offset_list[i][1] + image.shape[1], max_occupy_y)
                    # self.printAndWrite("Stitch " + str(i + 1) + "th, the roi_ltx is " + str(
                    #     roi_ltx) + " and the roi_lty is " + str(roi_lty) + " and the roi_rbx is " + str(
                    #     roi_rbx) + " and the roi_rby is " + str(roi_rby))
                    last_roi_fuse_region = stitch_result[roi_ltx:roi_rbx, roi_lty:roi_rby].copy()
                    stitch_result[
                        self.offset_list[i][0]: self.offset_list[i][0] + image.shape[0],
                        self.offset_list[i][1]: self.offset_list[i][1] + image.shape[1]] = image
                    next_roi_fuse_region = stitch_result[roi_ltx:roi_rbx, roi_lty:roi_rby].copy()
                    stitch_result[roi_ltx:roi_rbx, roi_lty:roi_rby] = self.fuse_image(
                        [last_roi_fuse_region, next_roi_fuse_region],
                        [offset_list_origin[i][0], offset_list_origin[i][1]])
        stitch_result[stitch_result == -1] = 0
        return stitch_result.astype(np.uint8)

    def fuse_image(self, overlap_rfrs, offset):
        """
        融合两个重合区域,其中rfr代表（roi_fuse_region）
        :param overlap_rfrs:重合区域
        :param offset: 原本两图像的位移
        :return:返回融合结果
        """
        (last_rfr, next_rfr) = overlap_rfrs
        (dx, dy) = offset
        if self.fuse_method != "fadeInAndFadeOut" and self.fuse_method != "trigonometric":
            # 将各自区域中为背景的部分用另一区域填充，目的是消除背景
            # 权值为-1是为了方便渐入检出融合和三角融合计算
            last_rfr[last_rfr == -1] = 0
            next_rfr[next_rfr == -1] = 0
            last_rfr[last_rfr == 0] = next_rfr[last_rfr == 0]
            next_rfr[next_rfr == 0] = last_rfr[next_rfr == 0]
        fuse_region = np.zeros(last_rfr.shape, np.uint8)
        image_fusion = ImageFusion.ImageFusion()
        if self.fuse_method == "notFuse":
            fuse_region = next_rfr
        elif self.fuse_method == "average":
            fuse_region = image_fusion.fuse_by_average([last_rfr, next_rfr])
        elif self.fuse_method == "maximum":
            fuse_region = image_fusion.fuse_by_maximum([last_rfr, next_rfr])
        elif self.fuse_method == "minimum":
            fuse_region = image_fusion.fuse_by_minimum([last_rfr, next_rfr])
        elif self.fuse_method == "fadeInAndFadeOut":
            fuse_region = image_fusion.fuse_by_fade_in_and_fade_out(overlap_rfrs, dx, dy)
        elif self.fuse_method == "trigonometric":
            fuse_region = image_fusion.fuse_by_trigonometric(overlap_rfrs, dx, dy)
        elif self.fuse_method == "multiBandBlending":
            fuse_region = image_fusion.fuse_by_multi_band_blending([last_rfr, next_rfr])
        return fuse_region

    def get_roi_region_for_incre(self, input_image):
        """
        获得图像的roi区域，由于视频拼接，前后两帧位移不明显，所以两张图片上部分的偏移量等同于整张图偏移量
        :param input_image: 输入图像
        :return: 感兴趣区域
        """
        row, col = input_image.shape[:2]
        search_len = np.floor(row * self.roi_ratio).astype(int)
        roi_region = input_image[0: search_len, :]
        return roi_region

    def get_offset_by_mode(self, last_kps, next_kps, matches):
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
            dx_list.append(int(round(last_pt[0] - next_pt[0])))
            dy_list.append(int(round(last_pt[1] - next_pt[1])))
            # dx_list.append(int(last_pt[0] - next_pt[0]))
            # dy_list.append(int(last_pt[1] - next_pt[1]))
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
        kps = []
        row, col = array.shape
        for i in range(row):
            kps.append([array[i, 0], array[i, 1]])
        return kps

    @staticmethod
    def np_to_list_for_matches(array):
        descriptors = []
        row, col = array.shape
        for i in range(row):
            descriptors.append((array[i, 0], array[i, 1]))
        return descriptors

    @staticmethod
    def np_to_kps_and_descriptors(array):
        kps = []
        descriptors = array[:, :, 1]
        for i in range(array.shape[0]):
            kps.append([array[i, 0, 0], array[i, 1, 0]])
        return kps, descriptors

    def detect_and_describe(self, image):
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
        #         matches = []
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

    def justify_result_shape(self, pre_image, gt_image):
        """
        根据真实图像校准视频拼接的结果
        由于视频拼接和图像拼接属于两次拍摄，起始位置可能不同，因此需要校准后才能对比
        :param pre_image: 视频拼接结果图像
        :param gt_image: 图像拼接真实图像
        :return: 校准后的视频拼接图像
        """
        pre_justify_image = np.zeros(gt_image.shape, dtype=np.uint8)
        # 获取两幅图像的形状
        gt_h, gt_w = gt_image.shape
        pre_h, pre_w = pre_image.shape

        # 裁剪两幅图像的ROI区域
        gt_roi = self.get_roi_region_for_incre(gt_image)
        pre_roi = self.get_roi_region_for_incre(pre_image)

        # 求取两张图像的偏移量
        gt_kps, gt_features = self.detect_and_describe(gt_roi)
        pre_kps, pre_features = self.detect_and_describe(pre_roi)

        status = False
        offset = [0, 0]
        if pre_features is not None and gt_features is not None:
            matches = self.match_descriptors(gt_features, pre_features)
            if self.offset_calculate == "mode":
                (status, offset) = self.get_offset_by_mode(gt_kps, pre_kps, matches)
        print(offset)
        if status is False:
            self.print_and_log("Matching error in justifying, please check, the reason is {}".format(offset))
        else:
            dx, dy = offset
            if dx >= 0 and dy >= 0:
                temp_height = min(gt_h - dx, pre_h)
                temp_width = min(gt_w - dy, pre_w)
                pre_justify_image[dx: dx + temp_height, dy: dy + temp_width] = pre_image[0: temp_height, 0: temp_width]
            elif dx < 0 <= dy:
                temp_height = min(pre_h - abs(dx), gt_h)
                temp_width = min(gt_w - dy, pre_w)
                pre_justify_image[0: temp_height, dy: dy + temp_width] = \
                    pre_image[abs(dx): pre_h, 0: temp_width]
            elif dx >= 0 > dy:
                temp_height = min(gt_h - abs(dx), pre_h)
                temp_width = min(pre_w - abs(dy), gt_w)
                pre_justify_image[dx: dx + temp_height, 0: temp_width] = \
                    pre_image[0: temp_height, abs(dy): abs(dy) + temp_width]
            elif dx < 0 and dy < 0:
                temp_height = min(pre_h - abs(dx), gt_h)
                temp_width = min(pre_w - abs(dy), gt_w)
                pre_justify_image[0: temp_height, 0: temp_width] = \
                    pre_image[abs(dx): abs(dx) + temp_height, abs(dy): abs(dy) + temp_width]
            elif dx == 0 and dy == 0:
                pre_justify_image = gt_image
        return pre_justify_image


if __name__ == "__main__":
    predict_image = cv2.imread("stitching_by_video.jpg", 0)
    groundT_image = cv2.imread("stitching_by_human.jpg", 0)
    # # pre_image = cv2.imread("pre_test.jpeg", 0)
    # # gt_image = cv2.imread("gt_test.jpeg", 0)
    stitcher = VideoStitch(".\\videos\\test_video.avi")
    # justified_pre_image = stitcher.justify_result_shape(predict_image, groundT_image)
    # cv2.imwrite("justified_result.jpg", justified_pre_image)
    roi_pre_justify = predict_image[0:400, 0:400]
    roi_gt = groundT_image[0:400, 0:400]
    cv2.imwrite("roi_pre.jpg", roi_pre_justify)
    cv2.imwrite("roi_gt.jpg", roi_gt)
