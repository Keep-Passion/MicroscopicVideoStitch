import numpy as np
import cv2
import os
from image_utility import Method
from image_fusion import ImageFusion
import time
import myGpuFeatures
import skimage.measure


class VideoStitch(Method):
    """
    The class of video stitcher
    """
    image_shape = None
    offset_list = []
    is_available_list = []
    images_address_list = None

    # 关于图像增强的操作
    is_enhance = False
    is_clahe = False
    clip_limit = 20
    tile_size = 5

    # 关于特征搜索的设置
    roi_ratio = 0.2
    feature_method = "surf"  # "sift","surf" or "orb"
    search_ratio = 0.75  # 0.75 is common value for matches

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

    def start_stitching(self, video_address, sample_rate=1):
        """
        stitching the video
        :param video_address: 视频地址
        :param sample_rate: 视频采样帧率
        :return: 返回拼接后的图像，ndarry
        """
        # *********** 对视频采样，将采样的所有图像输出到与视频文件同目录的temp文件夹 ***********
        # 建立 temp 文件夹
        input_dir = os.path.dirname(video_address)
        sample_dir = os.path.join(input_dir, "temp")

        # 将 video 采样到 temp 文件夹
        self.print_and_log("Video name:" + video_address)
        self.print_and_log("Sampling rate:" + str(sample_rate))
        self.print_and_log("We save sampling images in " + sample_dir)
        self.print_and_log("Sampling images ...")

        # if os.path.exists(sample_dir):
        #     self.delete_folder(sample_dir)
        # self.make_out_dir(sample_dir)
        #
        # # 解压文件时有可能会得到无法解压的错误，需要在工程中注意
        # cap = cv2.VideoCapture(video_address)
        # frame_num = 0
        # save_num = 0
        # start_time = time.time()
        # while True:
        #     ret, origin_frame = cap.read()
        #     if ret is False:
        #         break
        #     frame_num = frame_num + 1
        #     if frame_num % sample_rate == 0:
        #         gray_frame = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2GRAY)
        #         save_num = save_num + 1
        #         cv2.imwrite(os.path.join(sample_dir, str(save_num).zfill(10) + ".png"), gray_frame)
        # cap.release()
        # end_time = time.time()
        # self.print_and_log("Sampled done, The time of sampling is {:.3f} \'s".format(end_time - start_time))

        # **************************** 配准 ****************************
        # 开始拼接文件夹下的图片
        dirs = sorted(os.listdir(sample_dir), key=lambda i: int(i.split(".")[0]))
        self.images_address_list = [os.path.join(sample_dir, item) for item in dirs]
        self.print_and_log("start matching")

        start_time = time.time()

        self.is_available_list.append(True)
        status = False
        for file_index in range(1, len(self.images_address_list)):
            self.print_and_log("    Analyzing {}th frame and the name is {}".format(file_index, os.path.basename(
                self.images_address_list[file_index])))
            last_image = cv2.imdecode(np.fromfile(self.images_address_list[file_index - 1], dtype=np.uint8),
                                      cv2.IMREAD_GRAYSCALE)
            next_image = cv2.imdecode(np.fromfile(self.images_address_list[file_index], dtype=np.uint8),
                                      cv2.IMREAD_GRAYSCALE)
            self.image_shape = last_image.shape
            status, offset = self.calculate_offset_by_feature_in_roi([last_image, next_image])
            if status is False:
                self.print_and_log("    {}th frame can not be stitched, the reason is {}".format(file_index, offset))
                self.is_available_list.append(False)
                self.offset_list.append([0, 0])
            else:
                self.print_and_log("    {}th frame can be stitched, the offset is {}".format(file_index, offset))
                self.is_available_list.append(True)
                self.offset_list.append(offset)
        end_time = time.time()
        self.print_and_log("The time of registering is {:.3f} \'s".format(end_time - start_time))
        self.print_and_log("is_available_list:{}".format(self.is_available_list))
        self.print_and_log("offset_list:{}".format(self.offset_list))

        # self.offset_list = [[1, 0], [-2, 0], [1, 0], [-1, 0], [1, 0], [0, -1], [3, 1], [0, -1], [0, 1], [0, -1], [0, 1], [0, -1], [-1, 0], [-1, 0], [1, 0], [1, 0], [1, 0], [8, 2], [11, 1], [33, 6], [34, 7], [69, 14], [12, 2], [2, 0], [2, 0], [0, -1], [-1, 0], [0, 1], [-1, 0], [-1, 0], [-1, 0], [-1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [5, 1], [2, 0], [3, 0], [12, 2], [0, 0], [87, 18], [7, 1], [3, 0], [2, 0], [1, 0], [1, 0], [0, -1], [0, -1], [0, 1], [-1, 0], [0, -1], [-1, 0], [0, 1], [1, 0], [0, 1], [0, 1], [0, -1], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [4, 1], [4, 1], [2, 0], [1, 0], [1, 0], [5, -3], [4, -3], [1, 0], [1, 0], [-3, 0], [1, 0], [4, 1], [2, 0], [4, 0], [2, 0], [4, 0], [3, 1], [5, 0], [7, -2], [6, -1], [-7, -1], [1, 0], [2, 0], [0, 1], [-1, -1], [-1, 0], [0, 1], [0, -1], [1, 0], [-1, 0], [0, -1], [1, 0], [0, 1], [1, 0], [1, 0], [10, 2], [4, 0], [1, 0], [1, 0], [0, -1], [3, 1], [1, 0], [-4, -1], [-1, 0], [1, 0], [1, 0], [2, 0], [4, 1], [6, 1], [27, 0], [0, 0], [258, 39], [23, -13], [0, 1], [0, -1], [1, 0], [-1, 0], [-1, 0], [-1, 0], [-1, 0], [1, 0], [2, 0], [1, 0], [0, -1], [0, 1], [-1, 0], [1, 0], [1, 0], [1, 0], [0, -1], [-1, 0], [1, 0], [-1, 0], [1, 0], [4, 0], [0, -1], [0, -1], [1, 0], [12, -11], [1, 0], [-1, 0], [-1, 0], [-2, 0], [-1, 0], [-3, 0], [-2, 0], [0, 1], [0, -1], [0, 1], [-1, 0], [0, -1], [0, 1], [0, 1], [1, 0], [-1, 0], [-1, 0], [0, -1], [0, 1], [1, 0], [1, 0], [0, 1], [-1, 0], [0, 1], [1, 0], [0, -1], [1, 0], [0, 1], [0, -1], [1, 0], [0, 1], [1, 0], [8, 1], [1, 0], [3, -1], [4, -2], [9, -7], [4, -6], [0, -2], [-7, -3], [7, 1], [2, 0], [-5, -1], [1, 0], [1, 0], [1, 0], [-1, 0], [1, 0], [-1, 0], [-6, -1], [2, 0], [12, -3], [3, -2], [6, -6], [3, -2], [6, -5], [10, -9], [6, -5], [2, -3], [1, 0], [2, 0], [4, -3], [17, -15], [1, -1], [1, 0], [2, 0], [2, -3], [2, 0], [1, 0], [0, -1], [1, 0], [1, 0], [1, 0], [-2, 0], [-7, -1], [1, 0], [1, 0], [1, 0], [0, -1], [1, 0], [4, 1], [9, 1], [6, 1], [9, 1], [17, 4], [20, 4], [11, 2], [7, 1], [4, 1], [0, -1], [1, 0], [0, -1], [0, -1], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [0, 1], [1, 0], [0, -1], [0, 1], [0, -1], [1, 0], [1, 0], [0, -1], [1, 0], [-1, 0], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0], [-1, 0], [1, 0], [1, 0], [0, 1], [1, 0], [1, 0], [0, 1], [0, -1], [-1, 0], [0, 1], [1, 0], [1, 0], [1, 0], [0, 1], [0, -1], [0, -1], [0, 1], [-1, 0], [1, 0], [0, 1], [0, -1], [0, -1], [1, 0], [0, 1], [0, 1], [0, -1], [1, 0], [1, 1], [3, 0], [0, 0], [0, 0], [258, 57], [14, 3], [7, 0], [1, 0], [-6, -1], [-7, -1], [-1, 0], [3, 0], [5, 0], [-3, 0], [-1, 0], [6, 0], [-1, 0], [-7, -1], [-1, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [0, 1], [0, 1], [0, 1], [0, -1], [0, -1], [0, -1], [0, -1], [0, 1], [0, 1], [0, -1], [-1, 0], [0, 1], [0, -1], [0, 1], [0, -1], [0, -1], [0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [-1, 0], [0, 1], [0, -1], [0, -1], [0, 1], [1, 0], [0, -1], [0, 1], [0, -1], [0, 1], [-1, 0], [0, -1], [8, 1], [5, 1], [0, 1], [0, -1], [0, 1], [0, 1], [0, -1], [0, 1], [0, 1], [-1, 0], [0, -1], [0, 1], [0, -1], [1, 0], [1, 0], [0, -1], [-1, 0], [0, -1], [0, -1], [-1, 0], [1, 0], [1, 0], [0, 1], [0, -1], [0, -1], [0, 1]]
        # self.is_available_list = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
        # # *************************** 融合及拼接 ***************************
        self.print_and_log("start fusing")
        start_time = time.time()
        stitch_image = self.get_stitch_by_offset()
        end_time = time.time()
        self.print_and_log("The time of fusing is {:.3f} \'s".format(end_time - start_time))
        # self.delete_folder(sample_dir)
        self.is_available_list = None
        self.offset_list = None
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
            roi_last_image = last_image[0: int(self.image_shape[0] * temp_ratio), :]
            roi_next_image = next_image[0: int(self.image_shape[0] * temp_ratio), :]
            status, temp_offset = self.calculate_offset_by_feature(roi_last_image, roi_next_image)
            if status is False:
                continue
            else:
                break
        return status, temp_offset

    def calculate_offset_by_feature(self, last_image, next_image):
        """
        通过全局特征匹配计算偏移量
        :param last_image: 上一张图像
        :param next_image: 下一张图像
        :return: status, temp_offset, （拼接状态， 偏移量）
        """
        offset = [0, 0]
        status = False

        # get the feature points
        last_kps, last_features = self.calculate_feature(last_image)
        next_kps, next_features = self.calculate_feature(next_image)
        if last_features is not None and next_features is not None:
            matches = self.match_descriptors(last_features, next_features)
            # match all the feature points
            if self.offset_calculate == "mode":
                (status, offset) = self.get_offset_by_mode(last_kps, next_kps, matches)
            elif self.offset_calculate == "ransac":
                (status, offset, adjustH) = self.get_offset_by_ransac(last_kps, next_kps, matches)
            return status, offset
        else:
            return status, "there are one image have no features"

    def calculate_feature(self, input_image):
        """
        计算图像特征点
        :param input_image: 输入图像
        :return: kps, features 返回特征点，及其相应特征描述符
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
                range_x[i][0] = range_x[i - 1][0]
                range_x[i][1] = range_x[i - 1][1]
                range_y[i][0] = range_y[i - 1][0]
                range_y[i][1] = range_y[i - 1][1]
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
        # print(len(self.offset_list))
        # 如上算出各个图像相对于原点偏移量，并最终计算出输出图像大小，并构造矩阵，如下开始赋值
        for i in range(0, len(self.offset_list)):
            # print(i)
            # print(self.offset_list[i])
            if self.is_available_list[i] is False:
                continue
            image = cv2.imdecode(np.fromfile(self.images_address_list[i], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            self.print_and_log("  stitching " + str(self.images_address_list[i]))
            if i == 0:
                stitch_result[self.offset_list[0][0]: self.offset_list[0][0] + image.shape[0],
                              self.offset_list[0][1]: self.offset_list[0][1] + image.shape[1]] = image
            else:
                if self.fuse_method == "not_fuse":
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

    def get_offset_by_mode(self, last_kps, next_kps, matches):
        """
        通过众数的方法求取位移
        :param last_kps: 上一张图像的特征点
        :param next_kps: 下一张图像的特征点
        :param matches: 匹配矩阵
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
        :return: kps，features， （特征点，特征描述符）
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
        # matches = None
        # if self.is_gpu_available is False:        # CPU Mode
        #     # 建立暴力匹配器
        #     if self.feature_method == "surf" or self.feature_method == "sift":
        #         matcher = cv2.DescriptorMatcher_create("BruteForce")
        #         # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2，返回一个列表
        #         raw_matches = matcher.knnMatch(last_features, next_features, 2)
        #         matches = []
        #         for m in raw_matches:
        #             # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
        #             if len(m) == 2 and m[0].distance < m[1].distance * self.search_ratio:
        #                 # 存储两个点在featuresA, featuresB中的索引值
        #                 matches.append((m[0].trainIdx, m[0].queryIdx))
        #     elif self.feature_method == "orb":
        #         matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
        #         raw_matches = matcher.match(last_features, next_features)
        #         matches = []
        #         for m in raw_matches:
        #             matches.append((m.trainIdx, m.queryIdx))
        # else:                                   # GPU Mode
        #     if self.feature_method == "surf":
        #         matches = self.np_to_list_for_matches(myGpuFeatures.matchDescriptors(np.array(last_features),
        #                                                                              np.array(next_features),
        #                                                                              2, self.search_ratio))
        #     elif self.feature_method == "orb":
        #         matches = self.np_to_list_for_matches(myGpuFeatures.matchDescriptors(np.array(last_features),
        #                                                                              np.array(next_features),
        #                                                                              3, self.orb_max_distance))
        return matches

    def justify_result_shape(self, pre_image, gt_image):
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
