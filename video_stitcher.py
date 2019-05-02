import numpy as np
import cv2
import os
from utility import Method
from image_fusion import ImageFusion
import time
import copy

class VideoStitch(Method):
    """
    The class of video stitcher
    """
    image_shape = None
    offset_list = []
    # record_offset_list = []
    is_available_list = []
    images_address_list = None

    # 关于融合方法的设置
    fuse_method = "not_fuse"

    def start_stitching(self, video_address, sample_rate=1, use_pre_calculate=False,
                        pre_calculate_available = None, pre_calculate_offset=None, pre_register_time=None):
        """
        stitching the video
        :param video_address: 视频地址
        :param sample_rate: 视频采样帧率
        :param use_pre_calculate: 是否使用预先计算的偏移量
        :param pre_calculate_available: 预先计算的可用list
        :param pre_calculate_offset: 预先计算的的偏移量
        :param pre_register_times: 预先计算的配准时间
        :return: 返回拼接后的图像，ndarry
        """
        # *********** 对视频采样，将采样的所有图像输出到与视频文件同目录的temp文件夹 ***********
        # 建立 temp 文件夹
        input_dir = os.path.dirname(video_address)
        video_name = os.path.basename(video_address).split(".")[0]
        sample_dir = os.path.join(input_dir, "temp_" + video_name)

        # 将 video 采样到 temp 文件夹
        self.print_and_log("Video name:" + video_address)
        self.print_and_log("Sampling rate:" + str(sample_rate))
        self.print_and_log("We save sampling images in " + sample_dir)
        self.print_and_log("Sampling images ...")

        if os.path.exists(sample_dir):
            self.delete_folder(sample_dir)
        self.make_out_dir(sample_dir)

        # 解压文件时有可能会得到无法解压的错误，需要在工程中注意
        cap = cv2.VideoCapture(video_address)
        frame_num = 0
        save_num = 0
        start_time = time.time()
        while True:
            ret, origin_frame = cap.read()
            if ret is False:
                break
            frame_num = frame_num + 1
            if frame_num % sample_rate == 0:
                # 由于视频采集最后一行默认为0，所以需要去掉
                gray_frame = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2GRAY)[:-2, :]
                save_num = save_num + 1
                cv2.imwrite(os.path.join(sample_dir, str(save_num).zfill(10) + ".png"), gray_frame)
        cap.release()
        end_time = time.time()
        self.print_and_log("Sampled done, The time of sampling is {:.3f} \'s".format(end_time - start_time))

        # # **************************** 配准 ****************************
        # # 开始拼接文件夹下的图片
        dirs = sorted(os.listdir(sample_dir), key=lambda i: int(i.split(".")[0]))
        self.images_address_list = [os.path.join(sample_dir, item) for item in dirs]
        temp_image = cv2.imdecode(np.fromfile(self.images_address_list[0], dtype=np.uint8),
                                  cv2.IMREAD_GRAYSCALE)
        self.image_shape = temp_image.shape
        self.print_and_log("start matching")
        del temp_image
        start_time = time.time()
        if use_pre_calculate:
            self.is_available_list = pre_calculate_available
            self.offset_list = pre_calculate_offset
        else:
            self.is_available_list.append(True)
            for file_index in range(1, len(self.images_address_list)):
                self.print_and_log("    Analyzing {}th frame and the name is {}".format(file_index, os.path.basename(
                    self.images_address_list[file_index])))
                # 获得上一次可用的编号,不是所有帧都是有用的，所以在寻找时需要过滤
                # 先找到 is_available_list 中最后一个True的索引，在减去目前索引到其距离
                temp_available_list = self.is_available_list.copy()
                temp_available_list.reverse()
                last_file_index = file_index - temp_available_list.index(True) - 1
                # self.print_and_log("  The last file index is {}, the next file index is {}"
                #                    .format(last_file_index, file_index))
                last_image = cv2.imdecode(np.fromfile(self.images_address_list[last_file_index], dtype=np.uint8),
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
                del last_image, next_image
        record_offset_list = copy.deepcopy(self.offset_list)

        end_time = time.time()
        if use_pre_calculate:
            self.print_and_log("The time of registering is {:.3f} \'s".format(end_time - start_time + pre_register_time))
        else:
            self.print_and_log("The time of registering is {:.3f} \'s".format(end_time - start_time))
        self.print_and_log("is_available_list:{}".format(self.is_available_list))
        self.print_and_log("offset_list:{}".format(self.offset_list))
        self.print_and_log("len is_available_list:{}".format(len(self.is_available_list)))
        self.print_and_log("len offset_list:{}".format(len(self.offset_list)))
        # self.offset_list = [[1, 0], [-2, 0], [1, 0], [-1, 0], [1, 0], [0, -1], [3, 1], [0, -1], [0, 1], [0, -1], [0, 1], [0, -1], [-1, 0], [-1, 0], [1, 0], [1, 0], [1, 0], [8, 2], [11, 1], [33, 6], [34, 7], [69, 14], [12, 2], [2, 0], [2, 0], [0, -1], [-1, 0], [0, 1], [-1, 0], [-1, 0], [-1, 0], [-1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [5, 1], [2, 0], [3, 0], [12, 2], [0, 0], [87, 18], [7, 1], [3, 0], [2, 0], [1, 0], [1, 0], [0, -1], [0, -1], [0, 1], [-1, 0], [0, -1], [-1, 0], [0, 1], [1, 0], [0, 1], [0, 1], [0, -1], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [4, 1], [4, 1], [2, 0], [1, 0], [1, 0], [5, -3], [4, -3], [1, 0], [1, 0], [-3, 0], [1, 0], [4, 1], [2, 0], [4, 0], [2, 0], [4, 0], [3, 1], [5, 0], [7, -2], [6, -1], [-7, -1], [1, 0], [2, 0], [0, 1], [-1, -1], [-1, 0], [0, 1], [0, -1], [1, 0], [-1, 0], [0, -1], [1, 0], [0, 1], [1, 0], [1, 0], [10, 2], [4, 0], [1, 0], [1, 0], [0, -1], [3, 1], [1, 0], [-4, -1], [-1, 0], [1, 0], [1, 0], [2, 0], [4, 1], [6, 1], [27, 0], [0, 0], [258, 39], [23, -13], [0, 1], [0, -1], [1, 0], [-1, 0], [-1, 0], [-1, 0], [-1, 0], [1, 0], [2, 0], [1, 0], [0, -1], [0, 1], [-1, 0], [1, 0], [1, 0], [1, 0], [0, -1], [-1, 0], [1, 0], [-1, 0], [1, 0], [4, 0], [0, -1], [0, -1], [1, 0], [12, -11], [1, 0], [-1, 0], [-1, 0], [-2, 0], [-1, 0], [-3, 0], [-2, 0], [0, 1], [0, -1], [0, 1], [-1, 0], [0, -1], [0, 1], [0, 1], [1, 0], [-1, 0], [-1, 0], [0, -1], [0, 1], [1, 0], [1, 0], [0, 1], [-1, 0], [0, 1], [1, 0], [0, -1], [1, 0], [0, 1], [0, -1], [1, 0], [0, 1], [1, 0], [8, 1], [1, 0], [3, -1], [4, -2], [9, -7], [4, -6], [0, -2], [-7, -3], [7, 1], [2, 0], [-5, -1], [1, 0], [1, 0], [1, 0], [-1, 0], [1, 0], [-1, 0], [-6, -1], [2, 0], [12, -3], [3, -2], [6, -6], [3, -2], [6, -5], [10, -9], [6, -5], [2, -3], [1, 0], [2, 0], [4, -3], [17, -15], [1, -1], [1, 0], [2, 0], [2, -3], [2, 0], [1, 0], [0, -1], [1, 0], [1, 0], [1, 0], [-2, 0], [-7, -1], [1, 0], [1, 0], [1, 0], [0, -1], [1, 0], [4, 1], [9, 1], [6, 1], [9, 1], [17, 4], [20, 4], [11, 2], [7, 1], [4, 1], [0, -1], [1, 0], [0, -1], [0, -1], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [0, 1], [1, 0], [0, -1], [0, 1], [0, -1], [1, 0], [1, 0], [0, -1], [1, 0], [-1, 0], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0], [-1, 0], [1, 0], [1, 0], [0, 1], [1, 0], [1, 0], [0, 1], [0, -1], [-1, 0], [0, 1], [1, 0], [1, 0], [1, 0], [0, 1], [0, -1], [0, -1], [0, 1], [-1, 0], [1, 0], [0, 1], [0, -1], [0, -1], [1, 0], [0, 1], [0, 1], [0, -1], [1, 0], [1, 1], [3, 0], [0, 0], [0, 0], [258, 57], [14, 3], [7, 0], [1, 0], [-6, -1], [-7, -1], [-1, 0], [3, 0], [5, 0], [-3, 0], [-1, 0], [6, 0], [-1, 0], [-7, -1], [-1, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [0, 1], [0, 1], [0, 1], [0, -1], [0, -1], [0, -1], [0, -1], [0, 1], [0, 1], [0, -1], [-1, 0], [0, 1], [0, -1], [0, 1], [0, -1], [0, -1], [0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [-1, 0], [0, 1], [0, -1], [0, -1], [0, 1], [1, 0], [0, -1], [0, 1], [0, -1], [0, 1], [-1, 0], [0, -1], [8, 1], [5, 1], [0, 1], [0, -1], [0, 1], [0, 1], [0, -1], [0, 1], [0, 1], [-1, 0], [0, -1], [0, 1], [0, -1], [1, 0], [1, 0], [0, -1], [-1, 0], [0, -1], [0, -1], [-1, 0], [1, 0], [1, 0], [0, 1], [0, -1], [0, -1], [0, 1]]
        # self.is_available_list = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
        # # *************************** 融合及拼接 ***************************
        self.print_and_log("start fusing")
        start_time = time.time()
        stitch_image = self.get_stitch_by_offset()
        end_time = time.time()
        self.print_and_log("The time of fusing is {:.3f} \'s".format(end_time - start_time))
        self.delete_folder(sample_dir)
        record_available_list = self.is_available_list
        self.is_available_list = []
        self.offset_list = []
        return stitch_image, record_available_list, record_offset_list

    def calculate_offset_by_feature_in_roi(self, images):
        """
        采用ROI增长的方式进行特征搜索
        :param images: [last_image, next_image]
        :return: status, temp_offset, （拼接状态， 偏移量）
        """
        status = False
        temp_offset = [0, 0]
        last_image, next_image = images
        row, col = last_image.shape[:2]
        max_iteration = int(1 / self.roi_ratio)
        for i in range(0, max_iteration):
            temp_ratio = self.roi_ratio * (i + 1)
            search_len = np.floor(row * temp_ratio).astype(int)
            roi_last_image = last_image[0: search_len, :]
            roi_next_image = next_image[0: search_len, :]
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
                        self.offset_list[i][1]: self.offset_list[i][1] + image.shape[1]] = image.copy()
                    next_roi_fuse_region = stitch_result[roi_ltx:roi_rbx, roi_lty:roi_rby].copy()
                    stitch_result[roi_ltx:roi_rbx, roi_lty:roi_rby] = self.fuse_image(
                        [last_roi_fuse_region, next_roi_fuse_region],
                        [offset_list_origin[i][0], offset_list_origin[i][1]])
                    del last_roi_fuse_region, next_roi_fuse_region
            del image
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
        elif self.fuse_method == "spatial_frequency":
            fuse_region = image_fusion.fuse_by_spatial_frequency([last_rfr, next_rfr])
        elif self.fuse_method == "multi_band_blending":
            fuse_region = image_fusion.fuse_by_multi_band_blending([last_rfr, next_rfr])
        return fuse_region

    @staticmethod
    def record_video_stitch_parameters(output_dir, video_name, available_list, offset_list, register_time):
        '''
        Write parameters in video stitch
        :param output_dir: the address of record file
        :param video_name: str
        :param available_list: like [True, False]
        :param offset_list: like [[20, 30], [-1, -2]]
        :param register_time: float, time of finding feature and matching
        :return:
        '''
        file_address = os.path.join(output_dir, "video_stitch_record.txt")
        f = open(file_address, "a")
        f.write("###-" + video_name)
        f.write("\n")
        temp = [str(1) if item else str(0) for item in available_list]
        f.write(",".join(temp))
        f.write("\n")
        temp = ""
        for item in offset_list:
            temp = temp + str(item[0]) + " " + str(item[1]) + ","
        f.write(temp[:-1])
        f.write("\n")
        f.write(str(register_time))
        f.write("\n")
        f.close()

    def read_video_stitch_parameters(self, input_address):
        '''
        Read the record with video stitch parameters
        :param input_address: record file address
        :return: available_list, offset_list, register_time_list
        '''
        available_list = []
        offset_list = []
        register_time_list =[]
        if os.path.exists(input_address) is False:
            self.print_and_log("There is no record file with video parameters to read")
            return available_list, offset_list, register_time_list
        count = 0   # using count to judge which value to read, 0 is available_list, 1 is offset_list and 2 is time
        with open(input_address, 'r') as f:
            while True:
                lines = f.readline()
                if not lines:
                    break
                if lines.startswith("###-"):
                    count = 0
                    continue
                lines = lines.strip('\n')
                if count == 0:
                    temp_available = []
                    for item in lines.split(","):
                        if item == "0":
                            temp_available.append(False)
                        elif item == "1":
                            temp_available.append(True)
                    available_list.append(temp_available)
                    count += 1
                elif count == 1:
                    temp_offset = []
                    for item in lines.split(","):
                        temp = item.split(" ")
                        temp_offset.append([int(temp[0]), int(temp[1])])
                    offset_list.append(temp_offset)
                    count += 1
                elif count == 2:
                    register_time_list.append(float(lines))
        return available_list, offset_list, register_time_list

if __name__=="__main__":
    video_stitcher = VideoStitch()
    # output_dir = ".\\datasets\\"
    # file_name = "patch_1_1"
    # offset_list = [[-1, 20], [9, 10], [13, 10]]
    # available_list = [True, False, True]
    # time = 30.56
    # video_stitcher. record_video_stitch_parameters(output_dir, file_name, available_list, offset_list, time)
    # offset_list = [[-1, 11], [9, 198], [13, 245], [257, 66]]
    # available_list = [True, False, True, False]
    # time = 30.87
    # video_stitcher. record_video_stitch_parameters(output_dir, file_name, available_list, offset_list, time)
    # offset_list = [[-1, -100], [0, 0], [13, 10]]
    # available_list = [True, False, True, True]
    # time = 30.15
    # video_stitcher. record_video_stitch_parameters(output_dir, file_name, available_list, offset_list, time)
    available_list, offset_list, register_time_list = video_stitcher.read_video_stitch_parameters(".\\datasets\\video_stitch_record.txt")
    print(available_list)
    print(offset_list)
    print(register_time_list)