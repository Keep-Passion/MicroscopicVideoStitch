import cv2
import os
import shutil
import numpy as np
from myGpuFeatures import myGpuFeatures


class Method:
    # 关于 GPU 加速的设置
    is_gpu_available = False

    # 关于打印信息的设置
    input_dir = ""
    is_out_log_file = False
    log_file = "evaluate.txt"
    is_print_screen = True

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

    def print_and_log(self, content):
        """
        向屏幕或者txt打印信息
        :param content:
        :return:
        """
        if self.is_print_screen:
            print(content)
        if self.is_out_log_file:
            f = open(os.path.join(self.input_dir, self.log_file), "a")
            f.write(content)
            f.write("\n")
            f.close()

    @staticmethod
    def make_out_dir(dir_path):
        """
        创造一个文件夹
        :param dir_path:文件夹目录
        :return:
        """
        try:
            os.makedirs(dir_path)
        except OSError:
            pass

    @staticmethod
    def delete_folder(dir_address):
        """
        删除一个文件夹下所有文件以及该文件夹
        :param dir_address: 文件夹目录
        :return:
        """
        shutil.rmtree(dir_address)

    @staticmethod
    def resize_image(origin_image, resize_times, inter_method=cv2.INTER_AREA):
        """
        缩放图像
        :param origin_image:原始图像
        :param resize_times: 缩放比率
        :param inter_method: 插值方法
        :return: 缩放结果
        """
        (h, w) = origin_image.shape
        resize_h = int(h * resize_times)
        resize_w = int(w * resize_times)
        # cv2.INTER_AREA是测试后最好的方法
        resized_image = cv2.resize(origin_image, (resize_w, resize_h), interpolation=inter_method)
        return resized_image

    def generate_video_from_image(self, source_image, output_dir):
        """
        Convert source_image to video, simply crop sub-image in source_image in row direction with one pixel increment
        :param source_image: source_image
        :param output_dir: video output dir
        :return:
        """
        height, width = source_image.shape[:2]
        print(height, width)
        fps = 16
        self.make_out_dir(output_dir)
        # video_writer = cv2.VideoWriter(os.path.join(output_dir, "test_video.avi"),
        #                                cv2.VideoWriter_fourcc(*'XVID'), fps, (width, width))
        # video_writer = cv2.VideoWriter(os.path.join(output_dir, "test_video.avi"),
        #                                cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, (width, width))
        video_writer = cv2.VideoWriter(os.path.join(output_dir, "test_video.avi"),
                                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, width))
        self.print_and_log("Video setting: fps is {} and the frame size is {}".format(fps, (width, width)))
        self.print_and_log("Start converting")
        row_index = 0
        while True:
            if row_index + width > height:
                break
            image_temp = source_image[row_index: row_index + width, :, :]
            video_writer.write(image_temp)
            self.print_and_log("The {}th frame with shape of {}".format(row_index + 1, image_temp.shape))
            row_index = row_index + 1
        video_writer.release()
        self.print_and_log("Convert end")

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
# if __name__ == "__main__":
    # # 根据图像生成视频
    # image = cv2.imread("stitching_by_human.png")
    # project_address = os.getcwd()
    # method = Method()
    # method.generate_video_from_image(image, os.path.join(project_address, "result"))
    # # sub_image = method.resize_image(image, 0.5)
    # # cv2.imwrite("stitching_by_human.png", sub_image)