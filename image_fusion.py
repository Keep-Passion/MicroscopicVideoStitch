import numpy as np
import cv2
import math
from utility import Method
# import torch
# import torch.nn as nn
# import PIL.Image
# import torchvision.transforms as transforms

class ImageFusion(Method):

    @staticmethod
    def fuse_by_average(images):
        """
        均值融合，取两个重合区域逐像素的均值
        :param images: 输入两个相同区域的图像
        :return:融合后的图像
        """
        (last_image, next_image) = images
        # 由于相加后数值可能溢出，需要转变类型
        fuse_region = np.uint8((last_image.astype(int) + next_image.astype(int)) / 2)
        return fuse_region

    @staticmethod
    def fuse_by_maximum(images):
        """
        最大值融合,取两个重合区域逐像素的最大值
        :param images: 输入两个相同区域的图像
        :return:融合后的图像
        """
        (last_image, next_image) = images
        fuse_region = np.maximum(last_image, next_image)
        return fuse_region

    @staticmethod
    def fuse_by_minimum(images):
        """
        最小值融合,取两个重合区域逐像素的最小值
        :param images: 输入两个相同区域的图像
        :return: 融合后的图像
        """

        (last_image, next_image) = images
        fuse_region = np.minimum(last_image, next_image)
        # print(np.unique(last_image))
        # print(np.unique(next_image))
        # cv2.namedWindow("show", 0)
        # cv2.imshow("show", np.hstack([last_image, next_image]).astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.imwrite("test.tif", np.hstack([last_image, next_image]).astype(np.uint8))
        return fuse_region

    @staticmethod
    def get_weights_matrix(images):
        """
        获取权值矩阵
        :param images: 带融合两幅图像
        :return: last_weight_mat,next_weight_mat
        """
        (last_image, next_image) = images
        last_weight_mat = np.ones(last_image.shape, dtype=np.float32)
        next_weight_mat = np.ones(next_image.shape, dtype=np.float32)
        row, col = last_image.shape[:2]
        next_weight_mat_1 = next_weight_mat.copy()
        next_weight_mat_2 = next_weight_mat.copy()
        # 获取四条线的相加和，判断属于哪种模式
        compare_list = [np.count_nonzero(last_image[0: row // 2, 0: col // 2] > 0),
                        np.count_nonzero(last_image[row // 2: row, 0: col // 2] > 0),
                        np.count_nonzero(last_image[row // 2: row, col // 2: col] > 0),
                        np.count_nonzero(last_image[0: row // 2, col // 2: col] > 0)]
        index = compare_list.index(min(compare_list))
        if index == 2:
            # 重合区域在imageA的上左部分
            # self.printAndWrite("上左")
            row_index = 0
            col_index = 0
            for j in range(1, col):
                for i in range(row - 1, -1, -1):
                    if last_image[i, col - j] != -1:
                        row_index = i + 1
                        break
                if row_index != 0:
                    break
            for i in range(col - 1, -1, -1):
                if last_image[row_index, i] != -1:
                    col_index = i + 1
                    break
            # 赋值
            for i in range(row_index + 1):
                if row_index == 0:
                    row_index = 1
                next_weight_mat_1[row_index - i, :] = (row_index - i) * 1 / row_index
            for i in range(col_index + 1):
                if col_index == 0:
                    col_index = 1
                next_weight_mat_2[:, col_index - i] = (col_index - i) * 1 / col_index
            next_weight_mat = next_weight_mat_1 * next_weight_mat_2
            last_weight_mat = 1 - next_weight_mat
        # elif leftCenter != 0 and bottomCenter != 0 and upCenter == 0 and rightCenter == 0:
        elif index == 3:
            # 重合区域在imageA的下左部分
            # self.printAndWrite("下左")
            row_index = 0
            col_index = 0
            for j in range(1, col):
                for i in range(row):
                    if last_image[i, col - j] != -1:
                        row_index = i - 1
                        break
                if row_index != 0:
                    break
            for i in range(col - 1, -1, -1):
                if last_image[row_index, i] != -1:
                    col_index = i + 1
                    break
            # 赋值
            for i in range(row_index, row):
                if row_index == 0:
                    row_index = 1
                next_weight_mat_1[i, :] = (row - i - 1) * 1 / (row - row_index - 1)
            for i in range(col_index + 1):
                if col_index == 0:
                    col_index = 1
                next_weight_mat_2[:, col_index - i] = (col_index - i) * 1 / col_index
            next_weight_mat = next_weight_mat_1 * next_weight_mat_2
            last_weight_mat = 1 - next_weight_mat
        # elif rightCenter != 0 and bottomCenter != 0 and upCenter == 0 and leftCenter == 0:
        elif index == 0:
            # 重合区域在imageA的下右部分
            row_index = 0
            col_index = 0
            for j in range(0, col):
                for i in range(row):
                    if last_image[i, j] != -1:
                        row_index = i - 1
                        break
                if row_index != 0:
                    break
            for i in range(col):
                if last_image[row_index, i] != -1:
                    col_index = i - 1
                    break
            # 赋值
            for i in range(row_index, row):
                if row_index == 0:
                    row_index = 1
                next_weight_mat_1[i, :] = (row - i - 1) * 1 / (row - row_index - 1)
            for i in range(col_index, col):
                if col_index == 0:
                    col_index = 1
                next_weight_mat_2[:, i] = (col - i - 1) * 1 / (col - col_index - 1)
            next_weight_mat = next_weight_mat_1 * next_weight_mat_2
            last_weight_mat = 1 - next_weight_mat
        # elif upCenter != 0 and rightCenter != 0 and leftCenter == 0 and bottomCenter == 0:
        elif index == 1:
            # 重合区域在imageA的上右部分
            # self.printAndWrite("上右")
            row_index = 0
            col_index = 0
            for j in range(0, col):
                for i in range(row - 1, -1, -1):
                    if last_image[i, j] != -1:
                        row_index = i + 1
                        break
                if row_index != 0:
                    break
            for i in range(col):
                if last_image[row_index, i] != -1:
                    col_index = i - 1
                    break
            for i in range(row_index + 1):
                if row_index == 0:
                    row_index = 1
                next_weight_mat_1[row_index - i, :] = (row_index - i) * 1 / row_index
            for i in range(col_index, col):
                if col_index == 0:
                    col_index = 1
                next_weight_mat_2[:, i] = (col - i - 1) * 1 / (col - col_index - 1)
            next_weight_mat = next_weight_mat_1 * next_weight_mat_2
            last_weight_mat = 1 - next_weight_mat
        return last_weight_mat, next_weight_mat

    def fuse_by_fade_in_and_fade_out(self, images, offset):
        """
        渐入渐出融合
        :param images:输入两个相同区域的图像
        :param dx: 第二张图像相对于第一张图像原点在x方向上的位移
        :param dy: 第二张图像相对于第一张图像原点在y方向上的位移
        :return:融合后的图像
        """
        (last_image, next_image) = images
        (dx, dy) = offset
        row, col = last_image.shape[:2]
        last_weight_mat = np.ones(last_image.shape, dtype=np.float32)
        next_weight_mat = np.ones(next_image.shape, dtype=np.float32)
        # self.printAndWrite("    ratio: "  + str(np.count_nonzero(imageA > -1) / imageA.size))
        if np.count_nonzero(last_image > -1) / last_image.size > 0.65:
            # 如果对于imageA中，非0值占比例比较大，则认为是普通融合
            # 根据区域的行列大小来判断，如果行数大于列数，是水平方向
            if col <= row:
                # self.printAndWrite("普通融合-水平方向")
                for i in range(0, col):
                    # print(dy)
                    if dy >= 0:
                        last_weight_mat[:, i] = last_weight_mat[:, i] * i * 1.0 / col
                        next_weight_mat[:, col - i - 1] = next_weight_mat[:, col - i - 1] * i * 1.0 / col
                    elif dy < 0:
                        last_weight_mat[:, i] = last_weight_mat[:, i] * (col - i) * 1.0 / col
                        next_weight_mat[:, col - i - 1] = next_weight_mat[:, col - i - 1] * (col - i) * 1.0 / col
            # 根据区域的行列大小来判断，如果列数大于行数，是竖直方向
            elif row < col:
                # self.printAndWrite("普通融合-竖直方向")
                for i in range(0, row):
                    if dx <= 0:
                        last_weight_mat[i, :] = last_weight_mat[i, :] * i * 1.0 / row
                        next_weight_mat[row - i - 1, :] = next_weight_mat[row - i - 1, :] * i * 1.0 / row
                    elif dx > 0:
                        last_weight_mat[i, :] = last_weight_mat[i, :] * (row - i) * 1.0 / row
                        next_weight_mat[row - i - 1, :] = next_weight_mat[row - i - 1, :] * (row - i) * 1.0 / row
        else:
            # 如果对于imageA中，非0值占比例比较小，则认为是拐角融合
            # self.printAndWrite("拐角融合")
            last_weight_mat, next_weight_mat = self.get_weights_matrix(images)
        last_image[last_image < 0] = next_image[last_image < 0]
        next_image[next_image == -1] = 0
        result = last_weight_mat * last_image.astype(np.int) + next_weight_mat * next_image.astype(np.int)
        result[result < 0] = 0
        result[result > 255] = 255
        fuse_region = np.uint8(result)
        return fuse_region

    def fuse_by_trigonometric(self, images, offset):
        """
        三角函数融合
        引用自《一种三角函数权重的图像拼接算法》知网
        :param images:输入两个相同区域的图像
        :param dx: 第二张图像相对于第一张图像原点在x方向上的位移
        :param dy: 第二张图像相对于第一张图像原点在y方向上的位移
        :return:融合后的图像
        """
        (last_image, next_image) = images
        (dx, dy) = offset
        row, col = last_image.shape[:2]
        last_weight_mat = np.ones(last_image.shape, dtype=np.float64)
        next_weight_mat = np.ones(next_image.shape, dtype=np.float64)
        if np.count_nonzero(last_image > -1) / last_image.size > 0.65:
            # 如果对于imageA中，非0值占比例比较大，则认为是普通融合
            # 根据区域的行列大小来判断，如果行数大于列数，是水平方向
            if col <= row:
                # self.printAndWrite("普通融合-水平方向")
                for i in range(0, col):
                    if dy >= 0:
                        last_weight_mat[:, i] = last_weight_mat[:, i] * i * 1.0 / col
                        next_weight_mat[:, col - i - 1] = next_weight_mat[:, col - i - 1] * i * 1.0 / col
                    elif dy < 0:
                        last_weight_mat[:, i] = last_weight_mat[:, i] * (col - i) * 1.0 / col
                        next_weight_mat[:, col - i - 1] = next_weight_mat[:, col - i - 1] * (col - i) * 1.0 / col
            # 根据区域的行列大小来判断，如果列数大于行数，是竖直方向
            elif row < col:
                # self.printAndWrite("普通融合-竖直方向")
                for i in range(0, row):
                    if dx <= 0:
                        last_weight_mat[i, :] = last_weight_mat[i, :] * i * 1.0 / row
                        next_weight_mat[row - i - 1, :] = next_weight_mat[row - i - 1, :] * i * 1.0 / row
                    elif dx > 0:
                        last_weight_mat[i, :] = last_weight_mat[i, :] * (row - i) * 1.0 / row
                        next_weight_mat[row - i - 1, :] = next_weight_mat[row - i - 1, :] * (row - i) * 1.0 / row
        else:
            # 如果对于imageA中，非0值占比例比较小，则认为是拐角融合
            # self.printAndWrite("拐角融合")
            last_weight_mat, next_weight_mat = self.get_weights_matrix(images)

        last_weight_mat = np.power(np.sin(last_weight_mat * math.pi / 2), 2)
        next_weight_mat = 1 - last_weight_mat

        last_image[last_image < 0] = next_image[last_image < 0]
        next_image[next_image == -1] = 0
        result = last_weight_mat * last_image.astype(np.int) + next_weight_mat * next_image.astype(np.int)
        result[result < 0] = 0
        result[result > 255] = 255
        fuse_region = np.uint8(result)
        return fuse_region

    def fuse_by_possion_image_editing(self, images):
        """
        泊松融合
        引用自: Rez P, Gangnet M, Blake A. Poisson image editing.[J].
        Acm Transactions on Graphics, 2003, 22(3):313-318.
        :param images: 输入两个相同区域的图像
        :return: 融合后的图像
        """
        (last_image, next_image) = images
        fuse_region = last_image
        return fuse_region

    pyramid_level = 4

    def fuse_by_multi_band_blending(self, images):
        """
        多分辨率样条融合,重合区域逐像素各取权重0.5，然后使用拉普拉斯金字塔融合
        引用自：《A Multiresolution Spline With Application to Image Mosaics》
        :param images: 输入两个相同区域的图像
        :return: 融合后的图像
        """
        (last_image, next_image) = images
        last_lp, last_gp = self.get_laplacian_pyramid(last_image)
        next_lp, next_gp = self.get_laplacian_pyramid(next_image)
        fuse_lp = []
        for i in range(self.pyramid_level):
            fuse_lp.append(last_lp[i] * 0.5 + next_lp[i] * 0.5)
        fuse_region = np.uint8(self.reconstruct(fuse_lp))
        return fuse_region

    block_size = 5

    def fuse_by_spatial_frequency(self, images):
        """
        空间频率融合
        引用自：《Combination of images with diverse focuses using the spatial frequency》
        :param images:输入两个相同区域的图像
        :return:融合后的图像
        """
        (last_image, next_image) = images
        weight_matrix = self.get_spatial_frequency_matrix(images)
        fuse_region = last_image * weight_matrix + next_image * (1 - weight_matrix)
        # print(np.amin(fuse_region), np.amax(fuse_region))
        return fuse_region.astype(np.uint8)

    def get_spatial_frequency_matrix(self, images):
        """
        空间频率滤波的权值矩阵计算
        :param images: 输入两个相同区域的图像
        :return: 权值矩阵，第一张比第二张清晰的像素点为1，第二张比第一张清晰的像素点为0
        """
        (last_image, next_image) = images
        row, col = last_image.shape[0:2]
        weight_matrix = np.ones(last_image.shape)

        choice_full_zeros = np.array([(0, 0, 0),
                                      (0, 1, 0),
                                      (0, 0, 0)])
        choice_full_ones = np.array([(1, 1, 1),
                                     (1, 0, 1),
                                     (1, 1, 1)])
        # 矩阵并行处理
        rf_last_total_pow = (last_image[1:row, 1:col].__sub__(last_image[1:row, 0:col - 1])).__pow__(2)  # 向左减并平方
        cf_last_total_pow = (last_image[1:row, 1:col].__sub__(last_image[0:row - 1, 1:col])).__pow__(2)  # 向上减并平方
        rf_next_total_pow = (next_image[1:row, 1:col].__sub__(next_image[1:row, 0:col - 1])).__pow__(2)
        cf_next_total_pow = (next_image[1:row, 1:col].__sub__(next_image[0:row - 1, 1:col])).__pow__(2)
        #         print(rf_last_total_pow.shape)

        if row % self.block_size == 0:
            row_num = row // self.block_size - 1
        else:
            row_num = row // self.block_size
        if col % self.block_size == 0:
            col_num = col // self.block_size - 1
        else:
            col_num = col // self.block_size

        fusion_choice = np.ones((row_num + 1, col_num + 1))

        # 下一步需要并行化的部分：
        for i in range(row_num + 1):
            for j in range(col_num + 1):
                if i < row_num and j < col_num:
                    row_end_position = (i + 1) * self.block_size
                    col_end_position = (j + 1) * self.block_size
                elif i < row_num and j == col_num:
                    row_end_position = (i + 1) * self.block_size
                    col_end_position = col
                elif i == row_num and j < col_num:
                    row_end_position = row
                    col_end_position = (j + 1) * self.block_size
                else:
                    row_end_position = row
                    col_end_position = col

                sf_last = np.sum(
                    rf_last_total_pow[(i * self.block_size):row_end_position, (j * self.block_size):col_end_position]) + np.sum(
                    cf_last_total_pow[(i * self.block_size):row_end_position, (j * self.block_size):col_end_position])
                sf_next = np.sum(
                    rf_next_total_pow[(i * self.block_size):row_end_position, (j * self.block_size):col_end_position]) + np.sum(
                    cf_next_total_pow[(i * self.block_size):row_end_position, (j * self.block_size):col_end_position])

                if sf_last < sf_next:  # 该区域第二张图像较清楚 赋值为0
                    fusion_choice[i, j] = 0
                    weight_matrix[(i * self.block_size) + 1:row_end_position + 1,
                    (j * self.block_size) + 1:col_end_position + 1] -= 1

                # 用 3 * 3 的 majority filter 过滤一遍
                if i > 1 and j > 1:
                    if np.all(fusion_choice[(i - 2):(i + 1), (j - 2):(j + 1)] == choice_full_zeros):  # 取全0
                        # print("满足010")
                        fusion_choice[i - 1, j - 1] = 0
                        weight_matrix[(i - 1) * self.block_size + 1:i * self.block_size + 1,
                        (j - 1) * self.block_size + 1:j * self.block_size + 1] -= 1
                    elif np.all(fusion_choice[(i - 2):(i + 1), (j - 2):(j + 1)] == choice_full_ones):  # 取全1
                        # print("满足101")
                        fusion_choice[i - 1, j - 1] = 1
                        weight_matrix[(i - 1) * self.block_size + 1:i * self.block_size + 1,
                        (j - 1) * self.block_size + 1:j * self.block_size + 1] += 1
        weight_matrix = weight_matrix.astype(np.float32)
        return weight_matrix

    @staticmethod
    def calculate_spatial_frequency(image):
        """
        计算空间频率
        :param image:输入图像
        :return:返回其空间频率 sf
        """
        rf_temp = 0
        cf_temp = 0
        row, col = image.shape[0:2]
        image_temp = image.astype(int)
        for i in range(row):
            for j in range(col):
                if j < col - 1:
                    rf_temp = rf_temp + np.square(image_temp[i, j + 1] - image_temp[i, j])
                if i < row - 1:
                    cf_temp = cf_temp + np.square(image_temp[i + 1, j] - image_temp[i, j])
        rf = np.sqrt(float(rf_temp) / float(row * col))
        cf = np.sqrt(float(cf_temp) / float(row * col))
        sf = np.sqrt(np.square(rf) + np.square(cf))
        return sf

    def fuse_by_sf_and_mbb(self, images):
        """
        多分辨率样条和空间频率融合叠加,空间频率生成的权值矩阵，生成高斯金字塔然后与拉普拉斯金字塔结合，
        最后将上述金字塔生成图像
        :param images: 输入两个相同区域的图像
        :return: 融合后的图像
        """
        (last_image, next_image) = images
        last_lp, last_gp = self.get_laplacian_pyramid(last_image)
        next_lp, next_gp = self.get_laplacian_pyramid(next_image)
        weight_matrix = self.get_spatial_frequency_matrix(images)
        # wm_gp 为weight_matrix的高斯金字塔
        wm_gp = self.get_gaussian_pyramid(weight_matrix)
        fuse_lp = []
        for i in range(self.pyramid_level):
            fuse_lp.append(last_lp[i] * wm_gp[self.pyramid_level - i - 1] +
                           next_lp[i] * (1 - wm_gp[self.pyramid_level - i - 1]))
        fuse_region = np.uint8(self.reconstruct(fuse_lp))
        return fuse_region
    
    def fuse_by_deep_fuse(self, images):
        """
        Deep fuse 融合，引用自：
        Prabhakar K R, Srikar V S, Babu R V.DeepFuse: A Deep Unsupervised Approach
        for Exposure Fusion with Extreme Exposure Image Pairs[C]//ICCV. 2017: 4724-4732.
        :param images: 输入两个相同区域的图像
        :return: 融合后的图像
        """
        (last_image, next_image) = images
        fuse_region = 0
        return fuse_region

    input_size_cnn = 256
    center_size = 200
    # original_transform = transforms.Compose([
    #     transforms.ToTensor(),
    # ])
    max_input_num = 20
    def fuse_by_our_framework(self, images):
        """
        本文算法融合，引用自：
        :param images: 输入两个相同区域的图像
        :return: 融合后的图像
        """
        # 在这里提供接口，包括模型参数引用地址、模型，具体用什么模型在其他py文件封装
        (last_image, next_image) = images
        fuse_region = np.zeros(last_image.shape)

        # 使用overlap-tile策略裁切
        last_input_list = []
        next_input_list = []
        padding_num = int((self.input_size_cnn - self.center_size) // 2)
        last_expand = cv2.copyMakeBorder(last_image, padding_num, padding_num, padding_num, padding_num,
                                         cv2.BORDER_REFLECT)
        next_expand = cv2.copyMakeBorder(next_image, padding_num, padding_num, padding_num, padding_num,
                                         cv2.BORDER_REFLECT)
        row_expand, col_expand = last_expand.shape[0:2]
        row_have_remain = False
        col_have_remain = False
        if row_expand % self.input_size_cnn == 0:
            row_have_remain = True
        if col_expand % self.input_size_cnn == 0:
            col_have_remain = True
        row_num = row_expand // self.input_size_cnn
        col_num = col_expand // self.input_size_cnn
        for i in range(row_num + 1):
            for j in range(col_num + 1):
                row_start = i * self.input_size_cnn
                row_end = (i + 1) * self.input_size_cnn
                col_start = j * self.input_size_cnn
                col_end = (j + 1) * self.input_size_cnn
                if i == row_num:
                    if row_have_remain:
                        row_start = row_expand - self.input_size_cnn
                        row_end = row_expand
                    else:
                        break
                if j == col_num:
                    if col_have_remain:
                        col_start = col_expand - self.input_size_cnn
                        col_end = col_expand
                    else:
                        continue
                last_input_list.append(last_expand[row_start: row_end, col_start:col_end])
                next_input_list.append(next_expand[row_start: row_end, col_start:col_end])
        input_num = len(last_input_list)
        # 将list转化为Tensor
        last_input_tensors = self.trans_list_to_Tensor(last_input_list, input_num)
        next_input_tensors = self.trans_list_to_Tensor(next_input_list, input_num)

        # 分步送入网络
        output_tensors = None
        if input_num < self.max_input_num:
            output_tensors = self.run_network(last_input_tensors, next_input_tensors).data
        else:
            output_tensors = torch.zeros([input_num, 1, self.input_size_cnn, self.input_size_cnn])
            implement_num = 0
            while implement_num < input_num:
                remain_num = input_num - implement_num
                if remain_num > self.max_input_num:
                    output_tensors[implement_num:implement_num + self.max_input_num, :] = \
                        self.run_network(
                            last_input_tensors[implement_num:implement_num + self.max_input_num, :, :, :],
                            next_input_tensors[implement_num:implement_num + self.max_input_num, :, :, :],
                        ).data
                else:
                    output_tensors[implement_num:input_num, :] = \
                        self.run_network(
                            last_input_tensors[implement_num: input_num, :, :, :],
                            next_input_tensors[implement_num: input_num, :, :, :],
                        ).data
                implement_num += self.max_input_num

        # 将Tensor转化为list
        output_list = self.trans_Tensor_to_list(output_tensors, input_num)

        # 放置于图像中各个位置
        row, col = last_image.shape[0:2]
        row_num = row // self.input_size_cnn
        col_num = col // self.input_size_cnn
        for index, item in enumerate(output_list):
            if row_have_remain:
                row_start = index // (col + 1) * self.center_size
                row_end = ((index // (col + 1)) + 1) * self.center_size
                if row_start == row_num - 1:
                    row_start = col - self.center_size
                    row_end = col
            else:
                row_start = row - self.center_size
                row_end = row
            if col_have_remain:
                col_start = index % (col + 1) * self.center_size
                col_end = ((index % (col + 1)) + 1) * self.center_size
                if col_start == col_num - 1:
                    col_start = col - self.center_size
                    col_end = col
            else:
                col_start = index % col * self.center_size
                col_end = ((index % col) + 1) * self.center_size
            fuse_region[row_start, row_end, col_start, col_end] = \
                output_list[index][padding_num: padding_num + self.center_size, padding_num: padding_num + self.center_size]
        return fuse_region

    def run_network(self, last_input, next_input):
        pass

    def trans_list_to_Tensor(self, input_list, input_num):
        input_tensors = torch.zeros((input_num, 1, self.input_size_cnn, self.input_size_cnn))
        for index, array in enumerate(input_list):
            image_pil = PIL.Image.fromarray(array)
            input_tensors[index, :, :, :] = self.original_transform(image_pil)
        return input_tensors

    def trans_Tensor_to_list(self, output_tensors, input_num):
        output_list = []
        for i in range(input_num):
            output_list.append(output_tensors[i, :, :, :].clamp(0,255).numpy().transpose((1, 2, 0)))
        return output_list

    # # 权值矩阵归一化
    # def normalize_weight_mat(self, weight_mat):
    #     min_value = weight_mat.min()
    #     max_value = weight_mat.max()
    #     out = (weight_mat - min_value) / (max_value - min_value) * 255
    #     return out

    def get_gaussian_pyramid(self, input_image):
        """
        获得图像的高斯金字塔
        :param input_image:输入图像
        :return: 高斯金字塔，以list形式返回，第一个是原图，以此类推
        """
        g = input_image.copy().astype(np.float64)
        gp = [g]  # 金字塔结构存到list中
        for i in range(self.pyramid_level):
            g = cv2.pyrDown(g)
            gp.append(g)
        return gp

    def get_laplacian_pyramid(self, input_image):
        """
        求一张图像的拉普拉斯金字塔
        :param input_image: 输入图像
        :return: 拉普拉斯金字塔(laplacian_pyramid, lp, 从小到大)，高斯金字塔(gaussian_pyramid, gp,从大到小),
                  均以list形式
        """
        gp = self.get_gaussian_pyramid(input_image)
        lp = [gp[self.pyramid_level - 1]]
        for i in range(self.pyramid_level - 1, -1, -1):
            ge = cv2.pyrUp(gp[i])
            ge = cv2.resize(ge, (gp[i - 1].shape[1], gp[i - 1].shape[0]), interpolation=cv2.INTER_CUBIC)
            lp.append(cv2.subtract(gp[i - 1], ge))
        return lp, gp

    @staticmethod
    def reconstruct(input_pyramid):
        """
        根据拉普拉斯金字塔重构图像，该list第一个是最小的原图，后面是更大的拉普拉斯表示
        :param input_pyramid: 输入的金字塔
        :return: 返回重构的结果图
        """
        construct_result = input_pyramid[0]
        for i in range(1, len(input_pyramid)):
            construct_result = cv2.pyrUp(construct_result)
            construct_result = cv2.resize(construct_result, (input_pyramid[i].shape[1], input_pyramid[i].shape[0]),
                                          interpolation=cv2.INTER_CUBIC)
            construct_result = cv2.add(construct_result, input_pyramid[i])
        return construct_result