import numpy as np
import cv2
import math
import ImageUtility as Utility


class ImageFusion(Utility.Method):

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
        :return:融合后的图像
        """
        (last_image, next_image) = images
        fuse_region = np.minimum(last_image, next_image)
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

    def fuse_by_fade_in_and_fade_out(self, images, dx, dy):
        """
        渐入渐出融合
        :param images:输入两个相同区域的图像
        :param dx: 第二张图像相对于第一张图像原点在x方向上的位移
        :param dy: 第二张图像相对于第一张图像原点在y方向上的位移
        :return:融合后的图像
        """
        (last_image, next_image) = images
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
        last_image[last_image == -1] = 0
        next_image[next_image == -1] = 0
        result = last_weight_mat * last_image.astype(np.int) + next_weight_mat * next_image.astype(np.int)
        result[result < 0] = 0
        result[result > 255] = 255
        fuse_region = np.uint8(result)
        return fuse_region

    def fuse_by_trigonometric(self, images, dx, dy):
        """
        三角函数融合
        引用自《一种三角函数权重的图像拼接算法》知网
        :param images:输入两个相同区域的图像
        :param dx: 第二张图像相对于第一张图像原点在x方向上的位移
        :param dy: 第二张图像相对于第一张图像原点在y方向上的位移
        :return:融合后的图像
        """
        (last_image, next_image) = images
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

        last_image[last_image == -1] = 0
        next_image[next_image == -1] = 0
        result = last_weight_mat * last_image.astype(np.int) + next_weight_mat * next_image.astype(np.int)
        result[result < 0] = 0
        result[result > 255] = 255
        fuse_region = np.uint8(result)
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

    block_size = 4

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
        return fuse_region.astype(np.uint8)

    def get_spatial_frequency_matrix(self, images):
        """
        空间频率滤波的权值矩阵计算
        :param images: 输入两个相同区域的图像
        :return: 权值矩阵，第一张比第二张清晰的像素点为1，第二张比第一张清晰的像素点为0
        """
        (last_image, next_image) = images
        weight_matrix = np.ones(last_image.shape)
        if self.is_gpu_available:   # gpu模式
            pass
        else:   # cpu模式
            pass
        return weight_matrix

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