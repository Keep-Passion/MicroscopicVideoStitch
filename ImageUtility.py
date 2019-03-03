import cv2
import os
import skimage.measure


class Method:
    # 关于 GPU 加速的设置
    is_gpu_available = False

    # 关于打印信息的设置
    input_dir = ""
    is_out_log_file = False
    log_file = "evaluate.txt"
    is_print_screen = True

    def print_and_log(self, content):
        if self.is_print_screen:
            print(content)
        if self.is_out_log_file:
            f = open(os.path.join(self.input_dir, self.log_file), "a")
            f.write(content)
            f.write("\n")
            f.close()

    @staticmethod
    def make_out_dir(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError:
            pass

    @staticmethod
    def delete_folder(dir_address):
        file_list = os.listdir(dir_address)
        file_num = len(file_list)
        if file_num != 0:
            for i in range(file_num):
                path = os.path.join(dir_address, file_list[i])
                if os.path.isdir(path) is False:
                    os.remove(path)
        os.rmdir(dir_address)

    @staticmethod
    def resize_image(origin_image, resize_times, inter_method=cv2.INTER_AREA):
        (h, w) = origin_image.shape
        resize_h = int(h * resize_times)
        resize_w = int(w * resize_times)
        # cv2.INTER_AREA是测试后最好的方法
        resized_image = cv2.resize(origin_image, (resize_w, resize_h), interpolation=inter_method)
        return resized_image

    def generate_video_from_image(self, source_image, output_dir):
        """
        Convert sour_image to video, simply crop sub-image in source_image in row direction with one pixel increment
        :param source_image: source_image
        :param output_dir: video output dir
        :return:
        """
        height, width, depth = source_image.shape
        fps = 16
        self.make_out_dir(output_dir)
        # video_writer = cv2.VideoWriter(os.path.join(output_dir, "test_video.avi"),
        #                                cv2.VideoWriter_fourcc(*'XVID'), fps, (width, width))
        # video_writer = cv2.VideoWriter(os.path.join(output_dir, "test_video.avi"),
        #                                cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, (width, width))
        video_writer = cv2.VideoWriter(os.path.join(output_dir, "test_video.avi"),
                                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, width))
        print("Video setting: fps is {} and the frame size is {}".format(fps, (width, width)))
        print("Start converting")
        row_index = 0
        while True:
            if row_index + width > height:
                break
            image_temp = source_image[row_index: row_index + width, :, :]
            video_writer.write(image_temp)
            print("The {}th frame with shape of {}".format(row_index + 1, image_temp.shape))
            row_index = row_index + 1
        video_writer.release()
        print("Convert end")

    @staticmethod
    def compare_result_gt(stitch_image, gt_image):
        assert stitch_image.shape == gt_image.shape, "The shape of two image is not same"
        mse_score = skimage.measure.compare_mse(stitch_image, gt_image)
        psnr_score = skimage.measure.compare_psnr(stitch_image, gt_image)
        ssim_score = skimage.measure.compare_ssim(stitch_image, gt_image)
        print(" The mse is {}, psnr is {}, ssim is {}".format(mse_score, psnr_score, ssim_score))


if __name__ == "__main__":
    # 根据图像生成视频
    image = cv2.imread("stitching_by_human.png")
    project_address = os.getcwd()
    method = Method()
    method.generate_video_from_image(image, os.path.join(project_address, "result"))
