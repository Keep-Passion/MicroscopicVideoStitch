import cv2
import os
import shutil

class Method:
    # 关于 GPU 加速的设置
    is_gpu_available = True

    # 关于打印信息的设置
    input_dir = ""
    is_out_log_file = False
    log_file = "evaluate.txt"
    is_print_screen = True

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

if __name__ == "__main__":
    # 根据图像生成视频
    image = cv2.imread("stitching_by_human.png")
    project_address = os.getcwd()
    method = Method()
    method.generate_video_from_image(image, os.path.join(project_address, "result"))
    # sub_image = method.resize_image(image, 0.5)
    # cv2.imwrite("stitching_by_human.png", sub_image)
