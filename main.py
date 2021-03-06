# 在cycleGAN和ZM-Net基础上完成的MakeupEmbeddingGA
# cycleGAN部分的代码参考了官方开源的tensorflow implement
# https://github.com/hardikbansal/CycleGAN
# ZM-Net部分的代码参考了
# https://github.com/dikatok/styletransfer/tree/master/3-zmnet这个项目中hashmap的结构
# 为了更清晰的展示代码逻辑，去掉了多GPU并行计算的部分
# 完整的代码逻辑根据layer->model->main的结构建立


import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import tensorflow as tf

import numpy as np
from sklearn.manifold import TSNE
import cv2
import dlib
import random
import pickle
import time
import vgg16
from scipy.misc import imsave
from model import *

batch_size = 1
gpu_num = 1
img_height = 256
img_width = 256
img_layer = 3
# 负例池的大小
pool_size = 50
# 读取的图片数量
max_images = 1050

to_restore = False
to_train = True
to_test = False
save_training_images = False
# 输出结果路径
out_path = "./output"
# 模型保存路径
check_dir = "./output/checkpoints/"


class MakeupEmbeddingGAN():
    def input_setup(self):
        """
        读取数据，生成迭代器
        :return: None
        """
        filename_A = tf.train.match_filenames_once("./all/images/non-makeup/*.png")
        self.queue_length_A = tf.size(filename_A)
        filename_B = tf.train.match_filenames_once("./all/images/makeup/*.png")
        self.queue_length_B = tf.size(filename_B)

        filename_A_queue = tf.train.string_input_producer(filename_A,shuffle=False)
        filename_B_queue = tf.train.string_input_producer(filename_B,shuffle=False)

        image_reader = tf.WholeFileReader()
        _, image_file_A = image_reader.read(filename_A_queue)
        _, image_file_B = image_reader.read(filename_B_queue)
        self.image_A = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_A),[256,256]),127.5),1)
        self.image_B = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_B),[256,256]),127.5),1)


    def input_read(self, sess):
        """
        根据迭代器，生成输入
        :param sess: tf.Session()
        :return: None
        """
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        num_file_A = sess.run(self.queue_length_A)
        num_file_B = sess.run(self.queue_length_B)

        self.fake_images_A = np.zeros((pool_size, 1, img_height, img_width, img_layer))
        self.fake_images_B = np.zeros((pool_size, 1, img_height, img_width, img_layer))

        self.A_input = np.zeros((max_images, batch_size, img_height, img_width, img_layer))
        self.B_input = np.zeros((max_images, batch_size, img_height, img_width, img_layer))
        self.A_input_mask = np.zeros((max_images, 3, img_height, img_width))
        self.B_input_mask = np.zeros((max_images, 3, img_height, img_width))

        # 这一步是筛选那些人脸识别模型识别不出的图片，将其排除在输入之外
        cur_A = 0
        for i in range(max_images):
            image_tensor = sess.run(self.image_A)
            if image_tensor.size == img_width * img_height * img_layer:
                temp = ((image_tensor + 1) * 127.5).astype(np.uint8)
                res = self.get_mask(temp, self.detector, self.predictor)
                if res is not None:
                    self.A_input[i] = image_tensor.reshape((batch_size, img_height, img_width, img_layer))
                    self.A_input_mask[cur_A][0] = np.equal(res[0], 255)
                    self.A_input_mask[cur_A][1] = np.equal(res[1], 255)
                    self.A_input_mask[cur_A][2] = np.equal(res[2], 255)
                    cur_A += 1

        cur_B = 0
        for i in range(max_images):
            image_tensor = sess.run(self.image_B)
            if image_tensor.size == img_width * img_height * img_layer:
                temp = ((image_tensor + 1) * 127.5).astype(np.uint8)
                res = self.get_mask(temp, self.detector, self.predictor)
                if res is not None:
                    self.B_input[i] = image_tensor.reshape((batch_size, img_height, img_width, img_layer))
                    self.B_input_mask[cur_B][0] = np.equal(res[0], 255)
                    self.B_input_mask[cur_B][1] = np.equal(res[1], 255)
                    self.B_input_mask[cur_B][2] = np.equal(res[2], 255)
                    cur_B += 1

        self.train_num = min(cur_A, cur_B)
        print("load img number: ", self.train_num)

        coord.request_stop()
        coord.join(threads)


    def get_mask(self,input_face, detector, predictor,window=5):
        """
        分别对人脸的面部、唇部、眼影区域进行遮罩
        :param input_face: 需要进行遮罩的图像
        :param detector: 检测人脸的预训练模型
        :param predictor: 产生人脸68个特征点的预训练模型
        :param window: 在眼睛周围取得眼影区域的窗口大小
        :return: 元组（唇部遮罩，眼影遮罩，面部遮罩）
        """
        gray = cv2.cvtColor(input_face, cv2.COLOR_BGR2GRAY)
        dets = detector(gray, 1)

        for face in dets:
            shape = predictor(input_face, face)
            temp = []
            for pt in shape.parts():
                temp.append([pt.x, pt.y])
            lip_mask = np.zeros([256, 256])
            eye_mask = np.zeros([256,256])
            face_mask = np.full((256, 256), 255).astype(np.uint8)

            # 唇部遮罩
            cv2.fillPoly(lip_mask, [np.array(temp[48:60]).reshape((-1, 1, 2))], (255, 255, 255))
            cv2.fillPoly(lip_mask, [np.array(temp[60:68]).reshape((-1, 1, 2))], (0, 0, 0))

            # 左眼的眼影遮罩
            left_left = min(x[0] for x in temp[36:42])
            left_right = max(x[0] for x in temp[36:42])
            left_bottom = min(x[1] for x in temp[36:42])
            left_top = max(x[1] for x in temp[36:42])
            left_rectangle = np.array(
                [[left_left - window, left_top + window], [left_right + window, left_top + window],
                 [left_right + window, left_bottom - window], [left_left - window, left_bottom - window]]
            ).reshape((-1, 1, 2))
            cv2.fillPoly(eye_mask, [left_rectangle], (255, 255, 255))
            cv2.fillPoly(eye_mask, [np.array(temp[36:42]).reshape((-1, 1, 2))], (0, 0, 0))

            # 右眼的眼影遮罩
            right_left = min(x[0] for x in temp[42:48])
            right_right = max(x[0] for x in temp[42:48])
            right_bottom = min(x[1] for x in temp[42:48])
            right_top = max(x[1] for x in temp[42:48])
            right_rectangle = np.array(
                [[right_left - window, right_top + window], [right_right + window, right_top + window],
                 [right_right + window, right_bottom - window], [right_left - window, right_bottom - window]]
            ).reshape((-1, 1, 2))
            cv2.fillPoly(eye_mask, [right_rectangle], (255, 255, 255))
            cv2.fillPoly(eye_mask, [np.array(temp[42:47]).reshape((-1, 1, 2))], (0, 0, 0))

            # 面部遮罩
            cv2.polylines(face_mask, [np.array(temp[17:22]).reshape(-1, 1, 2)], False, (0, 0, 0), 7)
            cv2.polylines(face_mask, [np.array(temp[22:27]).reshape(-1, 1, 2)], False, (0, 0, 0), 7)
            cv2.fillPoly(face_mask, [np.array(temp[36:42]).reshape((-1, 1, 2))], (0, 0, 0))
            cv2.fillPoly(face_mask, [np.array(temp[42:48]).reshape((-1, 1, 2))], (0, 0, 0))
            cv2.fillPoly(face_mask, [np.array(temp[48:60]).reshape((-1, 1, 2))], (0, 0, 0))

            return lip_mask,eye_mask,face_mask



    # def average_gradients(self,tower_grads):
    #     """
    #     使用模型并行进行多GPU并行计算时平均梯度
    #     :param tower_grads: 多个tower产的梯度
    #     :return: average_grads: 平均后的梯度
    #     """
    #     average_grads = []
    #     for grad_and_vars in zip(*tower_grads):
    #         grads = []
    #         for g, _ in grad_and_vars:
    #             expend_g = tf.expand_dims(g, 0)
    #             grads.append(expend_g)
    #         grad = tf.concat(grads, 0)
    #         grad = tf.reduce_mean(grad, 0)
    #         v = grad_and_vars[0][1]
    #         grad_and_var = (grad, v)
    #         average_grads.append(grad_and_var)
    #     return average_grads


    def model_setup(self):
        """
        模型初始化
        :return: None
        """
        # 创建输入、遮罩、负例的placeholder
        self.input_A = tf.placeholder(dtype=tf.float32,shape=[batch_size,img_height,img_width,img_layer],name="input_A")
        self.input_B = tf.placeholder(dtype=tf.float32,shape=[batch_size,img_height,img_width,img_layer],name="input_B")
        self.input_A_mask = tf.placeholder(tf.bool,[3,img_height,img_width],name="input_A_mask")
        self.input_B_mask = tf.placeholder(tf.bool,[3,img_height,img_width],name="input_B_mask")
        self.fake_pool_A = tf.placeholder(dtype=tf.float32,shape=[None,img_height,img_width,img_layer],name="fake_pool_A")
        self.fake_pool_B = tf.placeholder(dtype=tf.float32,shape=[None,img_height,img_width,img_layer],name="fake_pool_B")

        # 设定学习率,步数，当前负例池中的负例个数
        self.global_step =tf.Variable(0,trainable=False,name="global_step")
        self.num_fake_inputs = 0
        self.lr = tf.placeholder(dtype=tf.float32,shape=[],name="lr")

        # 读入用于面部遮罩的预训练模型
        self.predictor = dlib.shape_predictor("./preTrainedModel/shape_predictor_68_face_landmarks.dat")
        self.detector = dlib.get_frontal_face_detector()

        with tf.variable_scope("Model") as scope:
            # 风格预测网络PNet和风格转换网络TNet都被复用了四次
            # 风格预测网络PNet分别根据内容人像A,风格人像B,生成的假风格人像fake_B和生成的假内容人像fake_A生成了
            # 对应的缩放系数gammas和位移系数betas
            # 风格转换网络TNet分别根据生成的gammas和betas，将内容人像A转换至假风格人像fake_B，将风格人像B转换
            # 至假内容人像fake_A，将假的风格人像fake_B转换回内容人像cyc_A，将假的内容人像fake_A转换回风格人像
            # cyc_B
            self.gammas_B,self.betas_B = Pnet(self.input_B,name="Pnet")
            self.fake_B = Tnet(self.input_A,self.gammas_B,self.betas_B,name="Tnet")
            self.rec_A = generate_discriminator(self.input_A, name="d_A")
            self.fake_B_rec = generate_discriminator(self.fake_B,name="d_B")
            scope.reuse_variables()

            self.gammas_A,self.betas_A = Pnet(self.input_A,name="Pnet")
            self.fake_A = Tnet(self.input_B,self.gammas_A,self.betas_A,name="Tnet")
            self.rec_B = generate_discriminator(self.input_B,name="d_B")
            self.fake_A_rec = generate_discriminator(self.fake_A,name="d_A")
            scope.reuse_variables()
            
            self.gammas_fakeA,self.betas_fakeA = Pnet(self.fake_A,name="Pnet")
            self.fake_pool_A_rec = generate_discriminator(self.fake_pool_A,name="d_A")
            self.fake_pool_B_rec = generate_discriminator(self.fake_pool_B,name="d_B")
            scope.reuse_variables()

            self.gammas_fakeB,self.betas_fakeB = Pnet(self.fake_B,name="Pnet")
            self.cyc_A = Tnet(self.fake_B,self.gammas_fakeA,self.betas_fakeA,name="Tnet")
            scope.reuse_variables()

            self.cyc_B = Tnet(self.fake_A,self.gammas_fakeB,self.betas_fakeB,name="Tnet")
            scope.reuse_variables()

            # 原本的模型到此结束，接下来是根据计院老师的想法cascade起来的部分，代码逻辑和上半部分是差不多的，只是再来一遍而已

            self.gammas_B_cyc,self.betas_B_cyc = Pnet(self.cyc_B,name="Pnet")
            self.fake_cyc_B = Tnet(self.cyc_A,self.gammas_B_cyc,self.betas_B_cyc,name="Tnet")
            self.rec_cyc_A = generate_discriminator(self.cyc_A,name="d_A")
            self.fake_cyc_B_rec = generate_discriminator(self.fake_cyc_B,name="d_B")
            scope.reuse_variables()

            self.gammas_A_cyc,self.betas_A_cyc = Pnet(self.cyc_A,name="Pnet")
            self.fake_cyc_A = Tnet(self.cyc_B,self.gammas_A_cyc,self.betas_A_cyc,name="Tnet")
            self.rec_cyc_B = generate_discriminator(self.cyc_B,name="d_B")
            self.fake_cyc_A_rec = generate_discriminator(self.fake_cyc_A,name="d_A")
            scope.reuse_variables()

            self.gammas_fake_cyc_A,self.betas_fake_cyc_A = Pnet(self.fake_cyc_A,name="Pnet")
            self.cyc_cyc_A = Tnet(self.fake_cyc_B,self.gammas_fake_cyc_A,self.betas_fake_cyc_A,name="Tnet")
            scope.reuse_variables()

            self.gammas_fake_cyc_B,self.betas_fake_cyc_B = Pnet(self.fake_cyc_B,name="Pnet")
            self.cyc_cyc_B = Tnet(self.fake_cyc_A,self.gammas_fake_cyc_B,self.betas_fake_cyc_B,name="Tnet")

            # 准备输入VGG16模型的数据，因为VGG16模型本身的输入是224*224，所以需要这一步额外处理一下
            # 根据后标，准备输入VGG16的数据分别是内容人像A, 风格人像B，生成的假风格人像fake_B和假内容人像fake_A
            # 后标中带有cyc的都是根据计院老师的想法后来加的
            self.perc_A = tf.cast(tf.image.resize_images((self.input_A+1)*127.5,[224,224]),tf.float32)
            self.perc_B = tf.cast(tf.image.resize_images((self.input_B+1)*127.5, [224, 224]), tf.float32)
            self.perc_fake_B = tf.cast(tf.image.resize_images((self.fake_B+1)*127.5, [224, 224]), tf.float32)
            self.perc_fake_A = tf.cast(tf.image.resize_images((self.fake_A+1)*127.5, [224, 224]), tf.float32)
            self.perc_cyc_A = tf.cast(tf.image.resize_images((self.cyc_A+1)*127.5,[224,224]),tf.float32)
            self.perc_cyc_B = tf.cast(tf.image.resize_images((self.cyc_B+1)*127.5,[224,224]),tf.float32)
            self.perc_fake_cyc_B = tf.cast(tf.image.resize_images((self.fake_cyc_B+1)*127.5,[224,224]),tf.float32)
            self.perc_fake_cyc_A = tf.cast(tf.image.resize_images((self.fake_cyc_A+1)*127.5,[224,224]),tf.float32)

            # 输入到预训练的VGG16模型中，返回中间层并进行标准化
            self.perc = self.perc_loss_cal(tf.concat([
                self.perc_A, self.perc_B, self.perc_fake_B, self.perc_fake_A,self.perc_cyc_A,self.perc_cyc_B,self.perc_fake_cyc_B,self.perc_fake_cyc_A
            ],axis=0))
            percep_norm,var = tf.nn.moments(self.perc, [1, 2], keep_dims=True)
            self.perc = tf.divide(self.perc,tf.add(percep_norm,1e-5))


    def perc_loss_cal(self,input_tensor):
        """
        得到输入图像在预训练的VGG16模型中的中间层，用于计算感知损失
        :param input_tensor: 输入人像
        :return: 输入人像的conv4_1参数
        """
        vgg = vgg16.Vgg16("./preTrainedModel/vgg16.npy")
        vgg.build(input_tensor)
        return vgg.conv4_1


    def histogram_loss_cal(self,source,template,source_mask,template_mask):
        """
        根据给定的图像template和其遮罩对原图像source的特定区域进行相应的直方图匹配，计算原图像在直方图匹配前后的差异

        :param source: 原图像
        :param template: 匹配的目标图像
        :param source_mask: 原图像的遮罩
        :param template_mask: 目标图像的遮罩
        :return: 原图像在匹配前后的均方误差
        """
        shape = tf.shape(source)
        source = tf.reshape(source, [1, -1])
        template = tf.reshape(template, [1, -1])
        source_mask = tf.reshape(source_mask,[-1, 256 * 256])
        template_mask = tf.reshape(template_mask,[-1,256*256])

        # 根据遮罩将特定区域之外的像素点全部略去
        source = tf.boolean_mask(source, source_mask)
        template = tf.boolean_mask(template, template_mask)

        # 获得原图像和目标图像在像素值上的直方图分布
        his_bins = 255

        max_value = tf.reduce_max([tf.reduce_max(source), tf.reduce_max(template)])
        min_value = tf.reduce_min([tf.reduce_min(source), tf.reduce_min(template)])

        hist_delta = (max_value - min_value) / his_bins
        hist_range = tf.range(min_value, max_value, hist_delta)
        hist_range = tf.add(hist_range, tf.divide(hist_delta, 2))

        s_hist = tf.histogram_fixed_width(source, [min_value, max_value], his_bins, dtype=tf.int32)
        t_hist = tf.histogram_fixed_width(template, [min_value, max_value], his_bins, dtype=tf.int32)

        # 将二者的直方图分布转换为累计百分比的形式
        s_quantiles = tf.cumsum(s_hist)
        s_last_element = tf.subtract(tf.size(s_quantiles), tf.constant(1))
        s_quantiles = tf.divide(s_quantiles, tf.gather(s_quantiles, s_last_element))

        t_quantiles = tf.cumsum(t_hist)
        t_last_element = tf.subtract(tf.size(t_quantiles), tf.constant(1))
        t_quantiles = tf.divide(t_quantiles, tf.gather(t_quantiles, t_last_element))

        # 根据目标图像的累计百分比获得与原图像中每一像素值最接近的目标像素值
        # 这个求最接近的方法不是最理想的，因为按照这样拟合后的直方图相对于原图像的直方分布来讲会有更大的波动
        nearest_indices = tf.map_fn(lambda x: tf.argmin(tf.abs(tf.subtract(t_quantiles, x))), s_quantiles,
                                    dtype=tf.int64)

        s_bin_index = tf.to_int64(tf.divide(source, hist_delta))
        s_bin_index = tf.clip_by_value(s_bin_index, 0, 254)
        matched_to_t = tf.gather(hist_range, tf.gather(nearest_indices, s_bin_index))

        # 对原图像和匹配后的图像进行标准化，计算均方误差
        matched_to_t = tf.subtract(tf.div(matched_to_t,127.5),1)
        source = tf.subtract(tf.divide(source,127.5),1)
        return tf.reduce_mean(tf.squared_difference(matched_to_t,source))


    def loss_cals(self):
        """
        计算损失函数
        cyc_loss: cyc_loss 完全参照cycleGAN中的循环一致损失
        cyc_cascade_loss: 后来增加的第二个cycle循环中的损失
        disc_loss_A: 对抗训练中优化生成器时判别器的损失函数，使fake_A尽可能为真，和cycleGAN设计一致
        disc_loss_B: 对抗训练中优化生成器时判别器的损失函数，使fake_B尽可能为真，和cycleGAN设计一致
        disc_loss_cyc_A: 第二个cycle循环中相应的判别损失
        disc_loss_cyc_B: 第二个cycle循环中相应的判别损失
        makeup_loss: 对每对风格人像和内容人像，分别在RGB三通道上对眼影、唇部、面部进行一次直方图匹配，因此要
        在9次直方图匹配的基础上计算化妆损失，沿用了beautyGAN的设计
        perceptual_loss: 在conv4_1上的均方误差
        d_loss_A: 对抗训练中优化判别器时的损失函数，优化目标是尽可能将A识别为真，将从负例池中随机产生的图片识别为假
        d_loss_B: 对抗训练中优化判别器时的损失函数，优化目标是尽可能将B识别为真，将从负例池中随机产生的图片识别为假
        :return:
        """
        # 循环一致损失函数
        cyc_loss = tf.reduce_mean(tf.abs(self.input_A-self.cyc_A))+tf.reduce_mean(tf.abs(self.input_B-self.cyc_B))
        cyc_cascade_loss = tf.reduce_mean(tf.abs(self.cyc_A-self.cyc_cyc_A))+tf.reduce_mean(tf.abs(self.cyc_B-self.cyc_cyc_B))

        # 判别损失
        disc_loss_A = tf.reduce_mean(tf.squared_difference(self.fake_A_rec,1))
        disc_loss_B = tf.reduce_mean(tf.squared_difference(self.fake_B_rec,1))
        disc_loss_cyc_A = tf.reduce_mean(tf.squared_difference(self.fake_cyc_A_rec,1))
        disc_loss_cyc_B = tf.reduce_mean(tf.squared_difference(self.fake_cyc_B_rec,1))


        # 化妆损失，具体权重上把面部的化妆损失权重设定为0
        temp_source = tf.cast((self.fake_B[0, :, :, 0] + 1) * 127.5, dtype=tf.float32)
        temp_template = tf.cast((self.input_B[0, :, :, 0] + 1) * 127.5, dtype=tf.float32)
        histogram_loss_r_lip = self.histogram_loss_cal(temp_source, temp_template, self.input_A_mask[0],
                                                       self.input_B_mask[0])
        histogram_loss_r_eye = self.histogram_loss_cal(temp_source, temp_template, self.input_A_mask[1],
                                                       self.input_B_mask[1])
        histogram_loss_r_face = self.histogram_loss_cal(temp_source, temp_template, self.input_A_mask[2],
                                                        self.input_B_mask[2])
        histogram_loss_r = histogram_loss_r_lip + histogram_loss_r_eye

        temp_source = tf.cast((self.fake_cyc_B[0, :, :, 0] + 1) * 127.5, dtype=tf.float32)
        temp_template = tf.cast((self.cyc_B[0, :, :, 0] + 1) * 127.5, dtype=tf.float32)
        histogram_loss_r_lip_cyc = self.histogram_loss_cal(temp_source, temp_template, self.input_A_mask[0],
                                                       self.input_B_mask[0])
        histogram_loss_r_eye_cyc = self.histogram_loss_cal(temp_source, temp_template, self.input_A_mask[1],
                                                       self.input_B_mask[1])
        histogram_loss_r_face_cyc = self.histogram_loss_cal(temp_source, temp_template, self.input_A_mask[2],
                                                        self.input_B_mask[2])
        histogram_loss_r_cyc = histogram_loss_r_lip_cyc + histogram_loss_r_eye_cyc


        temp_source = tf.cast((self.fake_B[0, :, :, 1] + 1) * 127.5, dtype=tf.float32)
        temp_template = tf.cast((self.input_B[0, :, :, 1] + 1) * 127.5, dtype=tf.float32)
        histogram_loss_g_lip = self.histogram_loss_cal(temp_source, temp_template, self.input_A_mask[0],
                                                       self.input_B_mask[0])
        histogram_loss_g_eye = self.histogram_loss_cal(temp_source, temp_template, self.input_A_mask[1],
                                                       self.input_B_mask[1])
        histogram_loss_g_face = self.histogram_loss_cal(temp_source, temp_template, self.input_A_mask[2],
                                                        self.input_B_mask[2])
        histogram_loss_g = histogram_loss_g_lip + histogram_loss_g_eye

        # 进行第二次cycle循环时的化妆损失
        temp_source = tf.cast((self.fake_cyc_B[0, :, :, 1] + 1) * 127.5, dtype=tf.float32)
        temp_template = tf.cast((self.cyc_B[0, :, :, 1] + 1) * 127.5, dtype=tf.float32)
        histogram_loss_g_lip_cyc = self.histogram_loss_cal(temp_source, temp_template, self.input_A_mask[0],
                                                       self.input_B_mask[0])
        histogram_loss_g_eye_cyc = self.histogram_loss_cal(temp_source, temp_template, self.input_A_mask[1],
                                                       self.input_B_mask[1])
        histogram_loss_g_face_cyc = self.histogram_loss_cal(temp_source, temp_template, self.input_A_mask[2],
                                                        self.input_B_mask[2])
        histogram_loss_g_cyc = histogram_loss_g_lip_cyc + histogram_loss_g_eye_cyc

        temp_source = tf.cast((self.fake_B[0, :, :, 2] + 1) * 127.5, dtype=tf.float32)
        temp_template = tf.cast((self.input_B[0, :, :, 2] + 1) * 127.5, dtype=tf.float32)
        histogram_loss_b_lip = self.histogram_loss_cal(temp_source, temp_template, self.input_A_mask[0],
                                                       self.input_B_mask[0])
        histogram_loss_b_eye = self.histogram_loss_cal(temp_source, temp_template, self.input_A_mask[1],
                                                       self.input_B_mask[1])
        histogram_loss_b_face = self.histogram_loss_cal(temp_source, temp_template, self.input_A_mask[2],
                                                        self.input_B_mask[2])
        histogram_loss_b = histogram_loss_b_lip + histogram_loss_b_eye

        temp_source = tf.cast((self.fake_cyc_B[0, :, :, 2] + 1) * 127.5, dtype=tf.float32)
        temp_template = tf.cast((self.cyc_B[0, :, :, 2] + 1) * 127.5, dtype=tf.float32)
        histogram_loss_b_lip_cyc = self.histogram_loss_cal(temp_source, temp_template, self.input_A_mask[0],
                                                       self.input_B_mask[0])
        histogram_loss_b_eye_cyc = self.histogram_loss_cal(temp_source, temp_template, self.input_A_mask[1],
                                                       self.input_B_mask[1])
        histogram_loss_b_face_cyc = self.histogram_loss_cal(temp_source, temp_template, self.input_A_mask[2],
                                                        self.input_B_mask[2])
        histogram_loss_b_cyc = histogram_loss_b_lip_cyc + histogram_loss_b_eye_cyc

        makeup_loss = histogram_loss_r + histogram_loss_g + histogram_loss_b+histogram_loss_r_cyc+histogram_loss_g_cyc+histogram_loss_b_cyc

        # 感知损失，包括第一个cycle循环的感知损失和第二个cycle循环的感知损失
        perceptual_loss = tf.reduce_mean(tf.squared_difference(self.perc[0], self.perc[2])) + tf.reduce_mean(
            tf.squared_difference(self.perc[1], self.perc[3]))+tf.reduce_mean(
            tf.squared_difference(self.perc[4], self.perc[6]))+tf.reduce_mean(
            tf.squared_difference(self.perc[5],self.perc[7])
        )

        # 生成网络整体损失函数
        g_loss = cyc_loss*20+cyc_cascade_loss*20+disc_loss_A+disc_loss_B+disc_loss_cyc_A+disc_loss_cyc_B+makeup_loss+perceptual_loss*0.05

        d_loss_A = tf.reduce_mean(tf.squared_difference(self.rec_A,1))+2*tf.reduce_mean(tf.square(self.fake_pool_A_rec))+tf.reduce_mean(tf.squared_difference(self.fake_cyc_A_rec,1))
        d_loss_B = tf.reduce_mean(tf.squared_difference(self.rec_B,1))+2*tf.reduce_mean(tf.square(self.fake_pool_B_rec))+tf.reduce_mean(tf.squared_difference(self.fake_cyc_B_rec,1))

        optimizer = tf.train.AdamOptimizer(self.lr,beta1=0.5)
        # optimizer = tfp.optimizer.StochasticGradientLangevinDynamics(learning_rate=self.lr,preconditioner_decay_rate=0.99)

        # 分别优化生成网路和判别网络中的参数
        self.model_vars = tf.trainable_variables()
        g_vars = [var for var in self.model_vars if "Pnet" in var.name or "Tnet" in var.name]
        d_A_vars = [var for var in self.model_vars if "d_A" in var.name]
        d_B_vars = [var for var in self.model_vars if "d_B" in var.name]
        self.g_trainer = optimizer.minimize(g_loss,var_list=g_vars)
        self.d_A_trainer = optimizer.minimize(d_loss_A,var_list=d_A_vars)
        self.d_B_trainer = optimizer.minimize(d_loss_B,var_list=d_B_vars)

        for var in self.model_vars:
            print(var.name)

        # 在tensorboard中记录参数的变化
        self.disc_A_loss_sum = tf.summary.scalar("disc_loss_A",disc_loss_A)
        self.disc_B_loss_sum = tf.summary.scalar("disc_loss_B",disc_loss_B)
        self.disc_A_cyc_loss_sum = tf.summary.scalar("disc_loss_cyc_A", disc_loss_cyc_A)
        self.disc_B_cyc_loss_sum = tf.summary.scalar("disc_loss_cyc_B", disc_loss_cyc_B)
        self.cyc_loss_sum = tf.summary.scalar("cyc_loss",cyc_loss)
        self.cyc_cascade_loss_sum = tf.summary.scalar("cyc_cascade_loss",cyc_cascade_loss)
        self.makeup_loss_sum = tf.summary.scalar("makeup_loss",makeup_loss)
        self.percep_loss_sum = tf.summary.scalar("perceptual_loss",perceptual_loss)
        self.g_loss_sum = tf.summary.scalar("g_loss",g_loss)

        self.g_summary = tf.summary.merge([
            self.disc_A_loss_sum,self.disc_B_loss_sum,self.cyc_loss_sum,self.makeup_loss_sum,self.percep_loss_sum,self.g_loss_sum,
            self.disc_A_cyc_loss_sum,self.disc_B_cyc_loss_sum,self.cyc_cascade_loss_sum,
        ],"g_summary")

        self.d_A_loss = tf.summary.scalar("d_loss_A",d_loss_A)
        self.d_B_loss = tf.summary.scalar("d_loss_B",d_loss_B)


    def save_training_images(self, sess, epoch):
        """
        在训练过程中间保存部分结果
        :param sess: tf.Session()
        :param epoch: 迭代次数
        :return:
        """
        # 检查、创建保存路径
        if not os.path.exists("./output/imgs"):
            os.makedirs("./output/imgs")

        for i in range(0, 10):
            fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run(
                [self.fake_A, self.fake_B, self.cyc_A, self.cyc_B], feed_dict={
                    self.input_A: self.A_input[i],
                    self.input_B: self.B_input[i]
                })
            imsave("./output/imgs/fakeA_" + str(epoch) + "_" + str(i) + ".jpg",
                   ((fake_A_temp[0] + 1) * 127.5).astype(np.uint8))
            imsave("./output/imgs/fakeB_" + str(epoch) + "_" + str(i) + ".jpg",
                   ((fake_B_temp[0] + 1) * 127.5).astype(np.uint8))
            imsave("./output/imgs/cycA_" + str(epoch) + "_" + str(i) + ".jpg",
                   ((cyc_A_temp[0] + 1) * 127.5).astype(np.uint8))
            imsave("./output/imgs/cycB_" + str(epoch) + "_" + str(i) + ".jpg",
                   ((cyc_B_temp[0] + 1) * 127.5).astype(np.uint8))


    def fake_image_pool(self,num_fake,fake,fake_pool):
        """
        沿用了cycleGAN中获得负例的策略，可以更好的防止mode collapse
        也就是每次取得向判别器输入负例的时候，不是直接输入当次循环中产生的负例，而是
        :param num_fake:
        :param fake:
        :param fake_pool:
        :return:
        """
        if num_fake<pool_size:
            fake_pool[num_fake] = fake
            return fake
        else:
            p = random.random()
            if p>0.5:
                random_id = random.randint(0,pool_size-1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else:
                return fake


    def train(self):
        """
        训练模型
        :return:None
        """
        self.input_setup()
        self.model_setup()
        self.loss_cals()

        init = [tf.local_variables_initializer(),tf.global_variables_initializer()]
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            self.input_read(sess)

            # 读取模型
            if to_restore:
                chkpt_fanem = tf.train.latest_checkpoint(check_dir)
                saver.restore(sess,chkpt_fanem)

            writer = tf.summary.FileWriter("./output/2")

            if not os.path.exists(check_dir):
                os.makedirs(check_dir)

            for epoch in range(sess.run(self.global_step),500):
                print("In the epoch ",epoch)
                saver.save(sess,os.path.join(check_dir,"MakeupEmbeddingGAN"),global_step=epoch)

                if epoch<100:
                    curr_lr = 0.0002
                else:
                    curr_lr = 0.0002-0.0002*(epoch-100)/400

                if save_training_images:
                    self.save_training_images(sess,epoch)

                for ptr in range(self.train_num):
                    print("In the iteration ",ptr)
                    print(time.ctime())

                    # optimize generator (Pnet & Tnet)
                    _,summary_str,fake_A_temp,fake_B_temp = sess.run(
                        [self.g_trainer,self.g_summary,self.fake_A,self.fake_B],feed_dict={
                            self.lr:curr_lr,
                            self.input_A:self.A_input[ptr],
                            self.input_B:self.B_input[ptr],
                            self.input_A_mask:self.A_input_mask[ptr],
                            self.input_B_mask:self.B_input_mask[ptr],
                        }
                    )
                    writer.add_summary(summary_str,global_step=epoch*self.train_num+ptr)

                    # optimize d_A
                    fake_pool_A_temp = self.fake_image_pool(self.num_fake_inputs,fake_A_temp,self.fake_images_A)
                    _,summary_str = sess.run(
                        [self.d_A_trainer,self.d_A_loss],feed_dict={
                            self.lr:curr_lr,
                            self.input_A:self.A_input[ptr],
                            self.input_B:self.B_input[ptr],
                            self.fake_pool_A:fake_pool_A_temp,
                        }
                    )
                    writer.add_summary(summary_str,global_step=epoch*self.train_num+ptr)

                    # optimize d_B
                    fake_pool_B_temp = self.fake_image_pool(self.num_fake_inputs,fake_B_temp,self.fake_images_B)
                    _,summary_str = sess.run(
                        [self.d_B_trainer,self.d_B_loss],feed_dict={
                            self.lr:curr_lr,
                            self.input_A:self.A_input[ptr],
                            self.input_B:self.B_input[ptr],
                            self.fake_pool_B:fake_pool_B_temp,
                        }
                    )
                    writer.add_summary(summary_str, global_step=epoch * self.train_num + ptr)

                    self.num_fake_inputs+=1
                sess.run(tf.assign(self.global_step,epoch+1))

            writer.add_graph(sess.graph)


    def test(self):
        """
        测试模型
        :return:None
        """
        print("Testing the results")

        self.input_setup()
        self.model_setup()
        self.loss_cals()

        saver = tf.train.Saver()
        init = [tf.local_variables_initializer(),tf.global_variables_initializer()]

        with tf.Session() as sess:
            sess.run(init)
            self.input_read(sess)

            chkpt_fname = tf.train.latest_checkpoint(check_dir)
            saver.restore(sess,chkpt_fname)

            if not os.path.exists("./output/imgs/test"):
                os.makedirs("./output/imgs/test")

            for i in range(self.train_num):
                fake_A_temp,fake_B_temp = sess.run([self.fake_A,self.fake_B],feed_dict={
                    self.input_A:self.A_input[i],
                    self.input_B:self.B_input[i]
                })
                imsave("./output/imgs/test/A_" + str(i) + ".jpg",((self.A_input[i][0]+1)*127.5).astype(np.uint8))
                imsave("./output/imgs/test/B_" + str(i) + ".jpg",((self.B_input[i][0]+1)*127.5).astype(np.uint8))
                imsave("./output/imgs/test/fakeA_" + str(i) + ".jpg",
                       ((fake_A_temp[0] + 1) * 127.5).astype(np.uint8))
                imsave("./output/imgs/test/fakeB_" + str(i) + ".jpg",
                       ((fake_B_temp[0] + 1) * 127.5).astype(np.uint8))


    def Embedding_test(self):
        """
        利用gammas和betas进行线性的风格融合
        :return: None
        """
        print("Testing Embedding Results")
        self.input_setup()
        self.model_setup()
        self.loss_cals()

        saver = tf.train.Saver()
        init = [tf.local_variables_initializer(),tf.global_variables_initializer()]

        with tf.Session() as sess:
            sess.run(init)
            self.input_read(sess)
            chkpt_fname = tf.train.latest_checkpoint(check_dir)
            saver.restore(sess,chkpt_fname)
            if not os.path.exists("./output/imgs/embedding_test"):
                os.makedirs("./output/imgs/embedding_test")

            content_img = self.A_input[404]
            style_A = self.B_input[404]
            style_B = self.B_input[234]
            self.style_combine(content_img,style_A,style_B,sess,[i/10 for i in range(11)])


    def style_combine(self,content,styleI,styleII,sess,ratio):
        """
        风格融合函数
        :param content: content img
        :param styleI: style A img
        :param styleII: style B img
        :param sess: tf.Session()
        :param ratio: list->[ratio1,ratio2,....]
        :return: None
        """
        gammas_A, betas_A = sess.run([self.gammas_B, self.betas_B], feed_dict={
            self.input_B: styleI,
        })
        gammas_B, betas_B = sess.run([self.gammas_B, self.betas_B], feed_dict={
            self.input_B: styleII,
        })
        imsave("./output/imgs/embedding_test/content.jpg",((content[0]+1)*127.5).astype(np.uint8))
        imsave("./output/imgs/embedding_test/styleI.jpg",((styleI[0]+1)*127.5).astype(np.uint8))
        imsave("./output/imgs/embedding_test/styleII.jpg",((styleII[0]+1)*127.5).astype(np.uint8))
        for r in ratio:
            gammas_input = {}
            betas_input = {}
            for key in gammas_A:
                gammas_input[key] = tf.add(r*gammas_A[key],(1-r)*gammas_B[key])
                betas_input[key] = tf.add(r*betas_A[key],(1-r)*betas_B[key])

            temp_dict_1 = {self.gammas_B[key]:sess.run(gammas_input[key]) for key in gammas_input}
            temp_dict_2 = {self.betas_B[key]:sess.run(betas_input[key]) for key in betas_input}
            temp_dict_3 = {self.input_A:content,self.input_B:self.B_input[404]}
            feed_dict = {}
            feed_dict.update(temp_dict_1)
            feed_dict.update(temp_dict_2)
            feed_dict.update(temp_dict_3)
            fake_A_temp = sess.run([self.fake_B],feed_dict=feed_dict)
            imsave("./output/imgs/embedding_test/fakeA_" +"styleA_"+ str(r*100) + ".jpg",
                   ((fake_A_temp[0] + 1) * 127.5).astype(np.uint8)[0])


def main():
    model = MakeupEmbeddingGAN()
    if to_train:
        model.train()
    elif to_test:
        model.test()

if __name__=="__main__":
    main()
