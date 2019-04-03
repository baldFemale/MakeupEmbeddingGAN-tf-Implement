import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import cv2
import dlib
import os
import random
import time
import vgg16
from scipy.misc import imsave
from model import *

batch_size = 1
img_height = 256
img_width = 256
img_layer = 3
pool_size = 50
max_images = 1

to_restore = False
to_train = True
to_test = False
save_training_images = False
out_path = "./output"
check_dir = "./output/checkpoints/"


class MakeupEmbeddingGAN():
    def input_setup(self):
        filename_A = tf.train.match_filenames_once("./Japanese_after_rotate/*.jpg")
        self.queue_length_A = tf.size(filename_A)
        filename_B = tf.train.match_filenames_once("./smokey_after_rotate/*.jpg")
        self.queue_length_B = tf.size(filename_B)

        filename_A_queue = tf.train.string_input_producer(filename_A)
        filename_B_queue = tf.train.string_input_producer(filename_B)

        image_reader = tf.WholeFileReader()
        _, image_file_A = image_reader.read(filename_A_queue)
        _, image_file_B = image_reader.read(filename_B_queue)
        self.image_A = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_A),[256,256]),127.5),1)
        self.image_B = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_B),[256,256]),127.5),1)


    def get_mask(self,input_face, detector, predictor,window=5):
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
            cv2.fillPoly(lip_mask, [np.array(temp[48:60]).reshape((-1, 1, 2))], (255, 255, 255))
            cv2.fillPoly(lip_mask, [np.array(temp[60:68]).reshape((-1, 1, 2))], (0, 0, 0))

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

            cv2.polylines(face_mask, [np.array(temp[17:22]).reshape(-1, 1, 2)], False, (0, 0, 0), 7)
            cv2.polylines(face_mask, [np.array(temp[22:27]).reshape(-1, 1, 2)], False, (0, 0, 0), 7)
            cv2.fillPoly(face_mask, [np.array(temp[36:42]).reshape((-1, 1, 2))], (0, 0, 0))
            cv2.fillPoly(face_mask, [np.array(temp[42:48]).reshape((-1, 1, 2))], (0, 0, 0))
            cv2.fillPoly(face_mask, [np.array(temp[48:60]).reshape((-1, 1, 2))], (0, 0, 0))
            return lip_mask,eye_mask,face_mask

    def input_read(self,sess):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        num_file_A = sess.run(self.queue_length_A)
        num_file_B = sess.run(self.queue_length_B)

        self.fake_images_A = np.zeros((pool_size,1,img_height,img_width,img_layer))
        self.fake_images_B = np.zeros((pool_size,1,img_height,img_width,img_layer))

        self.A_input = np.zeros((max_images,batch_size,img_height,img_width,img_layer))
        self.B_input = np.zeros((max_images,batch_size,img_height,img_width,img_layer))
        self.A_input_mask = np.zeros((max_images,3,img_height,img_width))
        self.B_input_mask = np.zeros((max_images,3,img_height,img_width))

        cur_A = 0
        for i in range(max_images):
            image_tensor = sess.run(self.image_A)
            if image_tensor.size==img_width*img_height*img_layer:
                temp = ((image_tensor+1)*127.5).astype(np.uint8)
                res = self.get_mask(temp,self.detector,self.predictor)
                if res is not None:
                    self.A_input[i] = image_tensor.reshape((batch_size,img_height,img_width,img_layer))
                    self.A_input_mask[cur_A][0] = np.equal(res[0],255)
                    self.A_input_mask[cur_A][1] = np.equal(res[1],255)
                    self.A_input_mask[cur_A][2] = np.equal(res[2],255)
                    cur_A+=1

        cur_B = 0
        for i in range(max_images):
            image_tensor = sess.run(self.image_B)
            if image_tensor.size==img_width*img_height*img_layer:
                temp = ((image_tensor+1)*127.5).astype(np.uint8)
                res = self.get_mask(temp,self.detector,self.predictor)
                if res is not None:
                    self.B_input[i] = image_tensor.reshape((batch_size,img_height,img_width,img_layer))
                    self.B_input_mask[cur_B][0] = np.equal(res[0],255)
                    self.B_input_mask[cur_B][1] = np.equal(res[1],255)
                    self.B_input_mask[cur_B][2] = np.equal(res[2],255)
                    cur_B += 1
        self.train_num = min(cur_A,cur_B)
        print("load img number: ",self.train_num)

        coord.request_stop()
        coord.join(threads)



    def model_setup(self):
        self.input_A = tf.placeholder(dtype=tf.float32,shape=[batch_size,img_height,img_width,img_layer],name="input_A")
        self.input_B = tf.placeholder(dtype=tf.float32,shape=[batch_size,img_height,img_width,img_layer],name="input_B")

        self.input_A_mask = tf.placeholder(tf.bool,[3,img_height,img_width],name="input_A_mask")
        self.input_B_mask = tf.placeholder(tf.bool,[3,img_height,img_width],name="input_B_mask")

        self.fake_pool_A = tf.placeholder(dtype=tf.float32,shape=[None,img_height,img_width,img_layer],name="fake_pool_A")
        self.fake_pool_B = tf.placeholder(dtype=tf.float32,shape=[None,img_height,img_width,img_layer],name="fake_pool_B")

        self.global_step =tf.Variable(0,trainable=False,name="global_step")
        self.num_fake_inputs = 0

        self.lr = tf.placeholder(dtype=tf.float32,shape=[],name="lr")
        self.predictor = dlib.shape_predictor("./preTrainedModel/shape_predictor_68_face_landmarks.dat")
        self.detector = dlib.get_frontal_face_detector()

        with tf.variable_scope("Model") as scope:
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

            self.fake_pool_A_rec = generate_discriminator(self.fake_pool_A,name="d_A")
            self.fake_pool_B_rec = generate_discriminator(self.fake_pool_B,name="d_B")

            scope.reuse_variables()
            self.cyc_A = Tnet(self.fake_B,self.gammas_A,self.betas_A,name="Tnet")

            scope.reuse_variables()
            self.cyc_B = Tnet(self.fake_A,self.gammas_B,self.betas_B,name="Tnet")

            self.perc_A = tf.cast(tf.image.resize_images((self.input_A+1)*127.5,[224,224]),tf.float32)
            self.perc_B = tf.cast(tf.image.resize_images((self.input_B+1)*127.5, [224, 224]), tf.float32)
            self.perc_fake_B = tf.cast(tf.image.resize_images((self.fake_B+1)*127.5, [224, 224]), tf.float32)
            self.perc_fake_A = tf.cast(tf.image.resize_images((self.fake_A+1)*127.5, [224, 224]), tf.float32)
            self.perc = self.perc_loss_cal(tf.concat([self.perc_A,self.perc_B,self.perc_fake_B,self.perc_fake_A],axis=0))
            percep_norm,var = tf.nn.moments(self.perc, [1, 2], keep_dims=True)
            self.perc = tf.divide(self.perc,tf.add(percep_norm,1e-5))


    def perc_loss_cal(self,input_tensor):
        vgg = vgg16.Vgg16("./preTrainedModel/vgg16.npy")
        vgg.build(input_tensor)
        return vgg.conv4_1


    def histogram_loss_cal(self,source,template,source_mask,template_mask):
        shape = tf.shape(source)
        source = tf.reshape(source, [1, -1])
        template = tf.reshape(template, [1, -1])
        source_mask = tf.reshape(source_mask,[-1, 256 * 256])
        template_mask = tf.reshape(template_mask,[-1,256*256])

        source = tf.boolean_mask(source, source_mask)
        template = tf.boolean_mask(template, template_mask)

        his_bins = 255

        max_value = tf.reduce_max([tf.reduce_max(source), tf.reduce_max(template)])
        min_value = tf.reduce_min([tf.reduce_min(source), tf.reduce_min(template)])

        hist_delta = (max_value - min_value) / his_bins
        hist_range = tf.range(min_value, max_value, hist_delta)
        hist_range = tf.add(hist_range, tf.divide(hist_delta, 2))

        s_hist = tf.histogram_fixed_width(source, [min_value, max_value], his_bins, dtype=tf.int32)
        t_hist = tf.histogram_fixed_width(template, [min_value, max_value], his_bins, dtype=tf.int32)

        s_quantiles = tf.cumsum(s_hist)
        s_last_element = tf.subtract(tf.size(s_quantiles), tf.constant(1))
        s_quantiles = tf.divide(s_quantiles, tf.gather(s_quantiles, s_last_element))

        t_quantiles = tf.cumsum(t_hist)
        t_last_element = tf.subtract(tf.size(t_quantiles), tf.constant(1))
        t_quantiles = tf.divide(t_quantiles, tf.gather(t_quantiles, t_last_element))

        nearest_indices = tf.map_fn(lambda x: tf.argmin(tf.abs(tf.subtract(t_quantiles, x))), s_quantiles,
                                    dtype=tf.int32)
        s_bin_index = tf.to_int32(tf.divide(source, hist_delta))
        s_bin_index = tf.clip_by_value(s_bin_index, 0, 254)

        matched_to_t = tf.gather(hist_range, tf.gather(nearest_indices, s_bin_index))
        # Using the same normalization as Gatys' style transfer: A huge variation--the normalization scalar is different according to different image
        # normalization includes variation constraints may be better
        matched_to_t = tf.subtract(tf.div(matched_to_t,127.5),1)
        source = tf.subtract(tf.divide(source,127.5),1)
        return tf.reduce_mean(tf.squared_difference(matched_to_t,source))


    def loss_cals(self):
        cyc_loss = tf.reduce_mean(tf.abs(self.input_A-self.cyc_A))+tf.reduce_mean(tf.abs(self.input_B-self.cyc_B))
        disc_loss_A = tf.reduce_mean(tf.squared_difference(self.fake_A_rec,1))
        disc_loss_B = tf.reduce_mean(tf.squared_difference(self.fake_B_rec,1))

        histogram_loss_r_lip = self.histogram_loss_cal(tf.cast((self.fake_B[0, :, :, 0] + 1) * 127.5, dtype=tf.float32),
                                                       tf.cast((self.input_B[0, :, :, 0] + 1) * 127.5,
                                                               dtype=tf.float32), self.input_A_mask[0],
                                                       self.input_B_mask[0])

        histogram_loss_r_eye = self.histogram_loss_cal(tf.cast((self.fake_B[0, :, :, 0] + 1) * 127.5, dtype=tf.float32),
                                                       tf.cast((self.input_B[0, :, :, 0] + 1) * 127.5,
                                                               dtype=tf.float32), self.input_A_mask[1],
                                                       self.input_B_mask[1])

        histogram_loss_r_face = self.histogram_loss_cal(
            tf.cast((self.fake_B[0, :, :, 0] + 1) * 127.5, dtype=tf.float32),
            tf.cast((self.input_B[0, :, :, 0] + 1) * 127.5,
                    dtype=tf.float32), self.input_A_mask[2],
            self.input_B_mask[2])
        histogram_loss_r = histogram_loss_r_face + histogram_loss_r_lip + histogram_loss_r_eye

        histogram_loss_g_lip = self.histogram_loss_cal(tf.cast((self.fake_B[0, :, :, 1] + 1) * 127.5, dtype=tf.float32),
                                                       tf.cast((self.input_B[0, :, :, 1] + 1) * 127.5,
                                                               dtype=tf.float32), self.input_A_mask[0],
                                                       self.input_B_mask[0])

        histogram_loss_g_eye = self.histogram_loss_cal(tf.cast((self.fake_B[0, :, :, 1] + 1) * 127.5, dtype=tf.float32),
                                                       tf.cast((self.input_B[0, :, :, 1] + 1) * 127.5,
                                                               dtype=tf.float32), self.input_A_mask[1],
                                                       self.input_B_mask[1])

        histogram_loss_g_face = self.histogram_loss_cal(
            tf.cast((self.fake_B[0, :, :, 1] + 1) * 127.5, dtype=tf.float32),
            tf.cast((self.input_B[0, :, :, 1] + 1) * 127.5,
                    dtype=tf.float32), self.input_A_mask[2],
            self.input_B_mask[2])
        histogram_loss_g = histogram_loss_g_lip + histogram_loss_g_face + histogram_loss_g_eye

        histogram_loss_b_lip = self.histogram_loss_cal(tf.cast((self.fake_B[0, :, :, 2] + 1) * 127.5, dtype=tf.float32),
                                                       tf.cast((self.input_B[0, :, :, 2] + 1) * 127.5,
                                                               dtype=tf.float32), self.input_A_mask[0],
                                                       self.input_B_mask[0])

        histogram_loss_b_eye = self.histogram_loss_cal(tf.cast((self.fake_B[0, :, :, 2] + 1) * 127.5, dtype=tf.float32),
                                                       tf.cast((self.input_B[0, :, :, 2] + 1) * 127.5,
                                                               dtype=tf.float32), self.input_A_mask[1],
                                                       self.input_B_mask[1])

        histogram_loss_b_face = self.histogram_loss_cal(
            tf.cast((self.fake_B[0, :, :, 2] + 1) * 127.5, dtype=tf.float32),
            tf.cast((self.input_B[0, :, :, 2] + 1) * 127.5,
                    dtype=tf.float32), self.input_A_mask[2],
            self.input_B_mask[2])

        histogram_loss_b = histogram_loss_b_lip + histogram_loss_b_face + histogram_loss_b_eye
        makeup_loss = histogram_loss_r + histogram_loss_g + histogram_loss_b

        perceptual_loss = tf.reduce_mean(tf.squared_difference(self.perc[0], self.perc[2])) + tf.reduce_mean(
            tf.squared_difference(self.perc[1], self.perc[3]))

        g_loss = cyc_loss*10+disc_loss_A+disc_loss_B+makeup_loss+perceptual_loss*0.05

        d_loss_A = tf.reduce_mean(tf.squared_difference(self.rec_A,1))+tf.reduce_mean(tf.square(self.fake_pool_A_rec))
        d_loss_B = tf.reduce_mean(tf.squared_difference(self.rec_B,1))+tf.reduce_mean(tf.square(self.fake_pool_B_rec))

        optimizer = tf.train.AdamOptimizer(self.lr,beta1=0.5)
        # optimizer = tfp.optimizer.StochasticGradientLangevinDynamics(learning_rate=self.lr,preconditioner_decay_rate=0.99)

        self.model_vars = tf.trainable_variables()

        g_vars = [var for var in self.model_vars if "Pnet" in var.name or "Tnet" in var.name]
        d_A_vars = [var for var in self.model_vars if "d_A" in var.name]
        d_B_vars = [var for var in self.model_vars if "d_B" in var.name]

        self.g_trainer = optimizer.minimize(g_loss,var_list=g_vars)
        self.d_A_trainer = optimizer.minimize(d_loss_A,var_list=d_A_vars)
        self.d_B_trainer = optimizer.minimize(d_loss_B,var_list=d_B_vars)

        for var in self.model_vars:
            print(var.name)


        self.disc_A_loss_sum = tf.summary.scalar("disc_loss_A",disc_loss_A)
        self.disc_B_loss_sum = tf.summary.scalar("disc_loss_B",disc_loss_B)
        self.cyc_loss_sum = tf.summary.scalar("cyc_loss",cyc_loss)
        self.makeup_loss_sum = tf.summary.scalar("makeup_loss",makeup_loss)
        self.percep_loss_sum = tf.summary.scalar("perceptual_loss",perceptual_loss)
        self.g_loss_sum = tf.summary.scalar("g_loss",g_loss)

        self.g_summary = tf.summary.merge([
            self.disc_A_loss_sum,self.disc_B_loss_sum,self.cyc_loss_sum,self.makeup_loss_sum,self.percep_loss_sum,self.g_loss_sum
        ],"g_summary")

        self.d_A_loss = tf.summary.scalar("d_loss_A",d_loss_A)
        self.d_B_loss = tf.summary.scalar("d_loss_B",d_loss_B)


    def save_training_images(self, sess, epoch):
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
        self.input_setup()
        self.model_setup()
        self.loss_cals()

        init = [tf.local_variables_initializer(),tf.global_variables_initializer()]
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            self.input_read(sess)

            if to_restore:
                chkpt_fanem = tf.train.latest_checkpoint(check_dir)
                saver.restore(sess,chkpt_fanem)

            writer = tf.summary.FileWriter("./output/2")

            if not os.path.exists(check_dir):
                os.makedirs(check_dir)

            for epoch in range(sess.run(self.global_step),100):
                print("In the epoch ",epoch)
                saver.save(sess,os.path.join(check_dir,"MakeupEmbeddingGAN"),global_step=epoch)

                if epoch<100:
                    curr_lr = 0.0002
                else:
                    curr_lr = 0.0002-0.0002*(epoch-100)/100

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
                imsave("./output/imgs/test/fakeA_" + str(i) + ".jpg",
                       ((fake_A_temp[0] + 1) * 127.5).astype(np.uint8))
                imsave("./output/imgs/test/fakeB_" + str(i) + ".jpg",
                       ((fake_B_temp[0] + 1) * 127.5).astype(np.uint8))


def main():
    model = MakeupEmbeddingGAN()
    if to_train:
        model.train()
    elif to_test:
        model.test()


if __name__=="__main__":
    main()