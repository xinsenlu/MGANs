from __future__ import print_function

import os
from io import StringIO #revised
import scipy.misc
import numpy as np
from glob import glob
from tqdm import trange
from itertools import chain
from collections import deque

from models import *
from utils import save_image

def next(loader):
    return loader.next()[0].data.numpy()

def to_nhwc(image, data_format):
    if data_format == 'NCHW':
        new_image = nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image

def to_nchw_numpy(image):
    if image.shape[3] in [1, 3]:
        new_image = image.transpose([0, 3, 1, 2])
    else:
        new_image = image
    return new_image

def norm_img(image, data_format=None):
    image = image/127.5 - 1.
    if data_format:
        image = to_nhwc(image, data_format)
    return image

def denorm_img(norm, data_format):
    return tf.clip_by_value(to_nhwc((norm + 1)*127.5, data_format), 0, 255)

def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

class Trainer(object):
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader
        self.dataset = config.dataset

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size

        self.step = tf.Variable(0, name='step', trainable=False)

        self.g_lr_1 = tf.Variable(config.g_lr, name='g_lr')
        self.d_lr_1 = tf.Variable(config.d_lr, name='d_lr')
        self.g_lr_2 = tf.Variable(config.g_lr, name='g_lr')
        self.d_lr_2=  tf.Variable(config.d_lr, name='d_lr')
        #self.dis_lr= tf.Variable(config.g_lr, name='dis_lr')#revised

        self.g_lr_1_update = tf.assign(self.g_lr_1, tf.maximum(self.g_lr_1 * 0.5, config.lr_lower_boundary), name='g_lr_1_update')
        self.d_lr_1_update = tf.assign(self.d_lr_1, tf.maximum(self.d_lr_1 * 0.5, config.lr_lower_boundary), name='d_lr_1_update')
        self.g_lr_2_update = tf.assign(self.g_lr_2, tf.maximum(self.g_lr_2 * 0.5, config.lr_lower_boundary), name='g_lr_2_update')
        self.d_lr_2_update = tf.assign(self.d_lr_2, tf.maximum(self.d_lr_2 * 0.5, config.lr_lower_boundary), name='d_lr_2_update')
        #self.dis_lr_update = tf.assign(self.dis_lr, tf.maximum(self.dis_lr * 0.5, config.lr_lower_boundary), name='dis_lr_update')#revised

        self.gamma = config.gamma
        self.o_gamma=1
        self.lambda_k = config.lambda_k

        self.z_num = config.z_num
        self.conv_hidden_num = config.conv_hidden_num
        self.input_scale_size = config.input_scale_size

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.use_gpu = config.use_gpu
        self.data_format = config.data_format

        #new_data=tf.image.crop_to_bounding_box(self.data_loader, 0, 25, 64, 64)
        _, height, width, self.channel = \
                get_conv_shape(self.data_loader, self.data_format)
        self.repeat_num = int(np.log2(height)) - 2

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step
        self.lr_update_step = config.lr_update_step

        self.is_train = config.is_train
        self.build_model()

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        sv = tf.train.Supervisor(logdir=self.model_dir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                save_model_secs=300,
                                global_step=self.step,
                                ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

        if not self.is_train:
            # dirty way to bypass graph finilization error
            g = tf.get_default_graph()
            g._finalized = False

            self.build_test_model()

    def train(self):
        z_fixed = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))

        x_fixed = self.get_image_from_loader()
        save_image(x_fixed, '{}/x_fixed.png'.format(self.model_dir))

        prev_measure = 1
        measure_history = deque([0]*self.lr_update_step, self.lr_update_step)

        for step in trange(self.start_step, self.max_step):
            fetch_dict = {
                "k_1_update": self.k_1_update,
                "k_2_update": self.k_2_update,
                "measure_1": self.measure_1,
                "measure_2": self.measure_2,
            }
            if step % self.log_step == 0:
                fetch_dict.update({
                    "summary": self.summary_op,
                    "g_1_loss": self.g_1_loss,
                    "d_1_loss": self.d_1_loss,
                    "g_2_loss": self.g_2_loss,
                    "d_2_loss": self.d_2_loss,
                    "k_t_1": self.k_t_1,
                    "k_t_2": self.k_t_2,
                    "converge": self.converge
                })
            result = self.sess.run(fetch_dict)

            measure_1 = result['measure_1']
            measure_history.append(measure_1)

            if step % self.log_step == 0:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

                g_1_loss = result['g_1_loss']
                d_1_loss = result['d_1_loss']
                k_t_1 = result['k_t_1']
                converge=result['converge']

                print("[{}/{}] Loss_D: {:.6f} Loss_G: {:.6f} coverge: {:.4f}, k_t: {:.4f}". \
                      format(step, self.max_step, d_1_loss, g_1_loss, converge, k_t_1))

            if step % (self.log_step * 10) == 0:
                x_fake = self.generate(z_fixed, self.model_dir, idx=step)
                self.autoencode(x_fixed, self.model_dir, idx=step, x_fake=x_fake)

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_1_update, self.d_lr_1_update,self.g_lr_2_update,self.d_lr_2_update])

    def build_model(self):
        self.x_1 = tf.image.crop_to_bounding_box(self.data_loader, 0, 0, 64, 64)
        self.x_2 = tf.image.crop_to_bounding_box(self.data_loader, 32, 0, 64, 64)

        x_1 = norm_img(self.x_1)
        x_2 = norm_img(self.x_2)
        self.z = tf.random_uniform(
                (tf.shape(x_1)[0], self.z_num), minval=-1.0, maxval=1.0)
        self.k_t_1 = tf.Variable(0., trainable=False, name='k_t_1')
        self.k_t_2 = tf.Variable(0., trainable=False, name='k_t_2')

        G_1, self.G_1_var = GeneratorCNN_1(
                self.z, self.conv_hidden_num, self.channel,
                self.repeat_num, self.data_format, reuse=False)
        G_2, self.G_2_var = GeneratorCNN_2(
                self.z, self.conv_hidden_num, self.channel,
                self.repeat_num, self.data_format, reuse=False)


        self.G_1 = denorm_img(G_1, self.data_format)
        self.G_2 = denorm_img(G_2, self.data_format)

        Overlapped_G_1= tf.image.crop_to_bounding_box(G_1, 32, 0, 32, 64)
        Overlapped_G_2= tf.image.crop_to_bounding_box(G_2, 0, 0, 32, 64)

        d_1_out, self.D_1_z, self.D_1_var = DiscriminatorCNN_1(
                    tf.concat([G_1, x_1], 0), self.channel, self.z_num, self.repeat_num,
                    self.conv_hidden_num, self.data_format)          
        d_2_out, self.D_2_z, self.D_2_var = DiscriminatorCNN_2(
                    tf.concat([G_2, x_2], 0), self.channel, self.z_num, self.repeat_num,
                    self.conv_hidden_num, self.data_format)




        AE_G_1, AE_x_1 = tf.split(d_1_out, 2)
        AE_G_2, AE_x_2 = tf.split(d_2_out, 2)
        #self.c_gamma=1
         #self.G_1 = denorm_img(G_1, self.data_format)
        self.AE_G_1, self.AE_x_1 = denorm_img(AE_G_1, self.data_format), denorm_img(AE_x_1, self.data_format)

        #self.G_2 = denorm_img(G_2, self.data_format)
        self.AE_G_2, self.AE_x_2 = denorm_img(AE_G_2, self.data_format), denorm_img(AE_x_2, self.data_format)

        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(config.optimizer))

        g_1_optimizer, d_1_optimizer = optimizer(self.g_lr_1), optimizer(self.d_lr_1)
        g_2_optimizer, d_2_optimizer = optimizer(self.g_lr_2), optimizer(self.d_lr_2)
        over_1_optimizer = optimizer(self.g_lr_1)
        over_2_optimizer = optimizer(self.g_lr_2)

        self.d_1_loss_real = tf.reduce_mean(tf.abs(AE_x_1 - x_1))
        self.d_1_loss_fake = tf.reduce_mean(tf.abs(AE_G_1 - G_1))
        self.d_2_loss_real = tf.reduce_mean(tf.abs(AE_x_2 - x_2))
        self.d_2_loss_fake = tf.reduce_mean(tf.abs(AE_G_2 - G_2))

        self.d_1_loss = self.d_1_loss_real - self.k_t_1 * self.d_1_loss_fake
        self.g_1_loss = tf.reduce_mean(tf.abs(AE_G_1 - G_1))
        self.d_2_loss = self.d_2_loss_real - self.k_t_2 * self.d_2_loss_fake
        self.g_2_loss = tf.reduce_mean(tf.abs(AE_G_2 - G_2))
        self.over_1_loss= tf.reduce_mean(self.o_gamma*tf.abs(Overlapped_G_1 - Overlapped_G_2))
        self.over_2_loss= tf.reduce_mean(self.o_gamma*tf.abs(Overlapped_G_1 - Overlapped_G_2))

        self.converge = tf.reduce_mean(self.o_gamma*tf.abs(Overlapped_G_1 - Overlapped_G_2))


        d_1_optim = d_1_optimizer.minimize(self.d_1_loss, var_list=self.D_1_var)
        g_1_optim = g_1_optimizer.minimize(self.g_1_loss, global_step=self.step, var_list=self.G_1_var)
        d_2_optim = d_2_optimizer.minimize(self.d_2_loss, var_list=self.D_2_var)
        g_2_optim = g_2_optimizer.minimize(self.g_2_loss, global_step=self.step, var_list=self.G_2_var)
        o_1_optim = over_1_optimizer.minimize(self.over_1_loss, var_list=self.G_1_var)
       # o_2_optim = over_2_optimizer.minimize(self.over_2_loss, var_list=self.G_2_var)

        self.balance_1 = self.gamma * self.d_1_loss_real - self.g_1_loss
        self.measure_1 = self.d_1_loss_real + tf.abs(self.balance_1)
        self.balance_2 = self.gamma * self.d_2_loss_real - self.g_2_loss
        self.measure_2 = self.d_2_loss_real + tf.abs(self.balance_2)


        with tf.control_dependencies([d_1_optim, g_1_optim,o_1_optim ]):
            self.k_1_update = tf.assign(
                self.k_t_1, tf.clip_by_value(self.k_t_1 + self.lambda_k * self.balance_1, 0, 1))
        with tf.control_dependencies([d_2_optim, g_2_optim ]):
            self.k_2_update = tf.assign(
                self.k_t_2, tf.clip_by_value(self.k_t_2 + self.lambda_k * self.balance_2, 0, 1))

        self.summary_op = tf.summary.merge([
            tf.summary.image("G_1", self.G_1),
            tf.summary.image("G_2", self.G_2),
            tf.summary.image("AE_G_1", self.AE_G_1),
            tf.summary.image("AE_G_2", self.AE_G_2),
            tf.summary.image("AE_x_1", self.AE_x_1),
            tf.summary.image("AE_x_2", self.AE_x_2),

            tf.summary.scalar("loss/d_1_loss", self.d_1_loss),
            tf.summary.scalar("loss/d_2_loss", self.d_2_loss),
            tf.summary.scalar("loss/d_1_loss_real", self.d_1_loss_real),
            tf.summary.scalar("loss/d_2_loss_real", self.d_2_loss_real),
            tf.summary.scalar("loss/d_1_loss_fake", self.d_1_loss_fake),
            tf.summary.scalar("loss/d_2_loss_fake", self.d_2_loss_fake),
            tf.summary.scalar("loss/g_1_loss", self.g_1_loss),
            tf.summary.scalar("loss/g_2_loss", self.g_2_loss),
            #tf.summary.scalar("loss/dis_1_loss", self.dis_1_loss),
            #tf.summary.scalar("loss/dis_2_loss", self.dis_2_loss),
            tf.summary.scalar("misc/measure_1", self.measure_1),
            tf.summary.scalar("misc/measure_2", self.measure_2),
            tf.summary.scalar("misc/k_t_1", self.k_t_1),
            tf.summary.scalar("misc/k_t_2", self.k_t_2),
            tf.summary.scalar("misc/d_lr_1", self.d_lr_1),
            tf.summary.scalar("misc/d_lr_2", self.d_lr_2),
            tf.summary.scalar("misc/g_lr_1", self.g_lr_1),
            tf.summary.scalar("misc/g_lr_2", self.g_lr_2),
            tf.summary.scalar("misc/balance_1", self.balance_1),
            tf.summary.scalar("misc/balance_2", self.balance_2),

            #tf.summary.scalar("misc/c_lr", self.c_lr),#revised
        ])

    def build_test_model(self):
        with tf.variable_scope("test") as vs:
            # Extra ops for interpolation
            z_optimizer = tf.train.AdamOptimizer(0.0001)

            self.z_r = tf.get_variable("z_r", [self.batch_size, self.z_num], tf.float32)
            self.z_r_update = tf.assign(self.z_r, self.z)

        G_z_r_1, _ = GeneratorCNN_1(
                self.z_r, self.conv_hidden_num, self.channel, self.repeat_num, self.data_format, reuse=True)
        G_z_r_2, _ = GeneratorCNN_2(
                self.z_r, self.conv_hidden_num, self.channel, self.repeat_num, self.data_format, reuse=True)
        
        with tf.variable_scope("test") as vs:
            self.z_r_loss = tf.reduce_mean(tf.abs(self.x_1 - G_z_r_1)+tf.abs(self.x_2-G_z_r_2))
            self.z_r_optim = z_optimizer.minimize(self.z_r_loss, var_list=[self.z_r])

        test_variables = tf.contrib.framework.get_variables(vs)
        self.sess.run(tf.variables_initializer(test_variables))

    def generate(self, inputs, root_path=None, path=None, idx=None, save=True):
        x_1 = self.sess.run(self.G_1, {self.z: inputs})
        tf.reset_default_graph()
        x_2 = self.sess.run(self.G_2, {self.z: inputs})
        Overlapped_G_1= x_1[:,32:64, :64,:]
        Overlapped_G_2= x_2[:,:32,   :64,:]
        average_overlapped=(Overlapped_G_1+Overlapped_G_2)/2
        x_1=np.pad(x_1,((0,0),(0,32) ,(0,0),(0,0)),'constant', constant_values=0)
        x_2=np.pad(x_2,((0,0),(32,0) ,(0,0),(0,0)),'constant', constant_values=0)
        average_overlapped=np.pad(average_overlapped,((0,0),(32,32) ,(0,0),(0,0)),'constant', constant_values=0)
        x=x_2+x_1-average_overlapped
        if path is None and save:
            path = os.path.join(root_path, '{}_G.png'.format(idx))
            save_image(x, path)
            print("[*] Samples saved: {}".format(path))
        return x

    def autoencode(self, inputs, path, idx=None, x_fake=None):
        items = {
            'real': inputs,
            'fake': x_fake,
        }
        for key, img in items.items():
            if img is None:
                continue
            #if img.shape[3] in [1, 3]:
            #   img = img.transpose([0, 3, 1, 2])
            x_path = os.path.join(path, '{}_D_{}.png'.format(idx, key))
            x_1 = self.sess.run(self.AE_x_1, {self.x_1: img[:,:64,:64,:]})
            tf.reset_default_graph()
            x_2 = self.sess.run(self.AE_x_2, {self.x_2: img[:,32:96,:64,:]})
            Overlapped_G_1= x_1[:,32:64, :64,:]
            Overlapped_G_2= x_2[:,:32,   :64,:]
            average_overlapped=(Overlapped_G_1+Overlapped_G_2)/2
            x_1=np.pad(x_1,((0,0),(0,32) ,(0,0),(0,0)),'constant', constant_values=0)
            x_2=np.pad(x_2,((0,0),(32,0) ,(0,0),(0,0)),'constant', constant_values=0)
            average_overlapped=np.pad(average_overlapped,((0,0),(32,32) ,(0,0),(0,0)),'constant', constant_values=0)
            x=x_2+x_1-average_overlapped           
            save_image(x, x_path)
            print("[*] Samples saved: {}".format(x_path))

    def encode(self, inputs):
        #if inputs.shape[3] in [1, 3]:
        #    inputs = inputs.transpose([0, 3, 1, 2])
        return self.sess.run(self.D_1_z, {self.x_1: inputs})

    def decode(self, z):
        return self.sess.run(self.AE_x, {self.D_z: z})

    def interpolate_G(self, real_batch, step=0, root_path='.', train_epoch=0):
        batch_size = len(real_batch)
        half_batch_size = int(batch_size)

        self.sess.run(self.z_r_update)
        tf_real_batch_o= to_nchw_numpy(real_batch)
        tf_real_batch_f=np.fliplr(tf_real_batch)
        tf_real_batch=np.concatenate([tf_real_batch_o,tf_real_batch_f])
        for i in trange(train_epoch):
            z_r_loss, _ = self.sess.run([self.z_r_loss, self.z_r_optim], {self.x: tf_real_batch})
        z = self.sess.run(self.z_r)

        z1, z2 = z[:half_batch_size], z[half_batch_size:]
        real1_batch, real2_batch = real_batch[:half_batch_size], real_batch[half_batch_size:]

        generated = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(z1, z2)])
            z_decode = self.generate(z, save=False)
            generated.append(z_decode)

        generated = np.stack(generated).transpose([1, 0, 2, 3, 4])
        for idx, img in enumerate(generated):
            save_image(img, os.path.join(root_path, 'test{}_interp_G_{}.png'.format(step, idx)), nrow=10)

        all_img_num = np.prod(generated.shape[:2])
        batch_generated = np.reshape(generated, [all_img_num] + list(generated.shape[2:]))
        save_image(batch_generated, os.path.join(root_path, 'test{}_interp_G.png'.format(step)), nrow=10)

    def interpolate_D(self, real1_batch, real2_batch, step=0, root_path="."):
        real1_encode = self.encode(real1_batch)
        real2_encode = self.encode(real2_batch)

        decodes = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(real1_encode, real2_encode)])
            z_decode = self.decode(z)
            decodes.append(z_decode)

        decodes = np.stack(decodes).transpose([1, 0, 2, 3, 4])
        for idx, img in enumerate(decodes):
            img = np.concatenate([[real1_batch[idx]], img, [real2_batch[idx]]], 0)
            save_image(img, os.path.join(root_path, 'test{}_interp_D_{}.png'.format(step, idx)), nrow=10 + 2)

    def test(self):
        root_path = "./"#self.model_dir

        all_G_z = None
        for step in range(3):
            real1_batch = self.get_image_from_loader()
            real2_batch = self.get_image_from_loader()

            save_image(real1_batch, os.path.join(root_path, 'test{}_real1.png'.format(step)))
            save_image(real2_batch, os.path.join(root_path, 'test{}_real2.png'.format(step)))

            self.autoencode(
                    real1_batch, self.model_dir, idx=os.path.join(root_path, "test{}_real1".format(step)))
            self.autoencode(
                    real2_batch, self.model_dir, idx=os.path.join(root_path, "test{}_real2".format(step)))

            self.interpolate_G(real1_batch, step, root_path)
            #self.interpolate_D(real1_batch, real2_batch, step, root_path)

            z_fixed = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))
            G_z = self.generate(z_fixed, path=os.path.join(root_path, "test{}_G_z.png".format(step)))

            if all_G_z is None:
                all_G_z = G_z
            else:
                all_G_z = np.concatenate([all_G_z, G_z])
            save_image(all_G_z, '{}/G_z{}.png'.format(root_path, step))

        save_image(all_G_z, '{}/all_G_z.png'.format(root_path), nrow=16)

    def get_image_from_loader(self):
        x = self.data_loader.eval(session=self.sess)
        if self.data_format == 'NCHW':
            x = x.transpose([0, 2, 3, 1])
        return x
