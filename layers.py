# 参考的代码主要是cycleGAN和ZM-Net
# 参考了beautyGAN在判别网络中的设计，在判别网络中引入了谱归一化，可以使判别器训练的过程更为平稳

import tensorflow as tf

def instance_norm(x,name="instance_norm",gamma=None,beta=None):
    """
    实例标准化，PNet中进行实例标准化时不输入gammas和betas，在网络结构中调用固定名称的的gammas和betas，进行迭代优化
    Tnet进行动态实例标准化时，输入由PNet学习得到gammas和betas
    在网络优化过程中也会进行梯度下降
    :param x: 输入张量
    :param name: namespace
    :param gamma: gamma
    :param beta: beta
    :return:
    """
    with tf.variable_scope(name):
        epsilon = 1e-5
        if gamma is None:
            gamma=tf.get_variable(name="gamma",shape=x.shape[-1],initializer=tf.truncated_normal_initializer(mean=1.0,stddev=0.02))
        if beta is None:
            beta = tf.get_variable(name="beta",shape=x.shape[-1],initializer=tf.constant_initializer(0.0))
        mean,var = tf.nn.moments(x,[1,2],keep_dims=True)
        out =gamma*tf.divide(x-mean,tf.sqrt(var+epsilon))+beta
        return out


def lrelu(x,leak=0.2,name="lrelu",alt_relu_impl=True):
    """
    :param x: 输入张量
    :param leak: 小于0时的斜率
    :param name: namespace
    :return:
    """
    with tf.variable_scope(name):
        if alt_relu_impl:
            f1 = 0.5*(1+leak)
            f2 = 0.5*(1-leak)
            return f1*x+f2*abs(x)
        else:
            return tf.maximum(x,leak*x)


def spectral_norm(x, iteration=1):
    """
    进行谱归一化，具体使用的是幂迭代法，参考的是
    https://github.com/taki0112/Spectral_Normalization-Tensorflow
    中的算法
    :param x: 输入的张量
    :param iteration: 幂迭代法的迭代次数
    :return: 谱归一化后的中间层
    """
    with tf.variable_scope("spectral_norm"):
        x_shape = x.shape.as_list()
        w = tf.reshape(x, [-1, x_shape[-1]])
        u = tf.get_variable("u", [1, x_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
        u_hat = u
        v_hat = None

        for i in range(iteration):
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = tf.nn.l2_normalize(v_, dim=None)
            u_ = tf.matmul(v_hat, w)
            u_hat = tf.nn.l2_normalize(u_, dim=None)
        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = w / sigma
            w_norm = tf.reshape(w_norm, x_shape)
        return w_norm


def generate_conv(input_tensor,num_outputs,kernel_size,stride,padding="SAME",name="conv",
                  stddev=0.02,do_norm=False,norm_gamma=None,norm_beta=None,do_spec=False,do_relu=False,relufactor=0):
    """
    生成卷积层
    :param input_tensor: 输入张量
    :param num_outputs: 输出通道数
    :param kernel_size: 卷积核大小
    :param stride: 步长
    :param padding: padding方式
    :param name: namesapce
    :param stddev: 初始化参数时的标准差
    :param do_norm: 是否进行实例标准化
    :param norm_gamma: 动态实例标准化的gamma
    :param norm_beta: 动态实例标准化的beta
    :param do_spec: 是否进行谱归一化
    :param do_relu: 是否使用relu/leaky-relu激活函数
    :param relufactor: leaky-relu中x小于0时的斜率
    :return: 卷积层
    """
    with tf.variable_scope(name):
        conv = tf.contrib.layers.conv2d(
            inputs=input_tensor,
            num_outputs = num_outputs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            activation_fn=None,
            weights_initializer = tf.truncated_normal_initializer(stddev=stddev),
            biases_initializer = tf.constant_initializer(0.0),
        )
        if do_norm:
            conv = instance_norm(conv,gamma=norm_gamma,beta=norm_beta)
        else:
            if do_spec:
                conv = spectral_norm(conv)

        if do_relu:
            if relufactor==0:
                conv = tf.nn.relu(conv,name="relu")
            else:
                conv = lrelu(conv,leak=relufactor,name="lrelu")
        return conv


def generate_deconv(input_tensor,num_outputs,kernel_size,stride,padding="SAME",name="deconv",
                    stddev=0.02,do_norm=False,do_spec=False,norm_gamma=None,norm_beta=None,do_relu=False,relufactor=0):
    """
    :param input_tensor: 输入张量
    :param num_outputs: 输出通道数
    :param kernel_size: 卷积核大小
    :param stride: 步长
    :param padding: padding方式
    :param name: namespace
    :param stddev: 初始化参数时的标准差
    :param do_norm: 是否进行实例标准化
    :param do_spec: 是否进行谱归一化
    :param norm_gamma: 动态实例标准化的gamma
    :param norm_beta: 动态实例标准化的beta
    :param do_relu: 是否使用relu/leaky-relu激活函数
    :param relufactor: leaky-relu中x小于0时的斜率
    :return: 反卷积层
    """
    with tf.variable_scope(name):
        deconv = tf.contrib.layers.conv2d_transpose(
            inputs=input_tensor,
            num_outputs=num_outputs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            biases_initializer=tf.constant_initializer(0.0),
        )
        if do_norm:
            deconv = instance_norm(deconv,gamma=norm_gamma,beta=norm_beta)
        else:
            if do_spec:
                deconv = spectral_norm(deconv)

        if do_relu:
            if relufactor==0:
                deconv = tf.nn.relu(deconv,name="relu")
            else:
                deconv = lrelu(deconv,leak=relufactor,name="lrelu")

        return deconv


def generate_resblock(input_res,dim,name="resnet",gamma1_norm=None,beta1_norm=None,
                      gamma2_norm=None,beta2_norm=None):
    """
    生成残差网络中每个block的中间层，具体的，每个残差block中有两个卷积层
    :param input_res: 输入张量
    :param dim: 输出通道数
    :param name: namespace
    :param gamma1_norm: 第一个卷积层进行动态实例标准化的gammas
    :param beta1_norm: 第一个卷积层进行动态实例标准化的betas
    :param gamma2_norm: 第二个卷积层进行动态实例标准化的gammas
    :param beta2_norm: 第二个卷积层进行动态实例标准化的betas
    :return:
    """
    with tf.variable_scope(name):
        out_res = tf.pad(input_res,[[0,0],[1,1],[1,1],[0,0]],"REFLECT")
        out_res = generate_conv(out_res,num_outputs=dim,kernel_size=3,stride=1,padding="VALID",name="c1",
                                do_norm=True,do_relu=True,norm_gamma=gamma1_norm,norm_beta=beta1_norm)
        out_res = tf.pad(out_res,[[0,0],[1,1],[1,1],[0,0]],"REFLECT")
        out_res = generate_conv(out_res, num_outputs=dim, kernel_size=3, stride=1, padding="VALID", name="c2",
                                do_norm=True,norm_gamma=gamma2_norm,norm_beta=beta2_norm)
        return tf.nn.relu(input_res+out_res)


def generate_pnet_resblock(input_tensor, dim, name="resnet"):
    """
    在Pnet中，无需进行动态实例标准化，生成不进行实例动态标准化的残差block
    :param input_tensor: 输入张量
    :param dim: 输出通道数
    :param name: namespace
    :return:
    """
    with tf.variable_scope(name):
        out_res = tf.pad(input_tensor, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        conv1 = generate_conv(out_res, num_outputs=dim, kernel_size=3, stride=1, padding="VALID", name="c1",
                              do_norm=True, do_relu=True)
        out_res = tf.pad(conv1, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        conv2 = generate_conv(out_res, num_outputs=dim, kernel_size=3, stride=1, padding="VALID", name="c2",
                              do_norm=True)
        return tf.nn.relu(input_tensor + conv2), conv1, conv2


def pnet_fc(input_tensor,name):
    """
    将输入的张量通过一个全连接层，生成对应的gammas或betas
    :param input_tensor: 输入的张量
    :param name: namespace
    :return: 全连接层后的结果
    """
    shape = input_tensor.shape.as_list()
    with tf.variable_scope(name):
        input_tensor = tf.reshape(input_tensor,shape=[-1,shape[3]])  # n^2*c
        shape = input_tensor.shape.as_list()
        w = tf.get_variable(name="w",shape=[1,shape[0]])
        b = tf.get_variable(name="b",shape=[1,shape[1]])
        fc = tf.squeeze(tf.add(tf.matmul(w,input_tensor),b))
        return fc


