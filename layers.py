import tensorflow as tf

def instance_norm(x,name="instance_norm",gamma=None,beta=None):
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
    with tf.variable_scope(name):
        if alt_relu_impl:
            f1 = 0.5*(1+leak)
            f2 = 0.5*(1-leak)
            return f1*x+f2*abs(x)
        else:
            return tf.maximum(x,leak*x)


def generate_conv(input_tensor,num_outputs,kernel_size,stride,padding="SAME",name="conv",
                  stddev=0.02,do_norm=False,norm_gamma=None,norm_beta=None,do_spec=False,do_relu=False,relufactor=0):
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
                pass

        if do_relu:
            if relufactor==0:
                conv = tf.nn.relu(conv,name="relu")
            else:
                conv = lrelu(conv,leak=relufactor,name="lrelu")
        return conv


def generate_deconv(input_tensor,num_outputs,kernel_size,stride,padding="SAME",name="deconv",
                    stddev=0.02,do_norm=False,do_spec=False,norm_gamma=None,norm_beta=None,do_relu=False,relufactor=0):
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
                pass

        if do_relu:
            if relufactor==0:
                deconv = tf.nn.relu(deconv,name="relu")
            else:
                deconv = lrelu(deconv,leak=relufactor,name="lrelu")

        return deconv


def generate_resblock(input_res,dim,name="resnet",gamma1_norm=None,beta1_norm=None,
                      gamma2_norm=None,beta2_norm=None):
    with tf.variable_scope(name):
        out_res = tf.pad(input_res,[[0,0],[1,1],[1,1],[0,0]],"REFLECT")
        out_res = generate_conv(out_res,num_outputs=dim,kernel_size=3,stride=1,padding="VALID",name="c1",
                                do_norm=True,do_relu=True,norm_gamma=gamma1_norm,norm_beta=beta1_norm)
        out_res = tf.pad(out_res,[[0,0],[1,1],[1,1],[0,0]],"REFLECT")
        out_res = generate_conv(out_res, num_outputs=dim, kernel_size=3, stride=1, padding="VALID", name="c2",
                                do_norm=True,norm_gamma=gamma2_norm,norm_beta=beta2_norm)
        return tf.nn.relu(input_res+out_res)


def pnet_fc(input_tensor,name):
    """
    :param input_tensor:1*n*n*c
    :param name:
    :return:
    """
    shape = input_tensor.shape.as_list()
    with tf.variable_scope(name):
        input_tensor = tf.reshape(input_tensor,shape=[-1,shape[3]])  # n^2*c
        shape = input_tensor.shape.as_list()
        w = tf.get_variable(name="w",shape=[1,shape[0]])
        b = tf.get_variable(name="b",shape=[1,shape[1]])
        fc = tf.squeeze(tf.add(tf.matmul(w,input_tensor),b))
        return fc


def generate_pnet_resblock(input_tensor,dim,name="resnet"):
    with tf.variable_scope(name):
        out_res = tf.pad(input_tensor,[[0,0],[1,1],[1,1],[0,0]],"REFLECT")
        conv1 = generate_conv(out_res,num_outputs=dim,kernel_size=3,stride=1,padding="VALID",name="c1",
                              do_norm=True,do_relu=True)
        out_res = tf.pad(conv1,[[0,0],[1,1],[1,1],[0,0]],"REFLECT")
        # should not be relu
        conv2 = generate_conv(out_res, num_outputs=dim, kernel_size=3, stride=1, padding="VALID", name="c2",
                              do_norm=True)
        return tf.nn.relu(input_tensor+conv2), conv1, conv2