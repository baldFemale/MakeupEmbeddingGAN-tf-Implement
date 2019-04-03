import tensorflow as tf

from layers import *


ngf = 32
ndf = 64


def Pnet(input_B,name="Pnet"):
    """
    :param input_B: 1*256*256*3
    :param name:
    :return: gammas and betas
    """
    gammas = {}
    betas = {}
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        ks = 3
        fs = 7

        input_pad_B = tf.pad(input_B,[[0,0],[3,3],[3,3],[0,0]],"REFLECT")
        c1 = generate_conv(input_pad_B,num_outputs=ngf,kernel_size=fs,stride=1,padding="VALID",name="c1",do_norm=True,do_relu=True)
        gammas["c1"], betas["c1"] = pnet_fc(c1,"fc_gamma_c1"),pnet_fc(c1,"fc_beta_c1")
        c2 = generate_conv(c1,num_outputs=ngf*2,kernel_size=ks,stride=2,padding="SAME",name="c2",do_norm=True,do_relu=True)
        gammas["c2"], betas["c2"] = pnet_fc(c2,"fc_gamma_c2"),pnet_fc(c2,"fc_beta_c2")
        c3 = generate_conv(c2,num_outputs=ngf*4,kernel_size=ks,stride=2,padding="SAME",name="c3",do_norm=True,do_relu=True)
        gammas["c3"], betas["c3"] = pnet_fc(c3,"fc_gamma_c3"),pnet_fc(c3,"fc_beta_c3")

        r1,r1_1,r1_2 = generate_pnet_resblock(c3,dim=ngf*4,name="r1")
        gammas["r1_1"],betas["r1_1"] = pnet_fc(r1_1,"fc_gamma_r1_1"),pnet_fc(r1_1,"fc_beta_r1_1")
        gammas["r1_2"], betas["r1_2"] = pnet_fc(r1_2, "fc_gamma_r1_2"), pnet_fc(r1_2, "fc_beta_r1_2")
        r2,r2_1,r2_2 = generate_pnet_resblock(r1,dim=ngf*4,name="r2")
        gammas["r2_1"],betas["r2_1"] = pnet_fc(r2_1,"fc_gamma_r2_1"),pnet_fc(r2_1,"fc_beta_r2_1")
        gammas["r2_2"], betas["r2_2"] = pnet_fc(r2_2, "fc_gamma_r2_2"), pnet_fc(r2_2, "fc_beta_r2_2")
        r3,r3_1,r3_2 = generate_pnet_resblock(r2,dim=ngf*4,name="r3")
        gammas["r3_1"],betas["r3_1"] = pnet_fc(r3_1,"fc_gamma_r3_1"),pnet_fc(r3_1,"fc_beta_r3_1")
        gammas["r3_2"], betas["r3_2"] = pnet_fc(r3_2, "fc_gamma_r3_2"), pnet_fc(r3_2, "fc_beta_r3_2")
        r4,r4_1,r4_2 = generate_pnet_resblock(r3,dim=ngf*4,name="r4")
        gammas["r4_1"],betas["r4_1"] = pnet_fc(r4_1,"fc_gamma_r4_1"),pnet_fc(r4_1,"fc_beta_r4_1")
        gammas["r4_2"], betas["r4_2"] = pnet_fc(r4_2, "fc_gamma_r4_2"), pnet_fc(r4_2, "fc_beta_r4_2")
        r5,r5_1,r5_2 = generate_pnet_resblock(r4,dim=ngf*4,name="r5")
        gammas["r5_1"],betas["r5_1"] = pnet_fc(r5_1,"fc_gamma_r5_1"),pnet_fc(r5_1,"fc_beta_r5_1")
        gammas["r5_2"], betas["r5_2"] = pnet_fc(r5_2, "fc_gamma_r5_2"), pnet_fc(r5_2, "fc_beta_r5_2")
        r6,r6_1,r6_2 = generate_pnet_resblock(r5,dim=ngf*4,name="r6")
        gammas["r6_1"],betas["r6_1"] = pnet_fc(r6_1,"fc_gamma_r6_1"),pnet_fc(r6_1,"fc_beta_r6_1")
        gammas["r6_2"], betas["r6_2"] = pnet_fc(r6_2, "fc_gamma_r6_2"), pnet_fc(r6_2, "fc_beta_r6_2")
        r7,r7_1,r7_2 = generate_pnet_resblock(r6,dim=ngf*4,name="r7")
        gammas["r7_1"],betas["r7_1"] = pnet_fc(r7_1,"fc_gamma_r7_1"),pnet_fc(r7_1,"fc_beta_r7_1")
        gammas["r7_2"], betas["r7_2"] = pnet_fc(r7_2, "fc_gamma_r7_2"), pnet_fc(r7_2, "fc_beta_r7_2")
        r8,r8_1,r8_2 = generate_pnet_resblock(r7,dim=ngf*4,name="r8")
        gammas["r8_1"],betas["r8_1"] = pnet_fc(r8_1,"fc_gamma_r8_1"),pnet_fc(r8_1,"fc_beta_r8_1")
        gammas["r8_2"], betas["r8_2"] = pnet_fc(r8_2, "fc_gamma_r8_2"), pnet_fc(r8_2, "fc_beta_r8_2")
        r9,r9_1,r9_2 = generate_pnet_resblock(r8,dim=ngf*4,name="r9")
        gammas["r9_1"],betas["r9_1"] = pnet_fc(r9_1,"fc_gamma_r9_1"),pnet_fc(r9_1,"fc_beta_r9_1")
        gammas["r9_2"], betas["r9_2"] = pnet_fc(r9_2, "fc_gamma_r9_2"), pnet_fc(r9_2, "fc_beta_r9_2")

        c4 = generate_deconv(r9,ngf*2,ks,stride=2,padding="SAME",name="c4",do_norm=True,do_relu=True)
        gammas["c4"], betas["c4"] = pnet_fc(c4,"fc_gamma_c4"),pnet_fc(c4,"fc_beta_c4")
        c5 = generate_deconv(c4,ngf,ks,stride=2,padding="SAME",name="c5",do_norm=True,do_relu=True)
        gammas["c5"], betas["c5"] = pnet_fc(c5,"fc_gamma_c5"),pnet_fc(c5,"fc_beta_c5")
        return gammas,betas


def Tnet(input_A,gammas,betas,name="Tnet"):
    """
    :param input_A: 1*256*256*3
    :param name:
    :return: 1*256*256*3
    """
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        ks = 3
        fs =7

        input_pad_A = tf.pad(input_A,[[0,0],[3,3],[3,3],[0,0]],"REFLECT")  # 1*262*262*3
        c1 =generate_conv(input_pad_A,num_outputs=ngf,kernel_size=fs,stride=1,padding="VALID",name="c1",
                          do_norm=True, norm_gamma=gammas["c1"],norm_beta=betas["c1"],do_relu=True)  # 1*256*256*32
        c2 = generate_conv(c1, num_outputs=ngf*2, kernel_size=ks, stride=2, padding="SAME", name="c2",
                           do_norm=True, norm_gamma=gammas["c2"],norm_beta=betas["c2"],do_relu=True)  # 1*128*128*64
        c3 = generate_conv(c2, num_outputs=ngf*4, kernel_size=ks, stride=2, padding="SAME", name="c3",
                           do_norm=True,norm_gamma=gammas["c3"],norm_beta=betas["c3"],do_relu=True)  # 1*64*64*128

        o_r1 = generate_resblock(c3,dim=ngf*4,name="o_r1",gamma1_norm=gammas["r1_1"],gamma2_norm=gammas["r1_2"],
                                 beta1_norm=betas["r1_1"],beta2_norm=betas["r1_2"])
        o_r2 = generate_resblock(o_r1, dim=ngf * 4, name="o_r2",gamma1_norm=gammas["r2_1"],gamma2_norm=gammas["r2_2"],
                                 beta1_norm=betas["r2_1"],beta2_norm=betas["r2_2"])
        o_r3 = generate_resblock(o_r2, dim=ngf * 4, name="o_r3",gamma1_norm=gammas["r3_1"],gamma2_norm=gammas["r3_2"],
                                 beta1_norm=betas["r3_1"],beta2_norm=betas["r3_2"])
        o_r4 = generate_resblock(o_r3, dim=ngf * 4, name="o_r4",gamma1_norm=gammas["r4_1"],gamma2_norm=gammas["r4_2"],
                                 beta1_norm=betas["r4_1"],beta2_norm=betas["r4_2"])
        o_r5 = generate_resblock(o_r4, dim=ngf * 4, name="o_r5",gamma1_norm=gammas["r5_1"],gamma2_norm=gammas["r5_2"],
                                 beta1_norm=betas["r5_1"],beta2_norm=betas["r5_2"])
        o_r6 = generate_resblock(o_r5, dim=ngf * 4, name="o_r6",gamma1_norm=gammas["r6_1"],gamma2_norm=gammas["r6_2"],
                                 beta1_norm=betas["r6_1"],beta2_norm=betas["r6_2"])
        o_r7 = generate_resblock(o_r6, dim=ngf * 4, name="o_r7",gamma1_norm=gammas["r7_1"],gamma2_norm=gammas["r7_2"],
                                 beta1_norm=betas["r7_1"],beta2_norm=betas["r7_2"])
        o_r8 = generate_resblock(o_r7, dim=ngf * 4, name="o_r8",gamma1_norm=gammas["r8_1"],gamma2_norm=gammas["r8_2"],
                                 beta1_norm=betas["r8_1"],beta2_norm=betas["r8_2"])
        o_r9 = generate_resblock(o_r8, dim=ngf * 4, name="o_r9",gamma1_norm=gammas["r9_1"],gamma2_norm=gammas["r9_2"],
                                 beta1_norm=betas["r9_1"],beta2_norm=betas["r9_2"])  # 1*64*64*128

        c4 = generate_deconv(o_r9, num_outputs=ngf * 2, kernel_size=ks, stride=2, padding="SAME", name="c4",
                             do_norm=True,norm_gamma=gammas["c4"],norm_beta=betas["c4"],do_relu=True)  # 1*128*128*64
        c5 = generate_deconv(c4, num_outputs=ngf, kernel_size=ks, stride=2, padding="SAME", name="c5",
                             do_norm=True,norm_gamma=gammas["c5"],norm_beta=betas["c5"],do_relu=True) #  1*256*256*32
        c6 = generate_deconv(c5, num_outputs=3, kernel_size=fs, stride=1, padding="SAME", name="c6")

        out_gen = tf.nn.tanh(c6,"t1")
        return out_gen


def generate_discriminator(inputdis,name="discriminator"):
    with tf.variable_scope(name):
        f = 4
        o_c1 = generate_conv(inputdis,num_outputs=ndf,kernel_size=f,stride=2,padding="SAME",name="c1",
                             do_relu=True,relufactor=0.2)  # 1*128*128*64
        o_c2 = generate_conv(o_c1,num_outputs=ndf*2,kernel_size=f,stride=2,padding="SAME",name="c2",
                             do_spec=True,do_relu=True,relufactor=0.2)  # 1*64*64*128
        o_c3 = generate_conv(o_c2, num_outputs=ndf * 4, kernel_size=f, stride=2, padding="SAME", name="c3",
                             do_spec=True, do_relu=True, relufactor=0.2)  # 1*32*32*256
        o_c4 = generate_conv(o_c3, num_outputs=ndf * 8, kernel_size=f, stride=1, padding="SAME", name="c4",
                             do_spec=True, do_relu=True, relufactor=0.2)  # 1*32*32*512
        o_c5 = generate_conv(o_c4, num_outputs=1, kernel_size=f, stride=1, padding="SAME", name="c5")  # 1*32*32*1
        return o_c5
