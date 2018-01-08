import re
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from cairosvg import svg2png
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import tensorflow as tf
import sys
import scipy
from keras.preprocessing.image import transform_matrix_offset_center


g_lr = 1e-6
g_l2 = 1e-20
g_batch_size = 8

"""
    从环境变量获取参数
    :param  g_lr          学习率
    :param  g_l2          l2 正则化常数
    :param  g_batch_size  批次大小
"""

if len(sys.argv) == 4:
    print("Using argv %s" % (" ".join(sys.argv)))
    g_lr = float(sys.argv[1])
    g_l2 = float(sys.argv[2])
    g_batch_size = int(sys.argv[3])


def get_image(path, shape):
    """
    使用 opencv 读取图像
    :param  path  输入图像路径
    :param  shape 输出图像大小
    :return image 以[H,W,C]的RGB矩阵形式，输出图像
    """
    image = cv2.imread(path)
    image = image[:,:,::-1]
    if shape != None:
        image = cv2.resize(image, shape)
    return image

def svg_process(svg_file, shape):
    """
    将癌细胞区域标注的矢量图 svg 格式的文件标注，转换成矩阵，并读入
    :param  svg_file  输入矢量图图像路径
    :param  shape     输出图像大小
    :return x         以灰度矩阵形式，输出癌细胞区域标注的结果
    """
    image_dir    = os.path.dirname(svg_file)
    image_prefix = os.path.basename(svg_file).split(".svg")[0]
    if not os.path.isfile("%s/%s.png" % (image_dir, image_prefix)):
        with open(svg_file, "r") as f_in:
            svg_code = f_in.readlines()
            svg_code = "".join(svg_code[1:])
            svg_code = re.sub(r'''fill="(\w+|None)"''', '''fill="#FFFFFF"''', svg_code)
            svg_code = re.sub(r'''stroke="(\w+|None)"''', '''stroke="#FFFFFF"''', svg_code)
            svg2png(bytestring=svg_code, write_to="%s/%s.png" % (image_dir, image_prefix))

    img = get_image("%s/%s.png" % (image_dir, image_prefix), shape)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='constant',
                    cval=0.):
    """
    进行图像旋转。简化改写了：
       https://github.com/fchollet/keras/blob/2.0.4/keras/preprocessing/image.py
    
    :param  x                输入需要旋转的图像向量
    :param  transform_matrix 图像旋转矩阵参数
    :param  channel_axis     哪一个维度代表图像的编号。对 [N,H,W,C]，N是图像编号，所以是0
    :param  fill_mode        填充由于旋转造成的边缘空白的方式。
                             可选 {'constant', 'nearest', 'reflect', 'wrap'}
    :param  cval             如果是'constant'填充，则在空白处填写什么内容
    
    :return x                旋转后的图像
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    if cval is False:
        cha_img1 = scipy.ndimage.interpolation.affine_transform(
            x[0], final_affine_matrix, final_offset, order=0, mode=fill_mode, cval=False)
        cha_img2 = scipy.ndimage.interpolation.affine_transform(
            x[1], final_affine_matrix, final_offset, order=0, mode=fill_mode, cval=True)
        channel_images = [cha_img1, cha_img2]
    else:
        channel_images = [scipy.ndimage.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=0,
            mode=fill_mode,
            cval=cval) for x_channel in x]

    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def picture_argument(image_input, image_gt, rotate, zoom):
    """
    对输入的病理切片，进行旋转、缩放操作的图像增强，并生成经过同样旋转、缩放操作的标注区域
    
    :param  image_input           输入的病理切片图像
    :param  image_gt              输入的病例切片癌症标注区域图像
    :param  rotate                对图像进行旋转的正负角度范围
    :param  zoom                  对图像进行缩放操作的放大缩小百分比
    :return image_input，image_gt 旋转、缩放后的病理切片图像，以及对应的癌症区域标注
    """
    theta = np.pi / 180 * np.random.uniform(-rotate, rotate)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    transform_matrix = rotation_matrix
    zx, zy = np.random.uniform(1-zoom, 1+zoom, 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    transform_matrix = np.dot(transform_matrix, zoom_matrix)

    h, w = image_input.shape[0], image_input.shape[1]
    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
    image_input = apply_transform(image_input, transform_matrix, 2,
                                fill_mode="constant", cval=255)

    image_gt = apply_transform(image_gt, transform_matrix, 2,
                                fill_mode="constant", cval=False)
    return image_input, image_gt

def gen_batch_func(l_sample, image_shape):
    """
    生成器函数，输入样本名称，每调用一次生成器，输出若干张病例切片图片以及对应的癌症区域标注图片
    
    :param  l_sample      输入的病理切片图像样本名称，
                          对应的病理切片图像放在 ./FCN/image/merge 目录下
                          对应的病理切片癌症区域标注放在 ./FCN/labels 目录下
    :param  image_shape   输出图片的大小
    :return batch_gen     输出生成器，每调用一次生成器，输出若干张病例切片图片以及对应的癌症区域标注图片
    """
    def batch_gen(batch_size, augmentation=True):
        random.shuffle(l_sample)
        for batch_i in range(0, len(l_sample), batch_size):
            l_images = []
            l_gt_images = []
            for sample in l_sample[batch_i:batch_i+batch_size]:
                gt_image_file = "./FCN/labels/%s.svg" % (sample)
                image_file = "./FCN/image/merge/%s.tiff" % (sample)
                if os.path.isfile(gt_image_file):
                    gt_image_raw = svg_process(gt_image_file, shape=image_shape)

                else:
                    gt_image_raw = np.zeros([image_shape[0], image_shape[1]])

                image = get_image(image_file, shape=image_shape)
                gt_image = gt_image_raw>100
                gt_image = gt_image.reshape(*gt_image.shape, 1)
                gt_image2 = np.bitwise_not(gt_image)
                gt_out = np.concatenate((gt_image, gt_image2), axis=2)

                if augmentation:
                    rotation = 180
                    zoom = 0.2
                    image,gt_out = picture_argument(image, gt_out, rotation, zoom)

                l_images.append(image)
                l_gt_images.append(gt_out)

            yield np.array(l_images), np.array(l_gt_images)

    return batch_gen


def load_vgg(sess, vgg_path):
    """
    载入 VGG16 预训练模型，返回我们基于 VGG16 训练全卷积神经网络(FCN)所必须的中间变量。
    :param  sess:     TensorFlow Session
    :param  vgg_path: vgg16 模型文件的下载路径。模型使用pb格式存储，
                     下载地址：https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip
    
    :return image_input, keep_prob, layer3_out, layer4_out, layer7_out
                     返回我们基于 VGG16 训练 全卷积神经网络(FCN) 所必须的中间变量
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    基于 load_vgg 返回的VGG16模型中间结果，设计全卷积神经网络(FCN)模型
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: 需要分类的种类数目。这里是肿瘤区域/非肿瘤区域的二分类
    :return: 全卷积神经网络模型（FCN）的输出结果
    """
    with tf.name_scope("32xUpsampled") as scope:
        conv7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1,
                                        padding='same', name="32x_1x1_conv",
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(g_l2),
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv7_2x  = tf.layers.conv2d_transpose(conv7_1x1, num_classes, 4,
                                        strides=2, padding='same', name="32x_conv_trans_upsample",
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(g_l2),
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope("16xUpsampled") as scope:
        conv4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1,
                                        padding='same', name="16x_1x1_conv",
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(g_l2),
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv_merge1 = tf.add(conv4_1x1, conv7_2x, name="16x_combined_with_skip")
        conv4_2x  = tf.layers.conv2d_transpose(conv_merge1, num_classes, 4,
                                        strides=2, padding='same', name="16x_conv_trans_upsample",
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(g_l2),
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope("8xUpsampled") as scope:
        conv3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1,
                                        padding='same', name="8x_1x1_conv",
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(g_l2),
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv_merge2 = tf.add(conv3_1x1, conv4_2x, name="8x_combined_with_skip")
        conv3_8x  = tf.layers.conv2d_transpose(conv_merge2, num_classes, 16,
                                        strides=8, padding='same', name="8x_conv_trans_upsample",
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(g_l2),
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())

    conv_image_0 = tf.slice(conv3_8x, [0,0,0,0], [-1,-1,-1,1])
    tf.summary.image("conv3_8x_results_0", conv_image_0)
    return conv3_8x


def optimize(nn_last_layer, correct_label, learning_rate, num_classes, batch_size, split_idx):
    """
    定义模型的优化目标（损失函数），设置优化器
    :param  nn_last_layer:     全卷积神经网络模型（FCN）的输出结果
    :param  correct_label:     病理切片对应的、准确的癌症区域标注
    :param  learning_rate:     初始学习率大小
    :param  num_classes:       需要分类的种类数目。这里是癌症区域/非癌症区域的二分类
    :return pred_label:        病理切片对应的、模型预测的癌症区域标注
    :return training_op:       优化器
    :return cross_entropy_loss 交叉熵损失函数
    :return f1                 比赛规定的评价指标 f1 值
    :return learning_rate2     随训练次数逐步衰减后的学习率的大小
    """
    pred_label = tf.reshape(nn_last_layer, [-1, num_classes], name="predicted_label")
    true_label = tf.reshape(correct_label, [-1, num_classes], name="true_label")

    with tf.name_scope("f1_score"):
        argmax_p = tf.argmax(pred_label, 1)
        argmax_y = tf.argmax(true_label, 1)
        TP = tf.count_nonzero( argmax_p   * argmax_y,    dtype=tf.float32)
        TN = tf.count_nonzero((argmax_p-1)*(argmax_y-1), dtype=tf.float32)
        FP = tf.count_nonzero( argmax_p   *(argmax_y-1), dtype=tf.float32)
        FN = tf.count_nonzero((argmax_p-1)* argmax_y,    dtype=tf.float32)
        precision = TP / (TP+FP)
        recall    = TP / (TP+FN)
        f1 = 2 * precision * recall / (precision + recall)

    with tf.name_scope("cross_entropy_loss"):
        entropy_val = tf.nn.softmax_cross_entropy_with_logits(labels=true_label, logits=pred_label)
        cross_entropy_loss = tf.reduce_sum(entropy_val)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = cross_entropy_loss +  sum(reg_losses)

    with tf.name_scope("train"):
        batch = tf.Variable(0, tf.float32)
        learning_rate2 = tf.train.exponential_decay(
                      learning_rate,       # Base learning rate.
                      batch * batch_size,  # Current index into the dataset.
                      split_idx,           # Decay step.
                      0.95,                # Decay rate.
                      staircase=True)
        
        optimizer = tf.train.AdamOptimizer(learning_rate2)
        training_op = optimizer.minimize(loss, global_step=batch)

    return pred_label, training_op, cross_entropy_loss, f1, learning_rate2


def validation(nn_last_layer, correct_label, num_classes):
    """
    每当模型遍历所有训练样本（80%，560个）之后，对剩下20%的验证样本执行一次验证操作，检验模型在新样本上的表现
    :param  nn_last_layer:        全卷积神经网络模型（FCN）的输出结果
    :param  correct_label:        病理切片对应的、准确的癌症区域标注
    :param  num_classes:          需要分类的种类数目。这里是癌症区域/非癌症区域的二分类
    :return cross_entropy_loss_cv 交叉熵损失函数(验证样本)
    :return f1_cv                 比赛规定的评价指标 f1 值(验证样本)
    """
    pred_label = tf.reshape(nn_last_layer, [-1, num_classes], name="predicted_label_cv")
    true_label = tf.reshape(correct_label, [-1, num_classes], name="true_label_cv")
    

    with tf.name_scope("f1_score_cv"):
        argmax_p = tf.argmax(pred_label, 1)
        argmax_y = tf.argmax(true_label, 1)
        TP = tf.count_nonzero( argmax_p   * argmax_y,    dtype=tf.float32)
        TN = tf.count_nonzero((argmax_p-1)*(argmax_y-1), dtype=tf.float32)
        FP = tf.count_nonzero( argmax_p   *(argmax_y-1), dtype=tf.float32)
        FN = tf.count_nonzero((argmax_p-1)* argmax_y,    dtype=tf.float32)
        precision = TP / (TP+FP)
        recall    = TP / (TP+FN)
        f1_cv = 2 * precision * recall / (precision + recall)

    with tf.name_scope("cross_entropy_loss_cv"):
        entropy_val = tf.nn.softmax_cross_entropy_with_logits(labels=true_label, logits=pred_label)
        cross_entropy_loss_cv = tf.reduce_sum(entropy_val)


    return cross_entropy_loss_cv, f1_cv

def train_nn(sess, epochs, batch_size, get_batches_train, get_batches_cv, train_op, cross_entropy_loss,
             cross_entropy_loss_cv, f1, f1_cv,lr, input_image, correct_label, keep_prob, learning_rate):
    """
    汇总之前的结果，训练定义的全卷积神经网络
    :param sess:                   TF Session
    :param epochs:                 训练几轮数据
    :param batch_size:             批次大小
    :param get_batches_train:      获取训练数据的生成器，使用方法 gen_batch_func(batch_size)
    :param get_batches_cv:         获取验证数据的生成器，使用方法 gen_batch_func(batch_size)
    :param train_op:               训练模型的操作子，优化目标 cross_entropy_loss+l2_loss 最小化
    :param cross_entropy_loss:     交叉熵损失函数(训练样本)
    :param cross_entropy_loss_cv:  交叉熵损失函数(验证样本)
    :param f1:                     比赛规定的评价指标 f1 值(训练样本)
    :param f1_cv:                  比赛规定的评价指标 f1 值(验证样本)
    :param input_image:            模型输入图片大小
    :param correct_label:          病理切片对应的、准确的癌症区域标注
    :param keep_prob:              VGG 模型中间参数
    :param learning_rate:          初始化学习率大小
    
    """
    #save training results for every eproch
    saver = tf.train.Saver()
    model_dir = './aaausingNonAug_models_l2_norm_ExpDecay_lr_%1.2e_l2_%1.2e_e10_batch_%d' % (g_lr, g_l2, g_batch_size)
    log_dir   = "./aaausingNonAug_logs_l2_norm_ExpDecay_lr_%1.2e_l2_%1.2e_e10_batch_%d"  % (g_lr, g_l2, g_batch_size)
    cv_dir    = "./aaausingNonAug_cv_ExpDecay_lr_%1.2e_l2_%1.2e_e10_batch_%d.csv"  % (g_lr, g_l2, g_batch_size)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    f_out = open(cv_dir, "w")
    f_out.write("Eproch,cv_CrossEntropy_loss,cv_F1\n")

    summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

    sess.run(tf.global_variables_initializer())
    tf.summary.scalar("train_loss", cross_entropy_loss)
    tf.summary.scalar("train_f1", f1)
    merged_summary_op = tf.summary.merge_all()

    global_iteration_idx = 0
    for i in range(epochs):
        print("Epoch %d" % i)
        ii = 0
        for batch_image,batch_label in get_batches_train(batch_size, augmentation=True):
            ii += 1
            global_iteration_idx += 1
            train_op_,cross_entropy_loss_,summary_str,f1_,lr_ = sess.run(
                     [train_op, cross_entropy_loss, merged_summary_op, f1, lr],
                     feed_dict={
                        input_image: batch_image,
                        correct_label: batch_label,
                        learning_rate : g_lr,
                        keep_prob : 0.5
            })
            summary_writer.add_summary(summary_str, global_iteration_idx)
            print("Eproch %d, Iteration %d, loss = %1.5f, f1 = %1.5f, lr = %1.5f" % (i, ii, cross_entropy_loss_, f1_, lr_))

        # Save the model every eproch
        l_f1_cv = []
        l_loss_cv = []

        for batch_image,batch_label in get_batches_cv(batch_size, augmentation=False):
            cross_entropy_loss_, f1_ = sess.run(
                     [cross_entropy_loss_cv, f1_cv],
                     feed_dict={
                        input_image: batch_image,
                        correct_label: batch_label,
                        keep_prob : 0.5
            })
            l_loss_cv.append(cross_entropy_loss_)
            l_f1_cv.append(f1_)

        np_loss_cv = np.array(l_loss_cv)
        np_f1_cv   = np.array(l_f1_cv)
        f_out.write("%d,%1.5f,%1.5f\n" % (i, np.nanmean(np_loss_cv), np.nanmean(np_f1_cv)))
        print("Validation, Eproch %d, loss = %1.5f, f1 = %1.5f" % (i, np.nanmean(np_loss_cv), np.nanmean(np_f1_cv)))
        tf.train.write_graph(sess.graph_def, model_dir,
                        'eproch_%d_loss' % (i), as_text=False)
        saver.save(sess, '%s/eproch_%d_loss' % (model_dir, i))


def main():
    image_shape = (256, 256)
    num_classes = 2
    random.seed(0)
    
    """
    获取所有800数据的样本名称
    以 8：2 比例划分所有样本的训练集以及验证集
    注意这里我为了方便分析，已经将病理切片图片统一放置在 ./FCN/image/merge/ 文件夹中
    """
    l_sample = os.listdir("./FCN/image/merge/")
    l_sample = list(filter(lambda x: x[-4:]=="tiff", l_sample ))
    l_sample = [ s.split(".tiff")[0] for s in l_sample]
    
    random.shuffle(l_sample)
    cv_ratio = 0.8
    split_idx = int(len(l_sample)*cv_ratio)
    l_sample_train = l_sample[0:split_idx]
    l_sample_cv = l_sample[split_idx:]

    get_batches_train = gen_batch_func(l_sample_train, image_shape)
    get_batches_cv = gen_batch_func(l_sample_cv, image_shape)

    """
    模型预计占用10G 左右显存。这里设置 tensorflow 不一次性耗尽显卡的所有显存
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        config = tf.ConfigProto()
        
        """
        模型训练准备。设置使用生成器，以及占位符
        """
        vgg_path = "./FCN/vgg/"
        get_batches_fn = gen_batch_func(l_sample, image_shape)
        epochs = 30
        batch_size = g_batch_size
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)
        
        
        """
        使用定义的函数，构建模型，并进行模型的训练
        """
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = \
            load_vgg(sess, vgg_path)

        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        pred_label, training_op, cross_entropy_loss, f1, lr = \
            optimize(nn_last_layer, correct_label, learning_rate, num_classes, batch_size, split_idx)

        cross_entropy_loss_cv, f1_cv = \
            validation(nn_last_layer, correct_label, num_classes)

        train_nn(sess, epochs, batch_size, get_batches_train, get_batches_cv, training_op, cross_entropy_loss,
                 cross_entropy_loss_cv, f1, f1_cv, lr, input_image, correct_label, keep_prob, learning_rate)



if __name__ == '__main__':
    main()
