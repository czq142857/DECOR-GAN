import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", action="store", dest="epoch", default=20, type=int, help="Epoch to train [20]")
parser.add_argument("--iteration", action="store", dest="iteration", default=0, type=int, help="Iteration to train. Either epoch or iteration need to be zero [0]")

parser.add_argument("--data_style", action="store", dest="data_style", help="The name of dataset")
parser.add_argument("--data_content", action="store", dest="data_content", help="The name of dataset")
parser.add_argument("--data_dir", action="store", dest="data_dir", help="Root directory of dataset")
parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="checkpoint", help="Directory name to save the checkpoints [checkpoint]")
parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="./samples/", help="Directory name to save the image samples [samples]")

parser.add_argument("--input_size", action="store", dest="input_size", default=64, type=int, help="Input voxel size [64]")
parser.add_argument("--output_size", action="store", dest="output_size", default=256, type=int, help="Output voxel size [256]")
# note -- valid settings:
# input 64, output 256, x4
# input 32, output 128, x4
# input 32, output 256, x8
# input 16, output 128, x8

parser.add_argument("--asymmetry", action="store_true", dest="asymmetry", default=False, help="True for training on asymmetric shapes [False]")

parser.add_argument("--alpha", action="store", dest="alpha", default=0.5, type=float, help="Parameter alpha [0.5]")
parser.add_argument("--beta", action="store", dest="beta", default=10.0, type=float, help="Parameter beta [10.0]")

parser.add_argument("--train", action="store_true", dest="train", default=False, help="True for trainin [False]")
parser.add_argument("--test", action="store_true", dest="test", default=False, help="True for rough testing [False]")

parser.add_argument("--prepvox", action="store_true", dest="prepvox", default=False, help="True for preparing voxels for evaluating IOU, LP and Div [False]")
parser.add_argument("--prepvoxstyle", action="store_true", dest="prepvoxstyle", default=False, help="True for preparing voxels for evaluating IOU, LP and Div [False]")
parser.add_argument("--evalvox", action="store_true", dest="evalvox", default=False, help="True for evaluating IOU, LP and Div [False]")

parser.add_argument("--prepimg", action="store_true", dest="prepimg", default=False, help="True for preparing rendered views for evaluating Cls_score [False]")
parser.add_argument("--prepimgreal", action="store_true", dest="prepimgreal", default=False, help="True for preparing rendered views of all content shapes (as real) for evaluating Cls_score [False]")
parser.add_argument("--evalimg", action="store_true", dest="evalimg", default=False, help="True for evaluating Cls_score [False]")

parser.add_argument("--prepFID", action="store_true", dest="prepFID", default=False, help="True for preparing voxels for evaluating FID [False]")
parser.add_argument("--prepFIDmodel", action="store_true", dest="prepFIDmodel", default=False, help="True for training a classifier for evaluating FID [False]")
parser.add_argument("--prepFIDreal", action="store_true", dest="prepFIDreal", default=False, help="True for computing the mean and sigma vectors (real) for evaluating FID [False]")
parser.add_argument("--evalFID", action="store_true", dest="evalFID", default=False, help="True for evaluating FID [False]")

parser.add_argument("--ui", action="store_true", dest="ui", default=False, help="launch a UI for latent space exploration [False]")

parser.add_argument("--gpu", action="store", dest="gpu", default="0", help="to use which GPU [0]")

FLAGS = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu


from modelAE import IM_AE

if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)


if FLAGS.train:
    im_ae = IM_AE(FLAGS)
    im_ae.train(FLAGS)

elif FLAGS.test:
    im_ae = IM_AE(FLAGS)
    im_ae.test(FLAGS)

elif FLAGS.prepvox:
    im_ae = IM_AE(FLAGS)
    im_ae.prepare_voxel_for_eval(FLAGS)

elif FLAGS.prepvoxstyle:
    import evalAE
    im_ae = IM_AE(FLAGS)
    im_ae.prepare_voxel_style(FLAGS)
    evalAE.precompute_unique_patches_per_style(FLAGS)
    #evalAE.precompute_unique_patches_all_styles(FLAGS) # useless

elif FLAGS.evalvox:
    import evalAE
    evalAE.eval_IOU(FLAGS)
    evalAE.eval_LP_Div_IOU(FLAGS)
    evalAE.eval_LP_Div_Fscore(FLAGS)
    #evalAE.eval_LP_Div_MAE(FLAGS) #not used

elif FLAGS.prepimg:
    im_ae = IM_AE(FLAGS)
    im_ae.render_fake_for_eval(FLAGS)

elif FLAGS.prepimgreal:
    im_ae = IM_AE(FLAGS)
    im_ae.render_real_for_eval(FLAGS)

elif FLAGS.evalimg:
    import evalResNet
    evalResNet.eval_Cls_score(FLAGS)

elif FLAGS.prepFID:
    im_ae = IM_AE(FLAGS)
    im_ae.prepare_voxel_for_FID(FLAGS)

elif FLAGS.prepFIDmodel:
    import evalFID
    evalFID.train_classifier(FLAGS)

elif FLAGS.prepFIDreal:
    import evalFID
    evalFID.compute_FID_for_real(FLAGS)

elif FLAGS.evalFID:
    import evalFID
    evalFID.eval_FID(FLAGS)

elif FLAGS.ui:
    im_ae = IM_AE(FLAGS)
    im_ae.launch_ui(FLAGS)

else:
    print('?!')
