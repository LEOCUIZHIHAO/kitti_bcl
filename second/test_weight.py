import os
import pathlib
import pickle
import shutil

import fire
from second.protos import pipeline_pb2

import caffe
from caffe import layers as L, params as P

import numpy as np

"""
"""
"""
################################################################################
This is for Forward testing,
we are loading pytorch weights and feed to caffe graph.
Compare with 1st forward cls and box predict
################################################################################
"""
"""
"""


################################################################################
#Pytorch Weights and Result Path
################################################################################
torch_mlp_output_path = "0_pfn_conv2d.npy"
torch_mlp_bn_output_path = "1_pfn_norm.npy"
torch_max_output_path = "3_pfn_max.npy"
torch_scatter_output_path = "4_pfn_batch_canvas.npy"
torch_block1_output_path = "5_block1.npy"
torch_deconv1_output_path = '6_deconv1.npy'
torch_box_output_path = '7_box_preds.npy'
torch_cls_output_path = '8_cls_preds.npy'

#mlp
torch_mlp_weight_path = '0_voxel_feature_extractor.pfn_layers.0.Conv2d.weight.npy'
torch_bn_mean_path = '14_voxel_feature_extractor.pfn_layers.0.norm.running_mean.npy'
torch_bn_var_path = '15_voxel_feature_extractor.pfn_layers.0.norm.running_var.npy'
torch_sc_weight_path = '1_voxel_feature_extractor.pfn_layers.0.norm.weight.npy'
torch_sc_bias_path = '2_voxel_feature_extractor.pfn_layers.0.norm.bias.npy'

#block1
torch_block1_weight_path = '3_rpn.block1.0.weight.npy'
torch_block1_bn_mean_path = '16_rpn.block1.1.running_mean.npy'
torch_block1_bn_var_path = '17_rpn.block1.1.running_var.npy'
torch_block1_alpha_path = '4_rpn.block1.1.weight.npy'
torch_block1_belta_path = '5_rpn.block1.1.bias.npy'

torch_block1_rpncov1_weight_path = '6_rpn.block1.3.weight.npy'
torch_block1_rpncov1_bn_mean_path = '18_rpn.block1.4.running_mean.npy'
torch_block1_rpncov1_bn_var_path = '19_rpn.block1.4.running_var.npy'
torch_block1_rpncov1_alpha_path = '7_rpn.block1.4.weight.npy'
torch_block1_rpncov1_belta_path = '8_rpn.block1.4.bias.npy'

torch_block1_decov1_weight_path = '9_rpn.deconv1.0.weight.npy'
torch_block1_decov1_bn_mean_path = '20_rpn.deconv1.1.running_mean.npy'
torch_block1_decov1_bn_var_path = '21_rpn.deconv1.1.running_var.npy'
torch_block1_decov1_alpha_path = '10_rpn.deconv1.1.weight.npy'
torch_block1_decov1_belta_path = '11_rpn.deconv1.1.bias.npy'

#result
torch_cls_weight_path = '12_rpn.conv_cls.weight.npy'
torch_cls_bias_path = '13_rpn.conv_cls.bias.npy'
torch_box_weight_path = '14_rpn.conv_box.weight.npy'
torch_box_bias_path = '15_rpn.conv_box.bias.npy'

################################################################################
#Caffe Net
################################################################################
net = caffe.Net("bcl_caffe/eval.prototxt", "bcl_caffe/snapshot_iter_10.caffemodel", caffe.TEST)
# net = caffe.Net("bcl_caffe/train.prototxt", "bcl_caffe/snapshot_iter_10.caffemodel", caffe.TRAIN)
solver = caffe.get_solver("bcl_caffe/solver.prototxt")

################################################################################
#Load Pytorch Weights
################################################################################
print("\nRead Pytorch Weights ------------------------->\n")
#mlp
torch_mlp_w = np.load('./weights/'+torch_mlp_weight_path)
torch_mlp_mean = np.load('./weights/'+torch_bn_mean_path)
torch_mlp_var = np.load('./weights/'+torch_bn_var_path)
torch_mlp_alpha = np.load('./weights/'+torch_sc_weight_path)
torch_mlp_belta = np.load('./weights/'+torch_sc_bias_path)

#block1
torch_block1_w = np.load('./weights/'+torch_block1_weight_path)
torch_block1_mean = np.load('./weights/'+torch_block1_bn_mean_path)
torch_block1_var = np.load('./weights/'+torch_block1_bn_var_path)
torch_block1_alpha = np.load('./weights/'+torch_block1_alpha_path)
torch_block1_belta = np.load('./weights/'+torch_block1_belta_path)

torch_block1_rpncov1_w = np.load('./weights/'+torch_block1_rpncov1_weight_path)
torch_block1_rpncov1_mean = np.load('./weights/'+torch_block1_rpncov1_bn_mean_path)
torch_block1_rpncov1_var = np.load('./weights/'+torch_block1_rpncov1_bn_var_path)
torch_block1_rpncov1_alpha = np.load('./weights/'+torch_block1_rpncov1_alpha_path)
torch_block1_rpncov1_belta = np.load('./weights/'+torch_block1_rpncov1_belta_path)

torch_block1_decov1_w = np.load('./weights/'+torch_block1_decov1_weight_path)
torch_block1_decov1_mean = np.load('./weights/'+torch_block1_decov1_bn_mean_path)
torch_block1_decov1_var = np.load('./weights/'+torch_block1_decov1_bn_var_path)
torch_block1_decov1_alpha = np.load('./weights/'+torch_block1_decov1_alpha_path)
torch_block1_decov1_belta = np.load('./weights/'+torch_block1_decov1_belta_path)

#result
torch_clshead_w = np.load('./weights/'+torch_cls_weight_path)
torch_clshead_b = np.load('./weights/'+torch_cls_bias_path)
torch_boxhead_w = np.load('./weights/'+torch_box_weight_path)
torch_boxhead_b = np.load('./weights/'+torch_box_bias_path)


################################################################################
#Feed Pytorch weights to Caffe Net
################################################################################
print("\nFeed Pytorch Weights into caffe model------------------------->\n")

#mlp
net.params["Mlp"][0].data[...] = torch_mlp_w ## feed torch weights
net.params["bn1"][0].data[...] = torch_mlp_mean ## feed torch BN mean
net.params["bn1"][1].data[...] = torch_mlp_var ## feed torch BN var
net.params["bn1"][2].data[...] = 1 ## Force set caffe bn[2] ==1 , because caffe using sum weights !
net.params["sc1"][0].data[...] = torch_mlp_alpha ## feed torch SC alpha
net.params["sc1"][1].data[...] = torch_mlp_belta ## feed torch SC belta


#block1
net.params["init_conv1"][0].data[...] = torch_block1_w ## feed torch weights
net.params["bn2"][0].data[...] = torch_block1_mean ## feed torch BN mean
net.params["bn2"][1].data[...] = torch_block1_var ## feed torch BN var
net.params["bn2"][2].data[...] = 1 ## Force set caffe bn[2] ==1 , because caffe using sum weights !
net.params["sc2"][0].data[...] = torch_block1_alpha ## feed torch SC alpha
net.params["sc2"][1].data[...] = torch_block1_belta ## feed torch SC belta

net.params["rpn_conv1_3"][0].data[...] = torch_block1_rpncov1_w ## feed torch weights
net.params["bn3"][0].data[...] = torch_block1_rpncov1_mean ## feed torch BN mean
net.params["bn3"][1].data[...] = torch_block1_rpncov1_var ## feed torch BN var
net.params["bn3"][2].data[...] = 1 ## Force set caffe bn[2] ==1 , because caffe using sum weights !
net.params["sc3"][0].data[...] = torch_block1_rpncov1_alpha ## feed torch SC alpha
net.params["sc3"][1].data[...] = torch_block1_rpncov1_belta ## feed torch SC belta

net.params["rpn_deconv1"][0].data[...] = torch_block1_decov1_w ## feed torch weights
net.params["bn4"][0].data[...] = torch_block1_decov1_mean ## feed torch BN mean
net.params["bn4"][1].data[...] = torch_block1_decov1_var ## feed torch BN var
net.params["bn4"][2].data[...] = 1 ## Force set caffe bn[2] ==1 , because caffe using sum weights !
net.params["sc4"][0].data[...] = torch_block1_decov1_alpha ## feed torch SC alpha
net.params["sc4"][1].data[...] = torch_block1_decov1_belta ## feed torch SC belta


#result
net.params["cls_head"][0].data[...] = torch_clshead_w ## feed torch weights
net.params["cls_head"][1].data[...] = torch_clshead_b ## feed torch weights
net.params["reg_head"][0].data[...] = torch_boxhead_w ## feed torch weights
net.params["reg_head"][1].data[...] = torch_boxhead_b ## feed torch weights


torch_weight = net.params["Mlp"][0].data[...]
handcal_mean = net.params["bn1"][0].data[...]
handcal_variance = net.params["bn1"][1].data[...]
handcal_alpha = net.params["sc1"][0].data[...]
handcal_belta = net.params["sc1"][1].data[...]

for i in range(50):
    net.forward()
    # solver.step(1)

print("\nPrint data ------------------------->\n")
data = net.blobs['data'].data
print("data shape ", data.shape)

# for eval
# img_idx = net.blobs['img_idx'].data
# print("img index\n", img_idx)


################################################################################
#Hand calculate
################################################################################
print("\nHand Calucalte using Torch weights result ------------------------->\n")
def mlp_mul(data, handcal_score, outer=None, inner=None, weights=None):
    for i in range(outer):
        for j in range(inner):
            for idx, w_row in enumerate(weights):
                handcal_score[:,idx,i,j] = np.sum(np.squeeze(data[:,:,i,j]) * np.squeeze(w_row))

    return handcal_score
handcal_score = np.zeros(shape=(1,64,data.shape[2],100))
handcal_score = mlp_mul(data, handcal_score, outer=2, inner=data.shape[3], weights=torch_weight)
# print("hand calculated_socre with Torch weights\n", handcal_score[:,:,:2,:])

# handcal_bn = np.zeros(shape=(1,64,data.shape[2],100))
# current_mlp_output = net.blobs['Mlp'].data
# for i in range(current_mlp_output.shape[1]):
#     handcal_bn[:,i,:,:] = (current_mlp_output[:,i,:,:] - handcal_mean[i])/np.sqrt(handcal_variance[i] + 1e-3)
# print("hand calculated_BN with Torch weights\n", handcal_bn[:,:,:2,:])

handcal_sc = np.zeros(shape=(1,64,data.shape[2],100))
current_bn_output = net.blobs['bn1'].data
for i in range(current_bn_output.shape[1]):
    handcal_sc[:,i,:,:] = (current_bn_output[:,i,:,:] * handcal_alpha[i]) + handcal_belta[i]
# print("hand calculated_SC with Torch weights\n", handcal_sc[:,:,:2,:])

################################################################################
#Caffe
################################################################################
print("\nCaffe Forward using Torch weights result ------------------------->\n")
caffe_mlp_scores = net.blobs['Mlp'].data
# print("Caffe net calculated_socre with Torch weights\n", caffe_mlp_scores[:,:,:2,:])


caffe_bn_scores = net.blobs['bn1'].data
# print("Caffe net BN calculated_socre with Torch weights\n", caffe_bn_scores[:,:,:2,:])


caffe_sc_scores = net.blobs['sc1'].data
# print("Caffe net SC calculated_socre with Torch weights\n", caffe_sc_scores[:,:,:2,:])


caffe_max_scores = net.blobs['max_pool'].data
# print("Caffe net Max Pool with Torch weights\n", caffe_max_scores[:,:,:2,:])


caffe_scatter_score = net.blobs['PillarScatter'].data
# print("Caffe net Pillar Scatter with Torch weights\n", caffe_scatter_score[:,:,:2,:])


caffe_scatter_score = net.blobs['PillarScatter'].data
# print("Caffe net Pillar Scatter with Torch weights\n", caffe_scatter_score[:,:,:2,:])


caffe_initcov_score = net.blobs['relu2'].data # initconv only
# print("Caffe net init_conv1 with Torch weights\n", caffe_initcov_score[:,:,:2,:])


caffe_rpncov1_score = net.blobs['relu3'].data # initconv + rpncov1
# print("Caffe net rpn_conv1 with Torch weights\n", caffe_rpncov1_score[:,:,:2,:])


caffe_decov1_score = net.blobs['relu4'].data # initconv + rpncov1 + deconv1
# print("Caffe net deconv1 with Torch weights\n", caffe_decov1_score[:,:,:2,:])


caffe_cls_preds_score = net.blobs['cls_preds'].data # initconv + rpncov1 + deconv1
# print("Caffe net cls_preds with Torch weights\n", caffe_cls_preds_score[:,:,:2,:])


caffe_box_preds_score = net.blobs['box_preds'].data # initconv + rpncov1 + deconv1
# print("Caffe net box_preds with Torch weights\n", caffe_box_preds_score[:,:,:2,:])

################################################################################
#Torch
################################################################################
def read_troch_output(**kargs):
    torch_name_dict = {}
    for key in kargs.keys():
        result = np.load('./output/'+kargs[str(key)])
        torch_name_dict[str(key)] = result
    return torch_name_list

print("\nPytorch Forward result ------------------------->\n")

# read_troch_output(torch_scores=torch_mlp_output_path,
#                 torch_bn_scores=torch_mlp_bn_output_path,
#                 torch_sc_scores=torch_mlp_bn_output_path,
#                 torch_max_scores=torch_max_output_path,
#                 torch_scatter_scores=torch_scatter_output_path,
#                 torch_initcov_score=torch_initcov_score)

torch_scores = np.load('./output/'+torch_mlp_output_path)
# print("Pytorch net calculated_socre\n", torch_scores[:,:,:2,:])


torch_bn_scores = np.load('./output/'+torch_mlp_bn_output_path)
# print("Pytorch net BN calculated_socre\n", torch_bn_scores[:,:,:2,:])


torch_sc_scores = np.load('./output/'+torch_mlp_bn_output_path)
# print("Pytorch net SC calculated_socre\n", torch_sc_scores[:,:,:2,:])


torch_max_scores = np.load('./output/'+torch_max_output_path)
# print("Pytorch net Max Pool\n", torch_max_scores[:,:,:2,:])


torch_scatter_scores = np.load('./output/'+torch_scatter_output_path)
# print("Pytorch net Pillar Scatter\n", torch_scatter_scores[:,:,:2,:])


torch_initcov_score = np.load('./output/'+torch_block1_output_path) # initconv only
# print("Pytorch net init_conv1 with Torch weights\n", torch_initcov_score[:,:,:2,:])


torch_rpncov1_score = np.load('./output/'+torch_block1_output_path) # initconv + rpncov1
# print("Pytorch net rpn_conv1 with Torch weights\n", torch_rpncov1_score[:,:,:2,:])


torch_decov1_score = np.load('./output/'+torch_deconv1_output_path) # initconv + rpncov1 + deconv1
# print("Pytorch net deconv1 with Torch weights\n", torch_decov1_score[:,:,:2,:])


torch_cls_preds_score = np.load('./output/'+torch_cls_output_path) # initconv + rpncov1 + deconv1
# print("Pytorch net cls_preds with Torch weights\n", torch_cls_preds_score[:,:,:2,:])


torch_box_preds_score = np.load('./output/'+torch_box_output_path) # initconv + rpncov1 + deconv1
# print("Pytorch net box_preds with Torch weights\n", torch_box_preds_score[:,:,:2,:])


################################################################################
#Log
################################################################################
print("\nCompare Hand calculate and Caffe Forward result <First 2 Voxels> ------------------------->\n")
print("MLP Result : ", np.unique(np.abs(handcal_score[:,:,:2,:] - caffe_mlp_scores[:,:,:2,:])<0.1 , return_counts=1))
# print("MLP BN Result : ", np.unique(np.abs(handcal_bn[:,:,:,:] - caffe_bn_scores[:,:,:,:])<0.1 , return_counts=1))
print("MLP BN+SC Result : ", np.unique(np.abs(handcal_sc[:,:,:,:] - caffe_sc_scores[:,:,:,:])<0.1 , return_counts=1))

print("\nCompare Hand calculate and Pytorch Forward result <First 2 Voxels>------------------------->\n")
print("MLP Result : ", np.unique(np.abs(handcal_score[:,:,:2,:] - torch_scores[:,:,:2,:])<0.1 , return_counts=1))
# print("BN Result : ", np.unique(np.abs(handcal_bn[:,:,:,:] - torch_bn_scores[:,:,:,:])<0.1 , return_counts=1))
print("MLP BN+SC Result : ", np.unique(np.abs(handcal_sc[:,:,:,:] - torch_sc_scores[:,:,:,:])<0.1 , return_counts=1))

print("\nCompare Pytorch and Caffe Forward result ------------------------->\n")
print("MLP Result : ", np.unique(np.abs(torch_scores[:,:,:,:] - caffe_mlp_scores[:,:,:,:])<0.1 , return_counts=1))
# print("BN Result : ", np.unique(np.abs(torch_bn_scores[:,:,:,:] - caffe_bn_scores[:,:,:,:])<0.1 , return_counts=1))
print("MLP BN+SC Result : ", np.unique(np.abs(torch_sc_scores[:,:,:,:] - caffe_sc_scores[:,:,:,:])<0.1 , return_counts=1))
print("MAX POOL Result : ", np.unique(np.abs(torch_max_scores[:,:,:,:] - caffe_max_scores[:,:,:,:])<0.1 , return_counts=1))
print("Scatter Result : ", np.unique(np.abs(torch_scatter_scores[...] - caffe_scatter_score[...])<0.1 , return_counts=1))
print("BLOCK1 (init+rpncov1) Result : ", np.unique(np.abs(torch_rpncov1_score[...] - caffe_rpncov1_score[...])<0.1 , return_counts=1))
print("DECONV Result : ", np.unique(np.abs(torch_decov1_score[...] - caffe_decov1_score[...])<0.1 , return_counts=1))
print("CLS Result : ", np.unique(np.abs(torch_cls_preds_score[...] - caffe_cls_preds_score[...])<0.1 , return_counts=1))
print("BOX Result : ", np.unique(np.abs(torch_box_preds_score[...] - caffe_box_preds_score[...])<0.001 , return_counts=1))





caffe_labels = net.blobs['labels'].data
print("Caffe net labels\n", np.unique(caffe_labels, return_counts=1))


caffe_cls_loss = net.blobs['cls_loss'].data
print("Caffe net cls_loss with Torch weights ", caffe_cls_loss[...])


caffe_box_loss = net.blobs['reg_loss'].data
print("Caffe net box_loss with Torch weights ", caffe_box_loss[...])


"""
"""
"""
################################################################################
This is for Backward testing
################################################################################
"""
"""
"""


print("\n\n####################################################################")
print("START BACKWARD-------------------------------------------------------->")
print("####################################################################\n\n")


net.backward()

print(net.blobs.keys())
################################################################################
#This is for the derivitive of the Loss check
################################################################################
print("\n\nDerivite Check------------------------------------------------>\n\n")

cls_dev = np.load('./gradient/'+'FocalLossGrad.npy')
reg_dev = np.load('./gradient/'+'regLossGrad.npy')

print("Torch cls_backward diff sum ", cls_dev.sum())
print("Caffe cls_backward diff sum ", net.blobs["cls_preds"].diff.sum(), "\n")

print("Torch reg_backward diff sum ", reg_dev.sum())
print("Caffe reg_backward diff sum ", net.blobs["box_preds"].diff.sum(), "\n")

# print("Torch reg_backward diff shape ", reg_dev.reshape(1,7,-1)[:,:-1,:].shape)
# print("Caffe reg_backward diff shape", net.blobs["box_preds"].diff.reshape(1,7,-1)[:,:-1,:].shape, "\n")

# print("Torch reg_backward diff sum without rotation ", reg_dev.reshape(1,7,-1)[:,:-1,:].sum())
# print("Caffe reg_backward diff sum  without rotation", net.blobs["box_preds"].diff.reshape(1,7,-1)[:,:-1,:].sum(), "\n")


# print("Torch reg_backward diff sum ", reg_dev.sum())
# print("Caffe sin_diff diff sum ", net.blobs["box_preds_sin_diff"].diff.sum(), "\n")

print("Caffe and Torch (0.001) cls_backward diff comparision: ", np.unique(np.abs(net.blobs['cls_preds'].diff - cls_dev)<0.0001 , return_counts=1))
print("Caffe and Torch (0.001) reg_backward diff comparision: ", np.unique(np.abs(net.blobs['box_preds'].diff - reg_dev)<0.0001 , return_counts=1), "\n")



################################################################################
#This is for the weights and bias gradients (delta(L)/delta(W))
################################################################################
print("\n\nGradients Check----------------------------------------------->\n\n")


cls_grad_w = np.load('./gradient/'+'12_rpn.conv_cls.weight_grad.npy')
cls_grad_b = np.load('./gradient/'+'13_rpn.conv_cls.bias_grad.npy')
box_grad_w = np.load('./gradient/'+'14_rpn.conv_box.weight_grad.npy')
box_grad_b = np.load('./gradient/'+'15_rpn.conv_box.bias_grad.npy')
decov_bn_rpn_w = np.load('./gradient/'+'10_rpn.deconv1.1.weight_grad.npy')
decov_bn_rpn_b = np.load('./gradient/'+'11_rpn.deconv1.1.bias_grad.npy')
decov_rpn_w = np.load('./gradient/'+'9_rpn.deconv1.0.weight_grad.npy')
ini_rpn_w = np.load('./gradient/'+'3_rpn.block1.0.weight_grad.npy')
mlp_w  = np.load('./gradient/'+'0_voxel_feature_extractor.pfn_layers.0.Conv2d.weight_grad.npy')


print("Torch Backward cls_grad weight sum ", cls_grad_w.sum())
print("Caffe Backward cls_grad weight sum ", net.params["cls_head"][0].diff.sum(), "\n")

print("Torch Backward cls_grad bias sum ", cls_grad_b.sum())
print("Caffe Backward cls_grad bias sum ", net.params["cls_head"][1].diff.sum(), "\n")

print("Torch Backward cls_grad bias ", cls_grad_b)
print("Caffe Backward cls_grad bias ", net.params["cls_head"][1].diff, "\n")

print("Torch Backward box_grad weight sum ", box_grad_w.sum())
print("Caffe Backward box_grad weight  sum ", net.params["reg_head"][0].diff.sum(), "\n")

# print("Torch Backward box_grad weight ", box_grad_w)
# print("Caffe Backward box_grad weight ", net.params["reg_head"][0].diff, "\n")


print("Torch Backward box_grad bias sum ", box_grad_b.sum())
print("Caffe Backward box_grad bias sum ", net.params["reg_head"][1].diff.sum(), "\n")

print("Torch Backward decov_bn_rpn weight sum ", decov_bn_rpn_w.sum())
print("Caffe Backward decov_bn_rpn weight sum ", net.params["sc4"][0].diff.sum(), "\n")

print("Torch Backward decov_bn_rpn bias sum ", decov_bn_rpn_b.sum())
print("Caffe Backward decov_bn_rpn bias sum ", net.params["sc4"][1].diff.sum(), "\n")

print("Torch Backward decov_rpn weight sum ", decov_rpn_w.sum())
print("Caffe Backward decov_rpn weight sum ", net.params["rpn_deconv1"][0].diff.sum(), "\n")

print("Torch Backward decov_rpn weight mean ", decov_rpn_w.mean())
print("Caffe Backward decov_rpn weight mean ", net.params["rpn_deconv1"][0].diff.mean(), "\n")

print("Torch Backward ini_rpn weight sum ", ini_rpn_w.sum())
print("Caffe Backward ini_rpn weight sum ", net.params["init_conv1"][0].diff.sum(), "\n")

print("Torch Backward ini_rpn weight mean ", ini_rpn_w.mean())
print("Caffe Backward ini_rpn weight mean ", net.params["init_conv1"][0].diff.mean(), "\n")

print("Torch Backward mlp weight sum ", mlp_w.sum())
print("Caffe Backward mlp weight sum ", net.params["Mlp"][0].diff.sum(), "\n")

print("Torch Backward mlp weight mean ", mlp_w.mean())
print("Caffe Backward mlp weight mean ", net.params["Mlp"][0].diff.mean(), "\n")


print("Caffe and Torch (0.0001) cls_grad grad weights comparision: ", np.unique(np.abs(net.params["cls_head"][0].diff - cls_grad_w)<0.0001 , return_counts=1))
print("Caffe and Torch (0.0001) cls_grad grad bias comparision: ", np.unique(np.abs(net.params["cls_head"][1].diff - cls_grad_b)<0.0001 , return_counts=1))
print("Caffe and Torch (0.0001) box_grad grad weights comparision: ", np.unique(np.abs(net.params["reg_head"][0].diff - box_grad_w)<0.0001 , return_counts=1))
print("Caffe and Torch (0.0001) box_grad grad bias comparision: ", np.unique(np.abs(net.params["reg_head"][1].diff - box_grad_b)<0.0001 , return_counts=1))
print("Caffe and Torch (0.0001) decov_bn_rpn grad weight comparision: ", np.unique(np.abs(net.params["sc4"][0].diff - decov_bn_rpn_w)<0.0001 , return_counts=1))
print("Caffe and Torch (0.0001) decov_bn_rpn grad bias comparision: ", np.unique(np.abs(net.params["sc4"][1].diff - decov_bn_rpn_b)<0.0001 , return_counts=1))
print("Caffe and Torch (0.0001) decov_rpn grad bias comparision: ", np.unique(np.abs(net.params["rpn_deconv1"][0].diff - decov_rpn_w)<0.0001 , return_counts=1))
print("Caffe and Torch (0.0001) ini_rpn grad weight comparision: ", np.unique(np.abs(net.params["init_conv1"][0].diff - ini_rpn_w)<0.0001 , return_counts=1))
print("Caffe and Torch (0.0001) mlp grad weight comparision: ", np.unique(np.abs(net.params["Mlp"][0].diff - mlp_w)<0.0001 , return_counts=1))
