import caffe
from caffe import layers as L, params as P, to_proto
from bcl_caffe.utils import get_prototxt, parse_channel_scale, map_channel_scale

def conv_bn_relu(n, name, top_prev, ks, nout, stride=1, pad=0, loop=1):

    for idx in range(loop):
        n[str(name)+"_"+str(idx)] = L.Convolution(top_prev, #name = name,
                                            convolution_param=dict(
                                                    kernel_size=ks, stride=stride,
                                                    num_output=nout, pad=pad,
                                                    engine=2,
                                                    weight_filler=dict(type = 'xavier'),
                                                    bias_term = False),
                                                    param=[dict(lr_mult=1)],
                                                    )
        top_prev = n[str(name)+"_"+str(idx)]
        n[str(name)+'_bn_'+str(idx)] = L.BatchNorm(top_prev, batch_norm_param=dict(eps=1e-3, moving_average_fraction=0.99))
        top_prev = n[str(name)+'_bn_'+str(idx)]
        n[str(name)+'_sc_'+str(idx)] = L.Scale(top_prev, scale_param=dict(bias_term=True))
        top_prev = n[str(name)+'_sc_'+str(idx)]
        n[str(name)+'_relu_'+str(idx)] = L.ReLU(top_prev, in_place=True)
        top_prev = n[str(name)+'_relu_'+str(idx)]

    return top_prev

def deconv_bn_relu(n, name, top_prev, ks, nout, stride=1, pad=0):
    n[str(name)] = L.Deconvolution(top_prev, # name = name,
                                            convolution_param=dict(kernel_size=ks, stride=stride,
                                                num_output=nout, pad=pad,
                                                engine=2,
                                                weight_filler=dict(type = 'xavier'),
                                                bias_term = False),
                                                param=[dict(lr_mult=1)],
                                                )
    top_prev = n[str(name)]
    n[str(name)+'_bn'] = L.BatchNorm(top_prev, batch_norm_param=dict(eps=1e-3, moving_average_fraction=0.99))
    top_prev = n[str(name)+'_bn']
    n[str(name)+'_sc'] = L.Scale(top_prev, scale_param=dict(bias_term=True))
    top_prev = n[str(name)+'_sc']
    n[str(name)+'_relu'] = L.ReLU(top_prev, in_place=True)
    top_prev = n[str(name)+'_relu']

    return top_prev

def bcl_bn_relu(n, name, top_prev, top_lat_feats, nout, lattic_scale=None, loop=1):

    for idx in range(loop):

        if lattic_scale:

            # if use python mode ["0*16_1*16_2*16", "0*8_1*8_2*8", "0*2_1*2_2*2"]
            # _lattic_scale = lattic_scale
            n[str(name)+"_scale_"+str(idx)] = L.Python(top_lat_feats, python_param=dict(module='bcl_layers',
                                                                    layer='PickAndScale',
                                                                    param_str=lattic_scale[idx]))
            _top_lat_feats = n[str(name)+"_scale_"+str(idx)]


        bltr_weight_filler = dict(type='gaussian', std=float(0.001))
        n[str(name)+"_"+str(idx)] = L.Permutohedral(top_prev, _top_lat_feats, _top_lat_feats,
                                                        ntop=1,
                                                        permutohedral_param=dict(
                                                            num_output=nout[idx],
                                                            group=1,
                                                            neighborhood_size=1,
                                                            bias_term=True,
                                                            norm_type=P.Permutohedral.AFTER,
                                                            offset_type=P.Permutohedral.NONE,
                                                            filter_filler=bltr_weight_filler,
                                                            bias_filler=dict(type='constant',
                                                                             value=0)),
                                                        param=[{'lr_mult': 1, 'decay_mult': 1},
                                                               {'lr_mult': 2, 'decay_mult': 0}])

        top_prev = n[str(name)+"_"+str(idx)]
        n[str(name)+'_bn_'+str(idx)] = L.BatchNorm(top_prev, batch_norm_param=dict(eps=1e-3, moving_average_fraction=0.99))
        top_prev = n[str(name)+'_bn_'+str(idx)]
        n[str(name)+'_sc_'+str(idx)] = L.Scale(top_prev, scale_param=dict(bias_term=True))
        top_prev = n[str(name)+'_sc_'+str(idx)]
        n[str(name)+'_relu_'+str(idx)] = L.ReLU(top_prev, in_place=True)
        top_prev = n[str(name)+'_relu_'+str(idx)]

    return top_prev


############################################################################

"""
There are two main modes. One is Point2BCL, the other is Voxel2BCL, First mode uses
raw points as BCL input and raw position as BCL's lattice feature. The Voxel2BCL
uses Voxel (like superpixel) as BCL input data and Voxels coordinates as the BCL's
lattice feature
"""


"""
Feature Creation has 4 types, if use  VoxelFeatureNet, VoxelFeatureNetV2 or False
need to keep one max point per voxels normally use PointNet and maxpooling afterwards.
The PointNet mode should equale True.

If Feature Creation is SimpleVoxel it means
calculate the mean points inside of voxels (without any learnable conv)
The PointNet mode should equal Fasle.
"""

############################################################################

def test_v2(phase,
            dataset_params=None,
            model_cfg = None,
            deploy=False,
            create_prototxt=True,
            save_path=None,
            ):

    #RPN config
    num_filters=list(model_cfg.rpn.num_filters)
    layer_nums=list(model_cfg.rpn.layer_nums)
    layer_strides=list(model_cfg.rpn.layer_strides)
    num_upsample_filters=list(model_cfg.rpn.num_upsample_filters)
    upsample_strides=list(model_cfg.rpn.upsample_strides)

    point_cloud_range=list(model_cfg.voxel_generator.point_cloud_range)
    voxel_size=list(model_cfg.voxel_generator.voxel_size)
    # anchors_fp_size = (point_cloud_range[3:]-point_cloud_range[:3])/voxel_size
    anchors_fp_w = 432 #1408
    anchors_fp_h = 496 #1600

    box_code_size = 7
    num_anchor_per_loc = 2

    ############################################################################
    # Voxel2BCL
    # Voxel2PointNet
    ############################################################################
    BCL_mode = 'Voxel2BCL'
    dataset_params['x2BCL'] = BCL_mode
    dataset_params['Voxel2BCL_numpoint'] = 6000 #num voxels

    ############################################################################
    # Featuer Creation
    # VoxelFeatureNet: xyzr + (cente_x, center_z, center_y), (cluster_x, cluster_z)
    # VoxelFeatureNetV2: xyzr + (cluster_x, cluster_z)
    # False: No Feature extraction only xyzr
    # SimpleVoxel: sum points in voxel and divided by num of points left 1 points
    #              if use SimpleVoxel PointNet Should disable!
    ############################################################################
    dataset_params['FeatureNet'] = 'SimpleVoxel'
    dataset_params['Save_Img'] = False
    ############################################################################
    # if PointNet == True then it means PointNet to extract high dimention features
    # and max pooling to reduce the point to 1
    # Normally except Simplex the rest of the Freature Creation must with
    # PointNet acticate
    ############################################################################


    n = caffe.NetSpec()

    if phase == "train":

        dataset_params_train = dataset_params.copy()
        dataset_params_train['subset'] = phase

        datalayer_train = L.Python(name='data', include=dict(phase=caffe.TRAIN),
                                   ntop= 4, python_param=dict(module='bcl_layers', layer='InputKittiData',
                                                     param_str=repr(dataset_params_train)))

        n.data, n.coors, n.labels, n.reg_targets = datalayer_train

    elif phase == "eval":
        dataset_params_eval = dataset_params.copy()
        dataset_params_eval['subset'] = phase

        datalayer_eval = L.Python(name='data', include=dict(phase=caffe.TEST),
                                  ntop= 9, python_param=dict(module='bcl_layers', layer='InputKittiData',
                                                     param_str=repr(dataset_params_eval)))

        n.data, n.coors, n.anchors, n.rect, n.trv2c, n.p2, n.anchors_mask, n.img_idx, n.img_shape = datalayer_eval

    if deploy:
        print("[debug] run deploy in caffe_model.py")
        # n.data = L.Input(shape=dict(dim=[1, len(input_dims), 1, sample_size]))


    top_prev = n.data

    """BCL fixed size before Scatter"""
    ############################################################################
    # Method 1
    # use new xyz as the lattice features
    # this reshape is used to make n.data from (1,Feature,Npoints,Voxels) -> (1,Feature,1,Voxels) Npoints in here should be 1
    # and n.data -> (1,Feature,Npoints,Voxels) -> (1, Feature[:3], 1, Voxeld) Npoints in here should be 1
    # this is particular for raw XYZ features as BCL data input which means there is no PointNet or any features extraction infront
    # Or must keep the origin xyz features inside the new features
    ############################################################################

    # n["input_feats"], n['lat_feats']= L.Python(n.data, ntop=2, python_param=dict(module='bcl_layers',
    #                                                                             layer='BCLReshape',
    #                                                                             param_str=str(dict(data_feature=True))))
    # top_prev, top_lat_feats = n["input_feats"], n['lat_feats']
    # top_prev = bcl_bn_relu(n, 'bcl0', top_prev, top_lat_feats, nout=[64,128,64], lattic_scale=["0*8_1*8_2*8", "0*4_1*4_2*4", "0*2_1*2_2*2"], loop=3)


    if BCL_mode=="Voxel2BCL":

        # Reshape to the (B,C,N,V) N is 1 here to fit in BCL
        n["input_feats"], n['lat_feats']= L.Python(top_prev, n.coors, ntop=2, python_param=dict(module='bcl_layers',
                                                                                    layer='BCLReshape',
                                                                                    param_str=str(dict(ReshapeMode=BCL_mode))))
        top_prev, top_lat_feats = n["input_feats"], n['lat_feats']

        # top_prev = bcl_bn_relu(n, 'bcl0', top_prev, top_lat_feats, nout=[64,128,64], lattic_scale=["0*8_1*8_2*8", "0*4_1*4_2*4", "0*2_1*2_2*2"], loop=3)
        # top_prev = bcl_bn_relu(n, 'bcl0', top_prev, top_lat_feats, nout=[64,128,64], lattic_scale=["0*1_1*1_2*1", "0*0.5_1*0.5_2*0.5", "0*0.25_1*0.25_2*0.25"], loop=3)
        top_prev = bcl_bn_relu(n, 'bcl0', top_prev, top_lat_feats, nout=[64,128,64], lattic_scale=["0*32_1*32_2*32", "0*16_1*16_2*16", "0*8_1*8_2*8"], loop=3)

        # Reshape to the (B,C,V,N) N is 1 here to fit in Scatter
        n["input_feats_inverse"]= L.Python(top_prev,python_param=dict(module='bcl_layers',
                                                                    layer='Voxel2Scatter',
                                                                        ))
        top_prev = n["input_feats_inverse"]

    if BCL_mode=="Voxel2PointNet":
        top_prev = conv_bn_relu(n, "mlp0", top_prev, 1, 64, stride=1, pad=0, loop=1)
        top_prev = conv_bn_relu(n, "mlp1", top_prev, 1, 128, stride=1, pad=0, loop=1)
        top_prev = conv_bn_relu(n, "mlp2", top_prev, 1, 64, stride=1, pad=0, loop=1)

    ###############################Scatter######################################
    n['PillarScatter'] = L.Python(top_prev, n.coors, ntop=1,python_param=dict(
                            module='bcl_layers',
                            layer='PointPillarsScatter',
                            param_str=str(dict(output_shape=[1, 1, anchors_fp_h, anchors_fp_w, 64], # [1, 1, 496, 432, 4]
                                            permutohedral=False # if true return shape is (b,c,1,h*w) else (b.c,h,w)
                                            ))))
    top_prev= n['PillarScatter']
    ###############################Scatter######################################


    #############################MODE1##########################################
    """ No Concate"""
    # top_prev = bcl_bn_relu(n, 'bcl0',
    #                         top_prev,
    #                         top_lat_feats,
    #                         nout=[64,128,128,128,64,64],
    #                         lattic_scale=["0*16_1*16_2*16", "0*8_1*8_2*8", "0*4_1*4_2*4", "0*2_1*2_2*2", "0*0.5_1*0.5_2*0.5"],
    #                         loop=6)

    #############################MODE1##########################################


    #############################MODE2##########################################
    """ Concate (might have rpn and feature extract function?)"""
    # top_prev_1 = bcl_bn_relu(n, 'bcl0',
    #                     top_prev,
    #                     top_lat_feats,
    #                     nout=[64,128],
    #                     lattic_scale=["0*8_1*8_2*8", "0*4_1*4_2*4"],
    #                     loop=2)
    #
    # top_prev_2 = bcl_bn_relu(n, 'bcl1',
    #                         top_prev_1,
    #                         top_lat_feats,
    #                         nout=[128,128],
    #                         lattic_scale=["0*2_1*2_2*2", "0*1_1*1_2*1"],
    #                         loop=2)
    #
    # top_prev_3 = bcl_bn_relu(n, 'bcl2',
    #                         top_prev_2,
    #                         top_lat_feats,
    #                         nout=[64,64],
    #                         lattic_scale=["0*0.5_1*0.5_2*0.5", "0*0.25_1*0.25_2*0.25"],
    #                         loop=2)
    #
    # n['rpn_out'] = L.Concat(top_prev_1, top_prev_2, top_prev_3)
    # top_prev = n['rpn_out']

    # n['reshape_rpn_out'] = L.Reshape(top_prev, reshape_param=dict(shape=dict(dim=[0, 0, int(anchors_fp_h/2), int(anchors_fp_w/2)])))# (B,H,W,C) -> (B, -1, C)
    # top_prev = n['reshape_rpn_out']
    #############################MODE2##########################################


    #############################MODE3##########################################
    top_prev = conv_bn_relu(n, "ini_conv1", top_prev, 3, num_filters[0], stride=layer_strides[0], pad=1, loop=1)
    top_prev = conv_bn_relu(n, "rpn_conv1", top_prev, 3, num_filters[0], stride=1, pad=1, loop=layer_nums[0]) #3
    deconv1 = deconv_bn_relu(n, "rpn_deconv1", top_prev, upsample_strides[0], num_upsample_filters[0], stride=upsample_strides[0], pad=0)

    top_prev = conv_bn_relu(n, "ini_conv2", top_prev, 3, num_filters[1], stride=layer_strides[1], pad=1, loop=1)
    top_prev = conv_bn_relu(n, "rpn_conv2", top_prev, 3, num_filters[1], stride=1, pad=1, loop=layer_nums[1]) #5
    deconv2 = deconv_bn_relu(n, "rpn_deconv2", top_prev, upsample_strides[1], num_upsample_filters[1], stride=upsample_strides[1], pad=0)

    top_prev = conv_bn_relu(n, "ini_conv3", top_prev, 3, num_filters[2], stride=layer_strides[2], pad=1, loop=1)
    top_prev = conv_bn_relu(n, "rpn_conv3", top_prev, 3, num_filters[2], stride=1, pad=1, loop=layer_nums[2]) #5
    deconv3 = deconv_bn_relu(n, "rpn_deconv3", top_prev, upsample_strides[2], num_upsample_filters[2], stride=upsample_strides[2], pad=0)

    n['rpn_out'] = L.Concat(deconv1, deconv2, deconv3)
    top_prev = n['rpn_out']
    #############################MODE3##########################################


    num_cls = 2
    n['cls_preds'] = L.Convolution(top_prev, name = "cls_head",
                         convolution_param=dict(num_output=num_cls,
                                                kernel_size=1, stride=1, pad=0,
                                                weight_filler=dict(type = 'xavier'),
                                                bias_term = True,
                                                bias_filler=dict(type='constant', value=0),
                                                engine=1,
                                                ),
                         param=[dict(lr_mult=1), dict(lr_mult=1)])
    cls_preds = n['cls_preds']


    box_code_size = 7
    num_anchor_per_loc = 2
    n['box_preds'] = L.Convolution(top_prev, name = "reg_head",
                          convolution_param=dict(num_output=num_anchor_per_loc * box_code_size,
                                                 kernel_size=1, stride=1, pad=0,
                                                 weight_filler=dict(type = 'xavier'),
                                                 bias_term = True,
                                                 bias_filler=dict(type='constant', value=0),
                                                 engine=1,
                                                 ),
                          param=[dict(lr_mult=1), dict(lr_mult=1)])
    box_preds = n['box_preds']

    if phase == "train":

        n['cared'],n['reg_outside_weights'], n['cls_weights']= L.Python(n.labels,
                                                                        name = "PrepareLossWeight",
                                                                        ntop = 3,
                                                                        python_param=dict(
                                                                                    module='bcl_layers',
                                                                                    layer='PrepareLossWeight'
                                                                                    ))
        reg_outside_weights, cared, cls_weights = n['reg_outside_weights'], n['cared'], n['cls_weights']

        # Gradients cannot be computed with respect to the label inputs (bottom[1])#
        n['labels_input'] = L.Python(n.labels, cared,
                            name = "Label_Encode",
                            python_param=dict(
                                        module='bcl_layers',
                                        layer='LabelEncode',
                                        ))
        labels_input = n['labels_input']


        n['cls_preds_permute'] = L.Permute(cls_preds, permute_param=dict(order=[0, 2, 3, 1])) #(B,C,H,W) -> (B,H,W,C)
        cls_preds_permute = n['cls_preds_permute']
        n['cls_preds_reshape'] = L.Reshape(cls_preds_permute, reshape_param=dict(shape=dict(dim=[0, -1, 1])))# (B,H,W,C) -> (B, -1, C)
        cls_preds_reshape = n['cls_preds_reshape']


        n.cls_loss= L.Python(cls_preds_reshape, labels_input, cls_weights,
                                name = "FocalLoss",
                                loss_weight = 1,
                                python_param=dict(
                                            module='bcl_layers',
                                            layer='WeightFocalLoss'
                                            ),
                                param_str=str(dict(focusing_parameter=2, alpha=0.25)))

        box_code_size = 7
        n['box_preds_permute'] = L.Permute(box_preds, permute_param=dict(order=[0, 2, 3, 1])) #(B,C,H,W) -> (B,H,W,C)
        box_preds_permute = n['box_preds_permute']
        n['box_preds_reshape'] = L.Reshape(box_preds_permute, reshape_param=dict(shape=dict(dim=[0, -1, box_code_size]))) #(B,H,W,C) -> (B, -1, C)
        box_preds_reshape = n['box_preds_reshape']

        n.reg_loss= L.Python(box_preds_reshape, n.reg_targets, reg_outside_weights,
                                name = "WeightedSmoothL1Loss",
                                loss_weight = 1,
                                python_param=dict(
                                            module='bcl_layers',
                                            layer='WeightedSmoothL1Loss'
                                            ))

        return n.to_proto()

    elif phase == "eval":

        n['e7'],n['m7'],n['h7'],n['e5'],n['m5'],n['h5']=L.Python(box_preds,cls_preds,
                                                    n.anchors, n.rect,
                                                    n.trv2c, n.p2, n.anchors_mask,
                                                    n.img_idx, n.img_shape,
                                                    name = "EvalLayer",
                                                    ntop=6,
                                                    python_param=dict(
                                                    module='bcl_layers',
                                                    layer='EvalLayer_v2',
                                                    param_str=repr(dataset_params_eval),
                                                    ))


        return n.to_proto()

    else:
        raise ValueError
