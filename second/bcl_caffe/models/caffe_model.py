from functools import reduce
import caffe
from caffe import layers as L, params as P, to_proto
from bcl_caffe.utils import get_prototxt, parse_channel_scale, map_channel_scale
import gc

def conv_bn_relu(bottom, ks, nout, stride=1, pad=0):
    conv = L.Convolution(bottom,
                        convolution_param=dict(
                                kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad,
                                engine=2,
                                weight_filler=dict(type = 'xavier'),
                                bias_term = False),
                                param=[dict(lr_mult=1)],
                                )
    bn = L.BatchNorm(conv, batch_norm_param=dict(eps=1e-3, moving_average_fraction=0.99))
    sc = L.Scale(bn, scale_param=dict(bias_term=True))

    return L.ReLU(sc)

def deconv_bn_relu(bottom, ks, nout, stride=1, pad=0):
    deconv = L.Deconvolution(bottom,
                            convolution_param=dict(kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad,
                                engine=2,
                                weight_filler=dict(type = 'xavier'),
                                bias_term = False),
                                param=[dict(lr_mult=1)],
                                )
    bn = L.BatchNorm(deconv, batch_norm_param=dict(eps=1e-3, moving_average_fraction=0.99))
    sc = L.Scale(bn, scale_param=dict(bias_term=True))

    return L.ReLU(sc)

def test_v1(phase,
            dataset_params=None,
            deploy=False,
            create_prototxt=True,
            save_path=None):

    n = caffe.NetSpec()

    if phase == "train":

        dataset_params_train = dataset_params.copy()
        dataset_params_train['subset'] = phase

        datalayer_train = L.Python(name='data', include=dict(phase=caffe.TRAIN),
                                   ntop= 4, python_param=dict(module='custom_layers', layer='InputKittiData',
                                                     param_str=repr(dataset_params_train)))

        n.data, n.coors, n.labels, n.reg_targets = datalayer_train

    elif phase == "eval":
        dataset_params_eval = dataset_params.copy()
        dataset_params_eval['subset'] = phase

        datalayer_eval = L.Python(name='data', include=dict(phase=caffe.TEST),
                                  ntop= 9, python_param=dict(module='custom_layers', layer='InputKittiData',
                                                     param_str=repr(dataset_params_eval)))

        n.data, n.coors, n.anchors, n.rect, n.trv2c, n.p2, n.anchors_mask, n.img_idx, n.img_shape = datalayer_eval

    if deploy:
        print("[debug] run deploy in caffe_model.py")
        # n.data = L.Input(shape=dict(dim=[1, len(input_dims), 1, sample_size]))
        # n.coors = L.Input(shape=dict(dim=[1, len(input_dims), 1, sample_size]))
        # n.reg_targets = L.Input(shape=dict(dim=[1, len(input_dims), 1, sample_size]))

    # top_prev = n.data

    top_prev = conv_bn_relu(n.data, 1, 64, stride=1, pad=0)

    # n['Mlp'] = L.Convolution(top_prev,
    #                      convolution_param=dict(num_output=64,
    #                                             kernel_size=1, stride=1, pad=0,
    #                                             weight_filler=dict(type = 'xavier'),
    #                                             bias_term = False,
    #                                             engine=2,
    #                                             ),
    #                      param=[dict(lr_mult=1, decay_mult=1)])
    # idx = 1
    # top_prev = n['Mlp']
    # n['bn'+str(idx)] = L.BatchNorm(top_prev, batch_norm_param=dict(eps=1e-3, moving_average_fraction=0.99))
    # top_prev = n['bn'+str(idx)]
    # n['sc'+str(idx)] = L.Scale(top_prev, scale_param=dict(bias_term=True))
    # top_prev = n['sc'+str(idx)]
    # n['relu'+str(idx)] = L.ReLU(top_prev) #in_place=True
    # top_prev = n['relu'+str(idx)]

    n['max_pool'] = L.Pooling(top_prev, pooling_param = dict(kernel_h=1, kernel_w=100, stride=1, pad=0,
                                 pool = caffe.params.Pooling.MAX)) #(1,64,voxel,1)

    top_prev = n['max_pool']

    n['PillarScatter'] = L.Python(top_prev, n.coors, python_param=dict(
                                                module='custom_layers',
                                                layer='PointPillarsScatter',
                                                param_str=str(dict(output_shape=[1, 1, 496, 432, 64],
                                                                ))))
    top_prev = n['PillarScatter']

    ###RPN layer
    num_filters = [64,128,256]
    layer_nums = [3,5,5]
    layer_strides = [2, 2, 2]
    num_upsample_filters = [128, 128, 128]
    upsample_strides = [1, 2, 4]


    ##############################init-1 cov w/2, h/2#############################
    n['init_conv1'] = L.Convolution(top_prev,
                                     convolution_param=dict(num_output=num_filters[0],
                                                            kernel_size=3, stride=layer_strides[0], pad=1,
                                                            weight_filler=dict(type = 'xavier'),
                                                            bias_term = False,
                                                            engine=2,
                                                            ),
                                     param=[dict(lr_mult=1)])
    idx = 2
    top_prev = n['init_conv1']
    n['bn'+str(idx)] = L.BatchNorm(top_prev, batch_norm_param=dict(eps=1e-3, moving_average_fraction=0.99))
    top_prev = n['bn'+str(idx)]
    n['sc'+str(idx)] = L.Scale(top_prev, scale_param=dict(bias_term=True))
    top_prev = n['sc'+str(idx)]
    n['relu'+str(idx)] = L.ReLU(top_prev)
    top_prev = n['relu'+str(idx)]

    #ini_conv1 = conv_bn_relu(top_prev, 3, num_filters[0], stride=layer_strides[0], pad=1)

    #for idx, _ in enumerate(range(layer_nums[0])):
    idx = 3
    n['rpn_conv1_' + str(idx)] = L.Convolution(top_prev,
                                         convolution_param=dict(num_output=num_filters[0],
                                                                kernel_size=3, stride=1, pad=1,
                                                                weight_filler=dict(type = 'xavier'),
                                                                bias_term = False,
                                                                engine=2,
                                                                ),
                                         param=[dict(lr_mult=1)])
    top_prev = n['rpn_conv1_' + str(idx)]
    n['bn'+str(idx)] = L.BatchNorm(top_prev, batch_norm_param=dict(eps=1e-3, moving_average_fraction=0.99))
    top_prev = n['bn'+str(idx)]
    n['sc'+str(idx)] = L.Scale(top_prev, scale_param=dict(bias_term=True))
    top_prev = n['sc'+str(idx)]
    n['relu'+str(idx)] = L.ReLU(top_prev)
    top_prev = n['relu'+str(idx)]

    # conv1 = conv_bn_relu(ini_conv1, 3, num_filters[0], stride=1, pad=1)

    ################################deconv1_start##############################1
    idx = 4
    n['rpn_deconv1'] = L.Deconvolution(top_prev,
                                         convolution_param=dict(num_output=num_upsample_filters[0],
                                                                kernel_size=upsample_strides[0], stride=upsample_strides[0], pad=0,
                                                                weight_filler=dict(type = 'xavier'),
                                                                bias_term = False,
                                                                engine=2,
                                                                ),
                                         param=[dict(lr_mult=1)])
    top_prev = n['rpn_deconv1']
    n['bn'+str(idx)] = L.BatchNorm(top_prev, batch_norm_param=dict(eps=1e-3, moving_average_fraction=0.99))
    top_prev = n['bn'+str(idx)]
    n['sc'+str(idx)] = L.Scale(top_prev, scale_param=dict(bias_term=True))
    top_prev = n['sc'+str(idx)]
    n['relu'+str(idx)] = L.ReLU(top_prev)
    top_prev = n['relu'+str(idx)]

    #deconv1 = deconv_bn_relu(conv1, upsample_strides[0], num_upsample_filters[0], stride=upsample_strides[0], pad=0)

    #################################deconv1-end###############################1

    ##############################init-2 cov w/2, h/2#############################
    # n['init_conv2'] = L.Convolution(top_prev,
    #                                      convolution_param=dict(num_output=num_filters[1],
    #                                                             kernel_size=3, stride=layer_strides[1], pad=1,
    #                                                             weight_filler=dict(type = 'xavier'),
    #                                                             bias_term = False,
    #                                                             engine=1,
    #                                                             ),
    #                                      param=[dict(lr_mult=1)])
    #
    # top_prev = L.BatchNorm(n['init_conv2'], batch_norm_param=dict(eps=1e-3, moving_average_fraction=0.01))
    # top_prev = L.Scale(top_prev, scale_param=dict(bias_term=True))
    # top_prev = L.ReLU(top_prev)#, relu_param=dict(negative_slope = 0.1))

    #ini_conv2 = conv_bn_relu(conv1, 3, num_filters[1], stride=layer_strides[1], pad=1)

    # for idx, _ in enumerate(range(layer_nums[1])):
    #     n['rpn_conv2_' + str(idx)] = L.Convolution(top_prev,
    #                                          convolution_param=dict(num_output=num_filters[1],
    #                                                                 kernel_size=3, stride=1, pad=1,
    #                                                                 weight_filler=dict(type = 'xavier'),
    #                                                                 bias_term = False,
    #                                                                 engine=1,
    #                                                                 ),
    #                                          param=[dict(lr_mult=1)])
    #
    #     top_prev = L.BatchNorm(n['rpn_conv2_' + str(idx)], batch_norm_param=dict(eps=1e-3, moving_average_fraction=0.01))
    #     top_prev = L.Scale(top_prev, scale_param=dict(bias_term=True))
    #     top_prev = L.ReLU(top_prev)# , relu_param=dict(negative_slope = 0.1))

    #conv2 = conv_bn_relu(ini_conv2, 3, num_filters[1], stride=1, pad=1)

    ################################deconv2_start##############################2
    # n['rpn_deconv2'] = L.Deconvolution(top_prev,
    #                                      convolution_param=dict(num_output=num_upsample_filters[1],
    #                                                             kernel_size=upsample_strides[1], stride=upsample_strides[1], pad=0,
    #                                                             weight_filler=dict(type = 'xavier'),
    #                                                             bias_term = False,
    #                                                             engine=1,
    #                                                             ),
    #                                      param=[dict(lr_mult=1)])
    #
    # deconv2 = L.BatchNorm(n['rpn_deconv2'], batch_norm_param=dict(eps=1e-3, moving_average_fraction=0.01))
    # deconv2 = L.Scale(deconv2, scale_param=dict(bias_term=True))
    # deconv2 = L.ReLU(deconv2)# , relu_param=dict(negative_slope = 0.1))

    #deconv2 = deconv_bn_relu(conv2, upsample_strides[1], num_upsample_filters[1], stride=upsample_strides[1], pad=0)

    #################################deconv2-end###############################2

    ##############################init-3 cov w/2, h/2#############################
    # n['init_conv3'] = L.Convolution(top_prev,
    #                                      convolution_param=dict(num_output=num_filters[2],
    #                                                             kernel_size=3, stride=layer_strides[2], pad=1,
    #                                                             weight_filler=dict(type = 'xavier'),
    #                                                             bias_term = False,
    #                                                             engine=1,
    #                                                             ),
    #                                      param=[dict(lr_mult=1)])
    #
    # top_prev = L.BatchNorm(n['init_conv3'], batch_norm_param=dict(eps=1e-3, moving_average_fraction=0.01))
    # top_prev = L.Scale(top_prev, scale_param=dict(bias_term=True))
    # top_prev = L.ReLU(top_prev)#, relu_param=dict(negative_slope = 0.1))

    #ini_conv3 = conv_bn_relu(conv2, 3, num_filters[2], stride=layer_strides[2], pad=1)

    # # ##
    # for idx, _ in enumerate(range(layer_nums[2])):
    #     n['rpn_conv3_' + str(idx)] = L.Convolution(top_prev,
    #                                          convolution_param=dict(num_output=num_filters[2],
    #                                                                 kernel_size=3, stride=1, pad=1,
    #                                                                 weight_filler=dict(type = 'xavier'),
    #                                                                 bias_term = False,
    #                                                                 engine=1,
    #                                                                 ),
    #                                          param=[dict(lr_mult=1)])
    #
    #     top_prev = L.BatchNorm(n['rpn_conv3_' + str(idx)], batch_norm_param=dict(eps=1e-3, moving_average_fraction=0.01))
    #     top_prev = L.Scale(top_prev, scale_param=dict(bias_term=True))
    #     top_prev = L.ReLU(top_prev)#, relu_param=dict(negative_slope = 0.1))

    #conv3 = conv_bn_relu(ini_conv3, 3, num_filters[2], stride=1, pad=1)

    ################################deconv3_start##############################3
    # n['rpn_deconv3'] = L.Deconvolution(top_prev,
    #                                      convolution_param=dict(num_output=num_upsample_filters[2],
    #                                                             kernel_size=upsample_strides[2], stride=upsample_strides[2], pad=0,
    #                                                             weight_filler=dict(type = 'xavier'),
    #                                                             bias_term = False,
    #                                                             engine=1,
    #                                                             ),
    #                                      param=[dict(lr_mult=1)])
    #
    # deconv3 = L.BatchNorm(n['rpn_deconv3'], batch_norm_param=dict(eps=1e-3, moving_average_fraction=0.01))
    # deconv3 = L.Scale(deconv3, scale_param=dict(bias_term=True))
    # deconv3 = L.ReLU(deconv3)#, relu_param=dict(negative_slope = 0.1))

    #deconv3 = deconv_bn_relu(conv3, upsample_strides[2], num_upsample_filters[2], stride=upsample_strides[2], pad=0)

    # ##
    #n['rpn_out'] = L.Concat(deconv1, deconv2, deconv3)
    # top_prev = n['rpn_out']

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
                                                                                    module='custom_layers',
                                                                                    layer='PrepareLossWeight'
                                                                                    ))
        reg_outside_weights, cared, cls_weights = n['reg_outside_weights'], n['cared'], n['cls_weights']

        # Gradients cannot be computed with respect to the label inputs (bottom[1])#
        n['labels_input'] = L.Python(n.labels, cared,
                            name = "Label_Encode",
                            python_param=dict(
                                        module='custom_layers',
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
                                            module='custom_layers',
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
                                loss_weight = 2,
                                python_param=dict(
                                            module='custom_layers',
                                            layer='WeightedSmoothL1Loss'
                                            ))

        return n.to_proto()

    elif phase == "eval":

        n['iou'] = L.Python(box_preds,
                            cls_preds,
                            n.anchors, n.rect,
                            n.trv2c, n.p2, n.anchors_mask,
                            n.img_idx, n.img_shape,
                            name = "EvalLayer",
                            python_param=dict(
                            module='custom_layers',
                            layer='EvalLayer_v2',
                            param_str=repr(dataset_params_eval),
                            ))


        return n.to_proto()
    else:
        raise ValueError
