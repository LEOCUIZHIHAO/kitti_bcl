from functools import reduce
import caffe
from caffe import layers as L, params as P, to_proto
from bcl_caffe.utils import get_prototxt, parse_channel_scale, map_channel_scale
import gc

# datalayer_test = L.Python(name='data', include=dict(phase=caffe.TEST), ntop=0,
#                           top= ['data', 'coors', 'labels', 'reg_targets'],
#                           python_param=dict(module='custom_layers', layer='InputKittiData',
#                                             param_str=repr(dataset_params_test)))

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

    top_prev = n.data
    for idx, x in enumerate(range(1)):
        n['Mlp'] = L.Convolution(top_prev,
                             convolution_param=dict(num_output=64,
                                                    kernel_size=1, stride=1, pad=0,
                                                    weight_filler=dict(type = 'xavier', std = 0.01),
                                                    bias_term = False,
                                                    #bias_filler=dict(type='constant', value=0),
                                                    engine=1,
                                                    ),
                             param=[dict(lr_mult=1)])
        # top_prev = L.Reshape(top_prev, reshape_param=dict(shape=dict(dim=[0,0,-1,1])))

        # n['Mlp'] = L.InnerProduct(top_prev,
        #                      inner_product_param=dict(num_output=64,
        #                                             weight_filler=dict(type = 'xavier'), #,std = 0.1
        #                                             bias_term = False,
        #                                             #bias_filler=dict(type='constant', value=0),
        #                                             ),
        #                      param=[dict(lr_mult=1)])

        top_prev = L.BatchNorm(n['Mlp'])
        top_prev = L.ReLU(top_prev)

        top_prev = L.Python(top_prev, name = "Test_Layer",
                                        ntop = 1,
                                        python_param=dict(
                                                    module='custom_layers',
                                                    layer='TestLayer')
                                                    )

    top_prev = L.Pooling(top_prev, pooling_param = dict(kernel_h=1, kernel_w=100, stride=1,
                                 pool = caffe.params.Pooling.MAX))

    n['PillarScatter'] = L.Python(top_prev, n.coors, python_param=dict(
                                                module='custom_layers',
                                                layer='PointPillarsScatter',
                                                param_str=str(dict(output_shape=[1, 1, 496, 432, 64],
                                                               num_input_features = 64
                                                                ))))
    ###RPN layer
    num_filters = [64,128,256]
    layer_nums = [3,5,5]
    layer_strides = [2, 2, 2]
    num_upsample_filters = [128, 128, 128]
    upsample_strides = [1, 2, 4]

    ##############################init-1 cov w/2, h/2#############################
    n['init_conv1'] = L.Convolution(n['PillarScatter'],
                                         convolution_param=dict(num_output=num_filters[0],
                                                                kernel_size=3, stride=layer_strides[0], pad=1,
                                                                weight_filler=dict(type = 'xavier',std = 0.1),
                                                                bias_term = False,
                                                                #bias_filler=dict(type='constant', value=0),
                                                                engine=1,
                                                                ),
                                         param=[dict(lr_mult=1)]) #0.1

    top_prev = L.BatchNorm(n['init_conv1'])
    top_prev = L.ReLU(top_prev)

    for idx, _ in enumerate(range(layer_nums[0])):
        n['rpn_conv1_' + str(idx)] = L.Convolution(top_prev,
                                             convolution_param=dict(num_output=num_filters[0],
                                                                    kernel_size=3, stride=1, pad=1,
                                                                    weight_filler=dict(type = 'xavier',std = 0.1),
                                                                    bias_term = False,
                                                                    #bias_filler=dict(type='constant', value=0),
                                                                    engine=1,
                                                                    ),
                                             param=[dict(lr_mult=1)]) #0.1

        top_prev = L.BatchNorm(n['rpn_conv1_' + str(idx)])
        top_prev = L.ReLU(top_prev)

    ################################deconv1_start##############################1
    n['rpn_deconv1'] = L.Deconvolution(top_prev,
                                         convolution_param=dict(num_output=num_upsample_filters[0],
                                                                kernel_size=upsample_strides[0], stride=upsample_strides[0], pad=0,
                                                                weight_filler=dict(type = 'xavier',std = 0.1),
                                                                bias_term = False,
                                                                #bias_filler=dict(type='constant', value=0),
                                                                engine=1,
                                                                ),
                                         param=[dict(lr_mult=1)])

    deconv1 = L.BatchNorm(n['rpn_deconv1'])
    deconv1 = L.ReLU(deconv1)

    #################################deconv1-end###############################1

    ##############################init-2 cov w/2, h/2#############################
    n['init_conv2'] = L.Convolution(top_prev,
                                         convolution_param=dict(num_output=num_filters[1],
                                                                kernel_size=3, stride=layer_strides[1], pad=1,
                                                                weight_filler=dict(type = 'xavier',std = 0.1),
                                                                bias_term = False,
                                                                #bias_filler=dict(type='constant', value=0),
                                                                engine=1,
                                                                ),
                                         param=[dict(lr_mult=1)])

    top_prev = L.BatchNorm(n['init_conv2'])
    top_prev = L.ReLU(top_prev)

    for idx, _ in enumerate(range(layer_nums[1])):
        n['rpn_conv2_' + str(idx)] = L.Convolution(top_prev,
                                             convolution_param=dict(num_output=num_filters[1],
                                                                    kernel_size=3, stride=1, pad=1,
                                                                    weight_filler=dict(type = 'xavier',std = 0.1),
                                                                    bias_term = False,
                                                                    #bias_filler=dict(type='constant', value=0),
                                                                    engine=1,
                                                                    ),
                                             param=[dict(lr_mult=1)])

        top_prev = L.BatchNorm(n['rpn_conv2_' + str(idx)])
        top_prev = L.ReLU(top_prev)

    ################################deconv2_start##############################2
    n['rpn_deconv2'] = L.Deconvolution(top_prev,
                                         convolution_param=dict(num_output=num_upsample_filters[1],
                                                                kernel_size=upsample_strides[1], stride=upsample_strides[1], pad=0,
                                                                weight_filler=dict(type = 'xavier',std = 0.1),
                                                                bias_term = False,
                                                                #bias_filler=dict(type='constant', value=0),
                                                                engine=1,
                                                                ),
                                         param=[dict(lr_mult=1)])

    deconv2 = L.BatchNorm(n['rpn_deconv2'])
    deconv2 = L.ReLU(deconv2)

    #################################deconv2-end###############################2

    ##############################init-3 cov w/2, h/2#############################
    n['init_conv3'] = L.Convolution(top_prev,
                                         convolution_param=dict(num_output=num_filters[2],
                                                                kernel_size=3, stride=layer_strides[2], pad=1,
                                                                weight_filler=dict(type = 'xavier',std = 0.1),
                                                                bias_term = False,
                                                                #bias_filler=dict(type='constant', value=0),
                                                                engine=1,
                                                                ),
                                         param=[dict(lr_mult=1)])

    top_prev = L.BatchNorm(n['init_conv3'])
    top_prev = L.ReLU(top_prev)

    # ##
    for idx, _ in enumerate(range(layer_nums[2])):
        n['rpn_conv3_' + str(idx)] = L.Convolution(top_prev,
                                             convolution_param=dict(num_output=num_filters[2],
                                                                    kernel_size=3, stride=1, pad=1,
                                                                    weight_filler=dict(type = 'xavier',std = 0.1),
                                                                    bias_term = False,
                                                                    #bias_filler=dict(type='constant', value=0),
                                                                    engine=1,
                                                                    ),
                                             param=[dict(lr_mult=1)])

        top_prev = L.BatchNorm(n['rpn_conv3_' + str(idx)])
        top_prev = L.ReLU(top_prev)

    ################################deconv3_start##############################3
    n['rpn_deconv3'] = L.Deconvolution(top_prev,
                                         convolution_param=dict(num_output=num_upsample_filters[2],
                                                                kernel_size=upsample_strides[2], stride=upsample_strides[2], pad=0,
                                                                weight_filler=dict(type = 'xavier',std = 0.1),
                                                                bias_term = False,
                                                                #bias_filler=dict(type='constant', value=0),
                                                                engine=1,
                                                                ),
                                         param=[dict(lr_mult=1)])

    deconv3 = L.BatchNorm(n['rpn_deconv3'])
    deconv3 = L.ReLU(deconv3)

    # ##
    n['rpn_out'] = L.Concat(deconv1, deconv2, deconv3)
    del deconv1, deconv2, deconv3, top_prev

    num_cls = 2
    n['cls_preds'] = L.Convolution(n['rpn_out'], name = "cls_head",
                             convolution_param=dict(num_output=num_cls,
                                                    kernel_size=1, stride=1, pad=0,
                                                    weight_filler=dict(type = 'xavier',std = 0.1),
                                                    bias_term = True,
                                                    # bias_filler=dict(type='constant', value=0),
                                                    engine=1,
                                                    ),
                             param=[dict(lr_mult=1), dict(lr_mult=1)])


    n['cls_preds'] = L.Python(n['cls_preds'],
                                    name = "Pred_Reshape",
                                    python_param=dict(
                                    module='custom_layers',
                                    layer='PredReshape',
                                    ))

    box_code_size = 7
    num_anchor_per_loc = 2
    n['box_preds'] = L.Convolution(n['rpn_out'], name = "reg_head",
                              convolution_param=dict(num_output=num_anchor_per_loc * box_code_size,
                                                     kernel_size=1, stride=1, pad=0,
                                                     weight_filler=dict(type = 'xavier',std = 0.1),
                                                     bias_term = True,
                                                     #bias_filler=dict(type='constant', value=0),
                                                     engine=1,
                                                     ),
                              param=[dict(lr_mult=1), dict(lr_mult=2)])

    if phase == "eval":

        n['box_preds'] = L.Python(n['box_preds'],
                                        name = "BoxPredReshape",
                                        python_param=dict(
                                        module='custom_layers',
                                        layer='BoxPredReshape',
                                        ))

        n['iou'] = L.Python(n['box_preds'],
                            n['cls_preds'],
                            n.anchors, n.rect,
                            n.trv2c, n.p2, n.anchors_mask,
                            n.img_idx, n.img_shape,
                            name = "EvalLayer",
                            python_param=dict(
                            module='custom_layers',
                            layer='EvalLayer',
                            param_str=repr(dataset_params_eval),
                            ))

        # return to_proto(n['box_preds'])
        return n.to_proto()

    elif phase == "train":

        n['box_preds'], n['_reg_targets'] = L.Python(n['box_preds'], n.reg_targets,
                                                        name = "RegLossCreate_train",
                                                        ntop =2,
                                                        python_param=dict(
                                                        module='custom_layers',
                                                        layer='RegLossCreate',
                                                        ))

        n['cared'], n['reg_outside_weights'], n['reg_inside_weights'] = L.Python(n.labels,
                                                                            name = "PrepareLossWeight",
                                                                            ntop = 3,
                                                                            python_param=dict(
                                                                                        module='custom_layers',
                                                                                        layer='PrepareLossWeight'
                                                                                        ))

        # Gradients cannot be computed with respect to the label inputs (bottom[1])#
        n['_labels'] = L.Python(n.labels, n['cared'], name = "Label_Encode",
                                                                    python_param=dict(
                                                                                module='custom_layers',
                                                                                layer='LabelEncode',
                                                                                ))

        # n['cls_preds'], n['_labels'] = L.Python(n['cls_preds'], n['_labels'],name = "Test_Layer",
        #                                             ntop = 2,
        #                                             python_param=dict(
        #                                                         module='custom_layers',
        #                                                         layer='TestLayer')
        #                                                         )

        n['cls_loss'] = L.FocalLoss(n['cls_preds'], n['_labels'],
                                    loss_weight = 1, #loss_param = dict(ignore_label=0, normalize=True),
                                    focal_loss_param=dict(axis=1, alpha=0.25, gamma=2.0)
                                    )

        n['reg_loss'] = L.SmoothL1Loss(n['box_preds'], n['_reg_targets'],
                                        n['reg_inside_weights'], n['reg_outside_weights'],
                                        loss_weight = 2, smooth_l1_loss_param=dict(sigma=1)
                                        ) # smoothl1 =  w_out * SmoothL1(w_in * (b0 - b1))


        return n.to_proto()
        # return to_proto(n['reg_loss'], n['cls_loss'])

    else:
        raise ValueError

    # if create_prototxt:
    #     net = get_prototxt(net, save_path)
