from functools import reduce
import caffe
from caffe import layers as L, params as P
from bcl_caffe.utils import get_prototxt, parse_channel_scale, map_channel_scale

"""
#############################################################################
from second.bcl_caffe.utils import get_paddings_indicator_caffe
from second.builder import target_assigner_builder, voxel_builder
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                      lr_scheduler_builder, optimizer_builder,
                                      second_builder)
from second.protos import pipeline_pb2
from google.protobuf import text_format #make prototxt work
from collections import defaultdict # for merge data to batch
import pathlib
import shutil
import numpy as np
#############################################################################

def merge_second_batch(batch_list):
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    ret = {}
    example_merged.pop("num_voxels")
    for key, elems in example_merged.items():
        if key in [
                'voxels', 'num_points', 'num_gt', 'gt_boxes', 'voxel_labels',
                'match_indices'
        ]:
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'match_indices_num':
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'coordinates':
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)),
                    mode='constant',
                    constant_values=i)
                coors.append(coor_pad)
            ret[key] = np.concatenate(coors, axis=0)
        else:
            ret[key] = np.stack(elems, axis=0)
    return ret

def prep_and_read_kitti(dataset_params_train):

    params = dict(subset='train', batch_size=2)
    params.update(dataset_params_train)

    model_dir = params['model_dir']
    config_path = params['config_path']
    batch_size = params['batch_size']

    create_folder = False
    result_path = None
    print("[config_path]", config_path)

    if create_folder:
        if pathlib.Path(model_dir).exists():
            model_dir = torchplus.train.create_folder(model_dir)

    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_checkpoint_dir = model_dir / 'eval_checkpoints'
    eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if result_path is None:
        result_path = model_dir / 'results'
    config_file_bkp = "pipeline.config"
    config = pipeline_pb2.TrainEvalPipelineConfig()


    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    shutil.copyfile(config_path, str(model_dir / config_file_bkp))

    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config
    ######################
    # BUILD VOXEL GENERATOR
    ######################
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    ######################
    # BUILD TARGET ASSIGNER
    ######################
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)

    dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)

    eval_dataset = input_reader_builder.build(
        eval_input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)

    data_iter = iter(dataset)

    # merge batch
    example_batch = []
    for _ in range(batch_size):
        example = next(data_iter)
        print("[debug] image_idx" , example["image_idx"])
        example_batch.append(example)

    ret = merge_second_batch(example_batch)

    # grid_size = voxel_generator.grid_size
    # dense_shape = [1] + grid_size[::-1].tolist() + [64]

    return ret
"""

def create_loss(loc_loss_ftor,
                cls_loss_ftor,
                box_preds,
                cls_preds,
                cls_targets,
                cls_weights,
                reg_targets,
                reg_weights,
                num_class,
                encode_background_as_zeros=True,
                encode_rad_error_by_sin=True,
                box_code_size=7):
    batch_size = int(box_preds.shape[0])
    box_preds = box_preds.view(batch_size, -1, box_code_size)
    if encode_background_as_zeros:
        cls_preds = cls_preds.view(batch_size, -1, num_class)
    else:
        cls_preds = cls_preds.view(batch_size, -1, num_class + 1)
    cls_targets = cls_targets.squeeze(-1)
    one_hot_targets = torchplus.nn.one_hot(
        cls_targets, depth=num_class + 1, dtype=box_preds.dtype)
    if encode_background_as_zeros:
        one_hot_targets = one_hot_targets[..., 1:]
    if encode_rad_error_by_sin:
        # sin(a - b) = sinacosb-cosasinb
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets)
    loc_losses = loc_loss_ftor(   # in losses_builder.build(model_cfg.loss)
        box_preds, reg_targets, weights=reg_weights)  # [N, M]
    cls_losses = cls_loss_ftor(
        cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
    return loc_losses, cls_losses

def add_sin_difference(boxes1, boxes2):
    rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(
        boxes2[..., -1:])
    rad_tg_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])
    boxes1 = torch.cat([boxes1[..., :-1], rad_pred_encoding], dim=-1)
    boxes2 = torch.cat([boxes2[..., :-1], rad_tg_encoding], dim=-1)
    return boxes1, boxes2

def test_v1(arch_str='64_128_256_256', batchnorm=True,
                                    skip_str=(),  # tuple of strings like '4_1_ga' - relu4 <- relu1 w/ options 'ga'
                                    bilateral_nbr=1,
                                    conv_weight_filler='xavier', bltr_weight_filler='gauss_0.001',
                                    dataset='kitti', dataset_params_train=None,
                                    sample_size=3000, batch_size=32,
                                    feat_dims_str='x_y_z', lattice_dims_str=None,
                                    renorm_class=False,
                                    deploy=False, create_prototxt=True, save_path=None):

    n = caffe.NetSpec()

    arch_str = [(v[0], int(v[1:])) if v[0] in {'b', 'c'} else ('c', int(v)) for v in arch_str.split('_')]
    num_bltr_layers = sum(v[0] == 'b' for v in arch_str)

    if num_bltr_layers > 0:
        if type(lattice_dims_str) == str:
            lattice_dims_str = (lattice_dims_str,) * num_bltr_layers
        elif len(lattice_dims_str) == 1:
            lattice_dims_str = lattice_dims_str * num_bltr_layers
        else:
            assert len(lattice_dims_str) == num_bltr_layers, '{} lattices should be provided'.format(num_bltr_layers)
        #print("[debug] lattice_dims_str -2 ", lattice_dims_str)
        #print(type(lattice_dims_str))
        feat_dims = parse_channel_scale(feat_dims_str, channel_str=True)[0]
        lattice_dims = [parse_channel_scale(s, channel_str=True)[0] for s in lattice_dims_str]
        input_dims_w_dup = feat_dims + reduce(lambda x, y: x + y, lattice_dims)
        input_dims = reduce(lambda x, y: x if y in x else x + [y], input_dims_w_dup, [])
        feat_dims_str = map_channel_scale(feat_dims_str, input_dims)
        lattice_dims_str = [map_channel_scale(s, input_dims) for s in lattice_dims_str]
        input_dims_str = '_'.join(input_dims)
    else:
        feat_dims = parse_channel_scale(feat_dims_str, channel_str=True)[0]
        input_dims = feat_dims
        feat_dims_str = map_channel_scale(feat_dims_str, input_dims)
        input_dims_str = '_'.join(input_dims)

    # dataset specific settings: nclass, datalayer_train, datalayer_test
    if dataset == 'kitti':
        nclass = 2 # # TODO: change config here

        # dataset_params_train = dataset_params.copy()
        dataset_params_train['subset'] = 'train'

        # data layers
        datalayer_train = L.Python(name='data', include=dict(phase=caffe.TRAIN),
                                   ntop= 4,
                                   python_param=dict(module='custom_layers', layer='InputKittiData',
                                                     param_str=repr(dataset_params_train)))
        # # # # TODO:  change evl path
        datalayer_test = L.Python(name='data', include=dict(phase=caffe.TEST), ntop=0,
                                  top= ['data', 'coors', 'labels', 'reg_targets'],
                                  python_param=dict(module='custom_layers', layer='InputKittiData',
                                                    param_str=repr(dataset_params_train)))
    else:
        raise ValueError('Dataset {} unknown'.format(dataset))

    # Input/Data layer
    if deploy:
        datalayer_train = L.Python(name='data', include=dict(phase=caffe.TRAIN),
                                   ntop= 4,
                                   python_param=dict(module='custom_layers', layer='InputKittiData',
                                                     param_str=repr(dataset_params_train)))
        # # # # TODO:  change evl path
        datalayer_test = L.Python(name='data', include=dict(phase=caffe.TEST), ntop=0,
                                  top= ['data', 'coors', 'labels', 'reg_targets'],
                                  python_param=dict(module='custom_layers', layer='InputKittiData',
                                                    param_str=repr(dataset_params_train)))
        n.data, n.coors, n.labels, n.reg_targets = datalayer_train
        n.test_data = datalayer_test
        # datalayer_train = L.Python(name='data', include=dict(phase=caffe.TRAIN),
        #                            ntop=3 if renorm_class else 2,
        #                            python_param=dict(module='custom_layers', layer='InputKittiData',
        #                                              param_str=repr(dataset_params_train)))

        # n.data, n.coors = L.Input(ntop=2 , shape=dict(dim=[1, len(input_dims), 1, sample_size])) # outputs (except labels) must equal datalayer_train
        # n.data = L.Input(shape=dict(dim=[1, len(input_dims), 1, sample_size]))
    else:
        n.data, n.coors, n.labels, n.reg_targets = datalayer_train
        n.test_data = datalayer_test
    # print(n.tops.keys())

    top_prev = n.data
    for idx, x in enumerate(range(1)):
        n['Mlp'] = L.Convolution(top_prev,
                                             convolution_param=dict(num_output=64,
                                                                    kernel_size=1, stride=1, pad=0,
                                                                    weight_filler=dict(type = 'xavier',std = 0.1),
                                                                    bias_filler=dict(type='constant', value=0),
                                                                    engine=1,
                                                                    ),
                                             param=[dict(lr_mult=1), dict(lr_mult=0.1)])

        top_prev = L.BatchNorm(n['Mlp'])
        top_prev = L.ReLU(top_prev)

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
    num_upsample_filters = [128, 128, 128]
    upsample_strides = [1, 2, 4]

    ##############################init cov w/2, h/2#############################
    n['init_conv1'] = L.Convolution(n['PillarScatter'],
                                         convolution_param=dict(num_output=num_filters[0],
                                                                kernel_size=3, stride=2, pad=1,
                                                                weight_filler=dict(type = 'xavier',std = 0.1),
                                                                bias_filler=dict(type='constant', value=0),
                                                                engine=1,
                                                                ),
                                         param=[dict(lr_mult=1), dict(lr_mult=0.1)])

    top_prev = L.BatchNorm(n['init_conv1'])
    top_prev = L.ReLU(top_prev)

    ##############################init cov w/2, h/2#############################

    for idx, _ in enumerate(range(layer_nums[0])):
        n['rpn_conv1_' + str(idx)] = L.Convolution(top_prev,
                                             convolution_param=dict(num_output=num_filters[0],
                                                                    kernel_size=3, stride=1, pad=1,
                                                                    weight_filler=dict(type = 'xavier',std = 0.1),
                                                                    bias_filler=dict(type='constant', value=0),
                                                                    engine=1,
                                                                    ),
                                             param=[dict(lr_mult=1), dict(lr_mult=0.1)])

        top_prev = L.BatchNorm(n['rpn_conv1_' + str(idx)])
        top_prev = L.ReLU(top_prev)

    ################################deconv1_start##############################1
    n['rpn_deconv1'] = L.Deconvolution(top_prev,
                                         convolution_param=dict(num_output=num_upsample_filters[0],
                                                                kernel_size=upsample_strides[0], stride=upsample_strides[0], pad=0,
                                                                weight_filler=dict(type = 'xavier',std = 0.1),
                                                                bias_filler=dict(type='constant', value=0),
                                                                engine=1,
                                                                ),
                                         param=[dict(lr_mult=1), dict(lr_mult=0.1)])

    deconv1 = L.BatchNorm(n['rpn_deconv1'])
    deconv1 = L.ReLU(deconv1)

    #################################deconv1-end###############################1

    ##############################init cov w/2, h/2#############################
    n['init_conv2'] = L.Convolution(top_prev,
                                         convolution_param=dict(num_output=num_filters[0],
                                                                kernel_size=3, stride=2, pad=1,
                                                                weight_filler=dict(type = 'xavier',std = 0.1),
                                                                bias_filler=dict(type='constant', value=0),
                                                                engine=1,
                                                                ),
                                         param=[dict(lr_mult=1), dict(lr_mult=0.1)])

    top_prev = L.BatchNorm(n['init_conv2'])
    top_prev = L.ReLU(top_prev)
    ##############################init cov w/2, h/2#############################

    for idx, _ in enumerate(range(layer_nums[1])):
        n['rpn_conv2_' + str(idx)] = L.Convolution(top_prev,
                                             convolution_param=dict(num_output=num_filters[1],
                                                                    kernel_size=3, stride=1, pad=1,
                                                                    weight_filler=dict(type = 'xavier',std = 0.1),
                                                                    bias_filler=dict(type='constant', value=0),
                                                                    engine=1,
                                                                    ),
                                             param=[dict(lr_mult=1), dict(lr_mult=0.1)])

        top_prev = L.BatchNorm(n['rpn_conv2_' + str(idx)])
        top_prev = L.ReLU(top_prev)

    ################################deconv2_start##############################2
    n['rpn_deconv2'] = L.Deconvolution(top_prev,
                                         convolution_param=dict(num_output=num_upsample_filters[1],
                                                                kernel_size=upsample_strides[1], stride=upsample_strides[1], pad=0,
                                                                weight_filler=dict(type = 'xavier',std = 0.1),
                                                                bias_filler=dict(type='constant', value=0),
                                                                engine=1,
                                                                ),
                                         param=[dict(lr_mult=1), dict(lr_mult=0.1)])

    deconv2 = L.BatchNorm(n['rpn_deconv2'])
    deconv2 = L.ReLU(deconv2)

    #################################deconv2-end###############################2

    ##############################init cov w/2, h/2#############################
    n['init_conv3'] = L.Convolution(top_prev,
                                         convolution_param=dict(num_output=num_filters[0],
                                                                kernel_size=3, stride=2, pad=1,
                                                                weight_filler=dict(type = 'xavier',std = 0.1),
                                                                bias_filler=dict(type='constant', value=0),
                                                                engine=1,
                                                                ),
                                         param=[dict(lr_mult=1), dict(lr_mult=0.1)])

    top_prev = L.BatchNorm(n['init_conv3'])
    top_prev = L.ReLU(top_prev)

    # ##
    for idx, _ in enumerate(range(layer_nums[2])):
        n['rpn_conv3_' + str(idx)] = L.Convolution(top_prev,
                                             convolution_param=dict(num_output=num_filters[2],
                                                                    kernel_size=3, stride=1, pad=1,
                                                                    weight_filler=dict(type = 'xavier',std = 0.1),
                                                                    bias_filler=dict(type='constant', value=0),
                                                                    engine=1,
                                                                    ),
                                             param=[dict(lr_mult=1), dict(lr_mult=0.1)])

        top_prev = L.BatchNorm(n['rpn_conv3_' + str(idx)])
        top_prev = L.ReLU(top_prev)

    ################################deconv3_start##############################3
    n['rpn_deconv3'] = L.Deconvolution(top_prev,
                                         convolution_param=dict(num_output=num_upsample_filters[2],
                                                                kernel_size=upsample_strides[2], stride=upsample_strides[2], pad=0,
                                                                weight_filler=dict(type = 'xavier',std = 0.1),
                                                                bias_filler=dict(type='constant', value=0),
                                                                engine=1,
                                                                ),
                                         param=[dict(lr_mult=1), dict(lr_mult=0.1)])

    deconv3 = L.BatchNorm(n['rpn_deconv3'])
    deconv3 = L.ReLU(deconv3)

    # ##

    n['rpn_out'] = L.Concat(deconv1, deconv2, deconv3)
    del deconv1, deconv2, deconv3, top_prev

    num_cls = 2
    n['cls_preds'] = L.Convolution(n['rpn_out'],
                             convolution_param=dict(num_output=num_cls,
                                                    kernel_size=1, stride=1, pad=0,
                                                    weight_filler=dict(type = 'xavier',std = 0.1),
                                                    bias_filler=dict(type='constant', value=0),
                                                    engine=1,
                                                    ),
                             param=[dict(lr_mult=1), dict(lr_mult=0.1)])

    box_code_size = 7
    num_anchor_per_loc = 2
    n['box_preds'] = L.Convolution(n['rpn_out'],
                              convolution_param=dict(num_output=num_anchor_per_loc * box_code_size,
                                                     kernel_size=1, stride=1, pad=0,
                                                     weight_filler=dict(type = 'xavier',std = 0.1),
                                                     bias_filler=dict(type='constant', value=0),
                                                     engine=1,
                                                     ),
                              param=[dict(lr_mult=1), dict(lr_mult=0.1)])


    n['cls_weights'], n['reg_weights'], n['cared'] = L.Python(n.labels,
                                                            name = "PrepareLossWeight",
                                                            ntop = 3,
                                                            python_param=dict(
                                                                        module='custom_layers',
                                                                        layer='PrepareLossWeight',
                                                                        ))

    # n['label_']  = L.Python(n.labels, name = "TestLayer", ntop=1, python_param=dict(
    #                                                                                             module='custom_layers',
    #                                                                                             layer='TestLayer',
    #                                                                                             ))
    n['cls_preds'], n['_labels'] = L.Python(n['cls_preds'], n.labels, n['cared'],
                                                                    name = "ClsLossCreate",
                                                                    ntop=2,
                                                                    python_param=dict(
                                                                                module='custom_layers',
                                                                                layer='ClsLossCreate',
                                                                                ))
    #
    n['box_preds'], n['_reg_targets'] = L.Python(n['box_preds'], n.reg_targets,
                                                        name = "RegLossCreate",
                                                        ntop=2,
                                                        python_param=dict(
                                                        module='custom_layers',
                                                        layer='RegLossCreate',
                                                        ))
    #
    n['cls_loss'] = L.FocalLoss(n['cls_preds'], n['_labels'])
    n['reg_loss'] = L.SmoothL1Loss(n['box_preds'], n['_reg_targets'])

    # n['cls_loss'] = L.FocalLoss(n['_labels'], n['cls_preds'])
    # n['reg_loss'] = L.SmoothL1Loss(n['_reg_targets'], n['box_preds'])
    # n['Reg_Loss'] = L.Python(n.reg_targets, n['box_preds'], n.labels,
    #                     name='Reg_Loss',
    #                     python_param=dict(module='custom_layers', layer='WeightedSmoothL1LocalizationLoss'))
    #
    # n['Cls_Loss'] = L.Python(n.labels, n['cls_preds'],
    #                     name='Cls_Loss',
    #                     python_param=dict(module='custom_layers', layer='SigmoidFocalClassificationLoss'))


    net = n.to_proto()

    if create_prototxt:
        net = get_prototxt(net, save_path)

    # n.data_feat = L.Python(n.data, python_param=dict(module='custom_layers', layer='PickAndScale',
    #                                                  param_str=feat_dims_str))
    # top_prev = n.data_feat
    #
    # if conv_weight_filler in {'xavier', 'msra'}:
    #     conv_weight_filler = dict(type=conv_weight_filler)
    # elif conv_weight_filler.startswith('gauss_'):
    #     conv_weight_filler = dict(type='gaussian', std=float(conv_weight_filler.split('_')[1]))
    # else:
    #     conv_weight_filler = eval(conv_weight_filler)
    # assert bltr_weight_filler.startswith('gauss_')
    # bltr_weight_filler = dict(type='gaussian', std=float(bltr_weight_filler.split('_')[1]))
    #
    # # multiple 1x1 conv-(bn)-relu blocks, optionally with a single global pooling somewhere among them
    #
    # idx = 1
    # bltr_idx = 0
    # lattices = dict()
    # last_in_block = dict()
    # for (layer_type, n_out) in arch_str:
    #     if layer_type == 'c':
    #         n['conv' + str(idx)] = L.Convolution(top_prev,
    #                                              convolution_param=dict(num_output=n_out,
    #                                                                     kernel_size=1, stride=1, pad=0,
    #                                                                     weight_filler=conv_weight_filler,
    #                                                                     bias_filler=dict(type='constant', value=0)),
    #                                              param=[dict(lr_mult=1), dict(lr_mult=0.1)])
    #     elif layer_type == 'b':
    #         lattice_dims_str_curr = lattice_dims_str[bltr_idx]
    #         if lattice_dims_str_curr in lattices:
    #             top_data_lattice, top_lattice = lattices[lattice_dims_str_curr]
    #             n['conv' + str(idx)] = L.Permutohedral(top_prev, top_data_lattice, top_data_lattice, top_lattice,
    #                                                    permutohedral_param=dict(num_output=n_out,
    #                                                                             group=1,
    #                                                                             neighborhood_size=bilateral_nbr,
    #                                                                             bias_term=True,
    #                                                                             norm_type=P.Permutohedral.AFTER,
    #                                                                             offset_type=P.Permutohedral.NONE,
    #                                                                             filter_filler=bltr_weight_filler,
    #                                                                             bias_filler=dict(type='constant',
    #                                                                                              value=0)),
    #                                                    param=[{'lr_mult': 1, 'decay_mult': 1},
    #                                                           {'lr_mult': 2, 'decay_mult': 0}])
    #         else:
    #             top_data_lattice = L.Python(n.data, python_param=dict(module='custom_layers', layer='PickAndScale',
    #                                                                   param_str=lattice_dims_str_curr))
    #             n['data_lattice' + str(len(lattices))] = top_data_lattice
    #             if lattice_dims_str.count(lattice_dims_str_curr) > 1:
    #                 n['conv' + str(idx)], top_lattice = L.Permutohedral(top_prev, top_data_lattice, top_data_lattice,
    #                                                                     ntop=2,
    #                                                                     permutohedral_param=dict(
    #                                                                         num_output=n_out,
    #                                                                         group=1,
    #                                                                         neighborhood_size=bilateral_nbr,
    #                                                                         bias_term=True,
    #                                                                         norm_type=P.Permutohedral.AFTER,
    #                                                                         offset_type=P.Permutohedral.NONE,
    #                                                                         filter_filler=bltr_weight_filler,
    #                                                                         bias_filler=dict(type='constant',
    #                                                                                          value=0)),
    #                                                                     param=[{'lr_mult': 1, 'decay_mult': 1},
    #                                                                            {'lr_mult': 2, 'decay_mult': 0}])
    #                 n['lattice' + str(len(lattices))] = top_lattice
    #             else:
    #                 n['conv' + str(idx)] = L.Permutohedral(top_prev, top_data_lattice, top_data_lattice,
    #                                                        permutohedral_param=dict(
    #                                                            num_output=n_out,
    #                                                            group=1,
    #                                                            neighborhood_size=bilateral_nbr,
    #                                                            bias_term=True,
    #                                                            norm_type=P.Permutohedral.AFTER,
    #                                                            offset_type=P.Permutohedral.NONE,
    #                                                            filter_filler=bltr_weight_filler,
    #                                                            bias_filler=dict(type='constant', value=0)),
    #                                                        param=[{'lr_mult': 1, 'decay_mult': 1},
    #                                                               {'lr_mult': 2, 'decay_mult': 0}])
    #                 top_lattice = None
    #
    #             lattices[lattice_dims_str_curr] = (top_data_lattice, top_lattice)
    #
    #         bltr_idx += 1
    #
    #     top_prev = n['conv' + str(idx)]
    #     if batchnorm:
    #         n['bn'+str(idx)] = L.BatchNorm(top_prev)
    #         top_prev = n['bn'+str(idx)]
    #     n['relu'+str(idx)] = L.ReLU(top_prev, in_place=True)
    #     top_prev = n['relu'+str(idx)]
    #
    #     # skip connection & global pooling
    #     if skip_str is None:
    #         skip_str = ()
    #     skip_tos = [v.split('_')[0] for v in skip_str]
    #     if str(idx) in skip_tos:
    #         skip_idxs = list(filter(lambda i: skip_tos[i] == str(idx), range(len(skip_tos))))
    #         skip_params = [skip_str[i].split('_') for i in skip_idxs]
    #         if len(skip_params[0]) == 2:
    #             assert all(len(v) == 2 for v in skip_params)
    #         else:
    #             assert all(v[2] == skip_params[0][2] for v in skip_params)
    #
    #         if len(skip_params[0]) > 2 and 'g' in skip_params[0][2]:  # global pooling on current layer
    #             n['gpool'+str(idx)] = L.Python(top_prev,
    #                                            python_param=dict(module='custom_layers', layer='GlobalPooling'))
    #             top_prev = n['gpool'+str(idx)]
    #
    #         if len(skip_params[0]) > 2 and 'a' in skip_params[0][2]:  # addition instead of concatenation
    #             n['add'+str(idx)] = L.Eltwise(top_prev, *[last_in_block[int(v[1])] for v in skip_params],
    #                                           eltwise_param=dict(operation=P.Eltwise.SUM))
    #             top_prev = n['add'+str(idx)]
    #         else:
    #             n['concat'+str(idx)] = L.Concat(top_prev, *[last_in_block[int(v[1])] for v in skip_params])
    #             top_prev = n['concat'+str(idx)]
    #
    #     last_in_block[idx] = top_prev
    #     idx += 1
    #
    # classification & loss
    # n['conv'+str(idx)] = L.Convolution(top_prev,
    #                                    convolution_param=dict(num_output=2, kernel_size=1, stride=1, pad=0,
    #                                                           weight_filler=dict(type = 'xavier',std = 0.1),
    #                                                           bias_filler=dict(type='constant', value=0)),
    #                                    param=[dict(lr_mult=1), dict(lr_mult=0.1)])
    # top_prev = n['conv'+str(idx)]
    #
    # if renorm_class:
    #     if deploy:
    #         n.prob = L.Softmax(top_prev)
    #     else:
    #         n.prob_raw = L.Softmax(top_prev)
    #         n.prob = L.Python(n.prob_raw, n.label_mask, python_param=dict(module='custom_layers', layer='ProbRenorm'))
    #         n.loss = L.Python(n.prob, n.label, python_param=dict(module='custom_layers', layer='LogLoss'), loss_weight=1)
    #         n.accuracy = L.Accuracy(n.prob, n.label)
    # else:
    #     if deploy:
    #         n.prob = L.Softmax(top_prev)
    #     else:
    #         n.loss = L.SoftmaxWithLoss(top_prev, n.label)
    #         n.accuracy = L.Accuracy(top_prev, n.label)

    # net = n.to_proto()
    #
    # if create_prototxt:
    #     net = get_prototxt(net, save_path)

    return net
