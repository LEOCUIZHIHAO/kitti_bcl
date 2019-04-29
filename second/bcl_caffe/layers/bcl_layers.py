import numpy as np
import caffe
import pathlib
import shutil
import timeit
import numba
from second.bcl_caffe.utils import get_paddings_indicator_caffe
from second.builder import target_assigner_builder, voxel_builder
from second.bcl_caffe.builder import input_reader_builder, box_coder_builder_caffe
from second.bcl_caffe.core import box_caffe_ops

from second.pytorch.core import box_torch_ops #for torch
from second.pytorch.builder import box_coder_builder

from second.utils.eval import get_coco_eval_result, get_official_eval_result
import second.data.kitti_common as kitti

from second.protos import pipeline_pb2
from google.protobuf import text_format #make prototxt work
from collections import defaultdict # for merge data to batch
from enum import Enum

import torch
import torch.utils.data
import gc

# import time
# from functools import partial

class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"

class InputKittiData(caffe.Layer):

    def setup(self, bottom, top):

        params = dict(batch_size=1)
        params.update(eval(self.param_str))

        self.model_dir = params['model_dir']
        self.config_path = params['config_path']
        self.batch_size = params['batch_size']
        self.phase = params['subset']
        self.example_batch = []

        # ########################################################################
        ## TODO:  pass by param
        point_cloud_range = [0, -40, -3, 70.4, 40, 1]
        voxel_size = [0.2, 0.2, 4]
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]

        #shuffle index
        # self.index_list = np.arange(3712)
        # np.random.shuffle(self.index_list)
        # self.iter = iter(self.index_list)
        self.cfg = self.load_config()
        self.data = iter(self.load_data(self.cfg))

        for _ in range(self.batch_size):
            # index = self.index_list[next(self.iter, None)]
            # if index == None:
            #     np.random.shuffle(self.index_list)
            #     self.iter = iter(self.index_list)
            #     index = self.index_list[next(self.iter)]
            example = next(self.data)
            self.example_batch.append(example)

        example = self.merge_second_batch(self.example_batch)
        # example = np.load("./pillar_data.npy").item() ## DEBUG:
        ########################################################################

        self.example_batch = [] #reset example_batch
        voxels = example['voxels']
        coors = example['coordinates']
        num_points = example['num_points']

        features = self.PillarFeatureNet(voxels, coors, num_points)

        self.data = iter(self.load_data(self.cfg))

        top[0].reshape(*features.shape) #[1,9,7000,100]
        top[1].reshape(*coors.shape) #[7000,4]

        if self.phase == 'train':
            labels = example['labels']
            reg_targets =example['reg_targets']
            top[2].reshape(*labels.shape) #[1 107136]
            top[3].reshape(*reg_targets.shape) #[]

        elif self.phase == 'eval':
            batch_size = example['anchors'].shape[0]
            batch_anchors = example["anchors"].reshape(batch_size, -1, 7)
            batch_rect = example["rect"]
            batch_Trv2c = example["Trv2c"]
            batch_P2 = example["P2"]
            if "anchors_mask" not in example:
                batch_anchors_mask = [None] * batch_size
            else:
                batch_anchors_mask = example["anchors_mask"].reshape(batch_size, -1)
            batch_imgidx = example['image_idx']
            batch_image_shape = example['image_shape']
            top[2].reshape(*batch_anchors.shape)
            top[3].reshape(*batch_rect.shape)
            top[4].reshape(*batch_Trv2c.shape)
            top[5].reshape(*batch_P2.shape)
            top[6].reshape(*batch_anchors_mask.shape)
            top[7].reshape(*batch_imgidx.shape)
            top[8].reshape(*batch_image_shape.shape)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        for _ in range(self.batch_size):
            # index = self.index_list[next(self.iter, None)]
            # if index == None:cfg = dict(input_cfg=input_cfg)
            #     np.random.shuffle(self.index_list)
            #     self.iter = iter(self.index_list)
            #     index = self.index_list[next(self.iter)]
            try:
                example = next(self.data)
            except StopIteration:
                print("[info]>>>>>>>>>>>>>>>>>>> start a new epoch for {} data ".format(self.phase))
                # self.data = iter(self.load_data())
                self.data = iter(self.load_data(self.cfg))
                example = next(self.data)

            self.example_batch.append(example)

        example = self.merge_second_batch(self.example_batch)
        # example = np.load("./pillar_data.npy").item() ## DEBUG:

        self.example_batch = [] #reset example_batch
        voxels = example['voxels']
        coors = example['coordinates']
        num_points = example['num_points']

        features = self.PillarFeatureNet(voxels, coors, num_points)

        top[0].reshape(*features.shape) #[1,9,7000,100]
        top[1].reshape(*coors.shape) #[7000,4]
        top[0].data[...] = features
        top[1].data[...] = coors

        if self.phase == 'train':
            labels = example['labels']
            reg_targets =example['reg_targets']
            top[2].reshape(*labels.shape) #[1 107136]
            top[3].reshape(*reg_targets.shape) #[]
            top[2].data[...] = labels
            top[3].data[...] = reg_targets
            # print("[debug] train img idx : ", example["image_idx"])

        elif self.phase == 'eval':

            batch_size = example['anchors'].shape[0]
            batch_anchors = example["anchors"].reshape(batch_size, -1, 7)
            batch_rect = example["rect"]
            batch_Trv2c = example["Trv2c"]
            batch_P2 = example["P2"]
            if "anchors_mask" not in example:
                batch_anchors_mask = [None] * batch_size
            else:
                batch_anchors_mask = example["anchors_mask"].reshape(batch_size, -1)
            batch_imgidx = example['image_idx']
            batch_image_shape = example['image_shape']
            # print("[debug] eval img idx : ", batch_imgidx)

            ###################################################################
            top[2].reshape(*batch_anchors.shape)
            top[3].reshape(*batch_rect.shape)
            top[4].reshape(*batch_Trv2c.shape)
            top[5].reshape(*batch_P2.shape)
            top[6].reshape(*batch_anchors_mask.shape)
            top[7].reshape(*batch_imgidx.shape)
            top[8].reshape(*batch_image_shape.shape)
            top[2].data[...] = batch_anchors
            top[3].data[...] = batch_rect
            top[4].data[...] = batch_Trv2c
            top[5].data[...] = batch_P2
            top[6].data[...] = batch_anchors_mask
            top[7].data[...] = batch_imgidx
            top[8].data[...] = batch_image_shape

    def backward(self, top, propagate_down, bottom):
        pass

    def load_config(self):

        model_dir = pathlib.Path(self.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        config_file_bkp = "pipeline.config"
        config = pipeline_pb2.TrainEvalPipelineConfig()

        with open(self.config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
        shutil.copyfile(self.config_path, str(model_dir / config_file_bkp))

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
        box_coder = box_coder_builder_caffe.build(model_cfg.box_coder)
        target_assigner_cfg = model_cfg.target_assigner
        target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                        bv_range, box_coder)


        return (input_cfg, eval_input_cfg, model_cfg, voxel_generator, target_assigner)

    def load_data(self, cfg):

        input_cfg, eval_input_cfg, model_cfg, \
            voxel_generator, target_assigner = cfg[0],cfg[1],cfg[2],cfg[3],cfg[4]

        if self.phase == 'train':
            dataset = input_reader_builder.build(
                input_cfg,
                model_cfg,
                training=True,
                voxel_generator=voxel_generator,
                target_assigner=target_assigner)

            return dataset

        elif self.phase == 'eval':
            eval_dataset = input_reader_builder.build(
                eval_input_cfg,
                model_cfg,
                training=False,
                voxel_generator=voxel_generator,
                target_assigner=target_assigner)

            return eval_dataset

        else:
            raise ValueError

    def PillarFeatureNet(self, voxels, coors, num_points):
        #for permutohedral may not need extrat featuers
        """
        points_mean = np.sum(voxels[:, :, :3], axis=1, keepdims=True) / num_points.reshape(-1,1,1)
        f_cluster = voxels[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center = voxels[:, :, :2]
        f_center[:, :, 0] = f_center[:, :, 0] - (np.expand_dims(coors[:, 3].astype(float), axis=1) * self.vx + self.x_offset)
        f_center[:, :, 1] = f_center[:, :, 1] - (np.expand_dims(coors[:, 2].astype(float), axis=1) * self.vy + self.y_offset)

        features_ls = [voxels, f_cluster, f_center]
        features = np.concatenate(features_ls, axis=-1) #[num_voxles, points_num, features]
        """

        features = voxels

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.

        points_per_voxels = features.shape[1]
        mask = get_paddings_indicator_caffe(num_points, points_per_voxels, axis=0)
        mask = np.expand_dims(mask, axis=-1)
        features *= mask

        #(voxel, npoint, channel) -> (channel, voxels, npoints)
        features = np.expand_dims(features, axis=0)
        features = features.transpose(0,3,1,2)

        return features

    def merge_second_batch(self, batch_list):
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

class PointPillarsScatter(caffe.Layer):

    def setup(self, bottom, top):

        param = eval(self.param_str)
        output_shape = param['output_shape']

        self.ny = output_shape[2]
        self.nx = output_shape[3]
        self.nchannels = output_shape[4]
        self.batch_size = 1
        self.coords_nchannels = 3

        voxel_features = bottom[0].data
        voxel_features = np.squeeze(voxel_features) #(1, 64, voxel, 1) -> (64,Voxel)
        coords = bottom[1].data # reverse_index is True, output coordinates will be zyx format

        batch_canvas = []
        batch_canvas_coords = []
        for batch_itt in range(self.batch_size):
            # Create the canvas for this sample
            canvas = np.zeros(shape=(self.nchannels, self.nx * self.ny))
            canvas_coords = np.zeros(shape=(self.coords_nchannels, self.nx * self.ny)) #for coords canvas

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.astype(int)
            voxels = voxel_features[:, batch_mask]

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels
            canvas_coords[:, indices] = this_coords.transpose()[1:,:][::-1,:] #need to be xyz!

            # Append to a list for later stacking.
            batch_canvas.append(canvas)
            batch_canvas_coords.append(canvas_coords)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = np.stack(batch_canvas, 0)
        batch_canvas_coords = np.stack(batch_canvas_coords, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.reshape(self.batch_size, self.nchannels, self.ny, self.nx)
        batch_canvas_coords = batch_canvas_coords.reshape(self.batch_size, self.coords_nchannels, self.ny, self.nx)

        top[0].reshape(*batch_canvas.shape)
        top[1].reshape(*batch_canvas_coords.shape)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):

        voxel_features = bottom[0].data #(1,64,-1,1)
        voxel_features = np.squeeze(voxel_features) #(1, 64, -1, 1) -> (64,-1)
        coords = bottom[1].data

        batch_canvas = []
        batch_canvas_coords = []
        for batch_itt in range(self.batch_size):
            # Create the canvas for this sample
            canvas = np.zeros(shape=(self.nchannels, self.nx * self.ny)) #(64,-1)
            canvas_coords = np.zeros(shape=(self.coords_nchannels, self.nx * self.ny)) #for coords canvas

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            self.indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            self.indices = self.indices.astype(int)
            voxels = voxel_features[:, batch_mask]

            # Now scatter the blob back to the canvas.
            canvas[:, self.indices] = voxels
            canvas_coords[:, self.indices] = this_coords.transpose()[1:,:][::-1,:] #need to be xyz!
            # Append to a list for later stacking.
            batch_canvas.append(canvas)
            batch_canvas_coords.append(canvas_coords)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = np.stack(batch_canvas, 0)
        batch_canvas_coords = np.stack(batch_canvas_coords, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.reshape(self.batch_size, self.nchannels, self.ny, self.nx)
        batch_canvas_coords = batch_canvas_coords.reshape(self.batch_size, self.coords_nchannels, self.ny, self.nx)

        batch_canvas_coords

        top[0].data[...] = batch_canvas
        top[1].data[...] = batch_canvas_coords

    def backward(self, top, propagate_down, bottom):
        diff = top[0].diff.reshape(self.batch_size, self.nchannels, self.nx * self.ny)[:,:,self.indices]
        bottom[0].diff[...] = np.expand_dims(diff, axis=-1)

class PrepareLossWeight(caffe.Layer):
    def setup(self, bottom, top):

        labels = bottom[0].data
        cls_weights, reg_weights, cared = self.prepare_loss_weights(labels)

        top[0].reshape(*cared.shape)
        top[1].reshape(*reg_weights.shape) #reg_outside_weights
        top[2].reshape(*cls_weights.shape)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):

        labels = bottom[0].data
        cls_weights, reg_weights, cared = self.prepare_loss_weights(labels)

        top[0].data[...] = cared
        top[1].data[...] = reg_weights #reg_outside_weights
        top[2].data[...] = cls_weights

    def prepare_loss_weights(self,
                            labels,
                            pos_cls_weight=1.0, # TODO: pass params here
                            neg_cls_weight=1.0,
                            loss_norm_type=LossNormType.NormByNumPositives,
                            dtype="float32"):
        """get cls_weights and reg_weights from labels.
        """
        cared = labels >= 0
        # print("label ", np.unique(labels, return_counts=True))
        # cared: [N, num_anchors]
        positives = labels > 0
        negatives = labels == 0
        negative_cls_weights = negatives.astype(dtype) * neg_cls_weight
        posetive_cls_weights = positives.astype(dtype) * pos_cls_weight #(1, 107136)
        cls_weights = negative_cls_weights + posetive_cls_weights
        reg_weights = positives.astype(dtype)

        if loss_norm_type == LossNormType.NormByNumExamples:
            num_examples = cared.astype(dtype).sum(1, keepdims=True)
            num_examples = np.clip(num_examples, a_min=1.0, a_max=None)
            cls_weights /= num_examples
            bbox_normalizer = np.sum(positives, 1, keepdims=True).astype(dtype)
            reg_weights /= np.clip(bbox_normalizer, a_min=1.0, a_max=None)
        elif loss_norm_type == LossNormType.NormByNumPositives:  # for focal loss
            pos_normalizer = np.sum(positives, 1, keepdims=True).astype(dtype)
            reg_weights /= np.clip(pos_normalizer, a_min=1.0, a_max=None) #(1, 107136)
            cls_weights /= np.clip(pos_normalizer, a_min=1.0, a_max=None) #(1, 107136)

        elif loss_norm_type == LossNormType.NormByNumPosNeg:
            pos_neg = np.stack([positives, negatives], a_min=-1).astype(dtype)
            normalizer = np.sum(pos_neg, 1, keepdims=True)  # [N, 1, 2]
            cls_normalizer = np.sum((pos_neg * normalizer),-1)  # [N, M]
            cls_normalizer = np.clip(cls_normalizer, a_min=1.0, a_max=None)
            # cls_normalizer will be pos_or_neg_weight/num_pos_or_neg
            normalizer = np.clip(normalizer, a_min=1.0, a_max=None)
            reg_weights /= normalizer[:, 0:1, 0]
            cls_weights /= cls_normalizer

        else:
            raise ValueError(
                f"unknown loss norm type. available: {list(LossNormType)}")
        return cls_weights, reg_weights, cared

    def backward(self, top, propagate_down, bottom):
        pass

class LabelEncode(caffe.Layer):
    def setup(self, bottom, top):

        labels = bottom[0].data
        cared = bottom[1].data

        cls_targets = labels * cared # (1, 107136)
        cls_targets = cls_targets.astype(int)

        self.num_class = 1

        one_hot_targets = np.eye(self.num_class+1)[cls_targets]   #One_hot label -- make sure one hot class is <num_class+1>
        one_hot_targets = one_hot_targets[..., 1:]

        top[0].reshape(*one_hot_targets.shape) #reshape to caffe pattern

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):

        labels = bottom[0].data # (1, 107136)
        cared = bottom[1].data

        cls_targets = labels * cared
        cls_targets = cls_targets.astype(int)


        one_hot_targets = np.eye(self.num_class+1)[cls_targets]   #One_hot label -- make sure one hot class is <num_class+1>
        one_hot_targets = one_hot_targets[..., 1:]

        top[0].data[...] = one_hot_targets

    def backward(self, top, propagate_down, bottom):
        pass

class WeightFocalLoss(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        # if len(bottom) != 2:
        #     raise Exception("Need two inputs to compute distance (inference and labels).")

        params = eval(self.param_str)
        self.gamma = int(params['focusing_parameter'])
        self.alpha = params['alpha']

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].num != bottom[1].num:
            raise Exception("Infered scores and labels must have the same dimension.")
        top[0].reshape(1)

    def forward(self, bottom, top):
        self._p = bottom[0].data
        self.label = bottom[1].data
        self.cls_weights = bottom[2].data
        self.cls_weights = np.expand_dims(self.cls_weights,-1)

        self._p_t =  1 / (1 + np.exp(-self._p)) # Compute sigmoid activations

        self.first = (1-self.label) * (1-self.alpha) + self.label * self.alpha

        self.second = (1-self.label) * ((self._p_t) ** self.gamma) + self.label * ((1 - self._p_t) ** self.gamma)

        # log1p = np.log(np.exp(-np.abs(self._p))+1)
        log1p = np.log1p(np.exp(-np.abs(self._p)))

        self.sigmoid_cross_entropy = (1-self.label) * (log1p + np.clip(self._p, a_min=0, a_max=None)) + \
                                    self.label * (log1p - np.clip(self._p, a_min=None, a_max=0))

        logprobs = ((1-self.label) * self.first * self.second * self.sigmoid_cross_entropy) + \
                    (self.label * self.first * self.second * self.sigmoid_cross_entropy)

        top[0].data[...] = np.sum(logprobs*self.cls_weights)

    def backward(self, top, propagate_down, bottom):

        # dev_log1p = self._p / ((np.exp(np.abs(self._p))+1) * np.abs(self._p)) #derivitive ln(e^(-abs(x) + 1)
        dev_log1p = np.sign(self._p) * (1 / (np.exp(np.abs(self._p))+1))  # might fix divided by 0 x/|x| bug

        self.dev_sigmoid_cross_entropy =  (1-self.label) * (dev_log1p - np.clip(self._p, a_min=0, a_max=None)/self._p)  + \
                                            self.label * (dev_log1p + np.clip(self._p, a_min=None, a_max=0)/self._p)

        delta = (1-self.label) *  (self.first * self.second * (self.gamma * (1-self._p_t) * self.sigmoid_cross_entropy - self.dev_sigmoid_cross_entropy)) + \
            self.label * (-self.first * self.second * (self.gamma * self._p_t * self.sigmoid_cross_entropy + self.dev_sigmoid_cross_entropy))

        bottom[0].diff[...] = delta * self.cls_weights

class WeightedSmoothL1Loss(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        # if len(bottom) != 2:
        #     raise Exception("Need two inputs to compute distance (inference and labels).")
        self.sigma = 3
        self.encode_rad_error_by_sin = True

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].num != bottom[1].num:
            raise Exception("Infered scores and labels must have the same dimension.")
        top[0].reshape(1)

    def forward(self, bottom, top):
        box_preds = bottom[0].data
        reg_targets = bottom[1].data
        self.reg_weights = bottom[2].data
        self.reg_weights = np.expand_dims(self.reg_weights,-1)

        self.diff = box_preds - reg_targets

        #use sin_difference rad to sin
        if self.encode_rad_error_by_sin:

            diff_rot = self.diff[...,-1:].copy() #copy rotation without add sin
            self.sin_diff = np.sin(diff_rot)
            self.cos_diff = np.cos(diff_rot)
            self.diff[...,-1] = np.sin(self.diff[...,-1]) #use sin_difference

        self.abs_diff = np.abs(self.diff)

        self.cond = self.abs_diff < (1/(self.sigma**2))
        loss = np.where(self.cond, 0.5 * self.sigma**2 * self.abs_diff**2,
                                    self.abs_diff - 0.5/self.sigma**2)

        reg_loss = loss * self.reg_weights

        top[0].data[...] = np.sum(reg_loss) * 2

    def backward(self, top, propagate_down, bottom):

        if self.encode_rad_error_by_sin:

            # delta = np.where(self.cond[...,:-1], (self.sigma**2) * self.diff[...,:-1], self.diff[...,:-1]/self.abs_diff[...,:-1])
            delta = np.where(self.cond[...,:-1], (self.sigma**2) * self.diff[...,:-1], np.sign(self.diff[...,:-1]))

            # delta_rotation = np.where(self.cond[...,-1:], (self.sigma**2) * self.sin_diff * self.cos_diff,
                                    # self.sin_diff/np.abs(self.sin_diff) * self.cos_diff)
            delta_rotation = np.where(self.cond[...,-1:], (self.sigma**2) * self.sin_diff * self.cos_diff,
                                        np.sign(self.sin_diff) * self.cos_diff) #if sign(0) is gonna be 0!!!!!!!!!!!!!!

            delta = np.concatenate([delta, delta_rotation], axis=-1)

        else:

            # delta = np.where(self.cond, (self.sigma**2) * self.diff, self.diff/self.abs_diff)
            delta = np.where(self.cond, (self.sigma**2) * self.diff, np.sign(self.diff))

        bottom[0].diff[...] = delta * self.reg_weights * 2

class EvalLayer(caffe.Layer):

    def setup(self, bottom, top):

        params= eval(self.param_str)
        self.model_dir = params['model_dir']
        self.config_path = params['config_path']

        self.target_assigner, self.gt_annos = self.load_generator()
        self._box_coder = self.target_assigner.box_coder

        self._num_class = 1 ## TODO: pass
        self._encode_background_as_zeros = True
        self._use_direction_classifier = False
        self._use_sigmoid_score = True
        self._use_rotate_nms = False
        self._multiclass_nms = False
        self._nms_score_threshold = 0.05 ## TODO:  double check
        self._nms_pre_max_size = 1000
        self._nms_post_max_size = 300
        self._nms_iou_threshold = 0.5

        self.dt_annos = []

        top[0].reshape(1)

    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):

        batch_box_preds = bottom[0].data
        batch_cls_preds = bottom[1].data
        batch_anchors = bottom[2].data
        batch_rect = bottom[3].data
        batch_Trv2c = bottom[4].data
        batch_P2 = bottom[5].data
        batch_anchors_mask = bottom[6].data.astype(bool)
        batch_imgidx = bottom[7].data
        self.batch_image_shape = bottom[8].data


        batch_size = batch_anchors.shape[0]
        batch_box_preds = batch_box_preds.reshape(batch_size, -1, self._box_coder.code_size)

        batch_anchors_mask = batch_anchors_mask.reshape(batch_size, -1)

        num_class_with_bg = self._num_class
        if not self._encode_background_as_zeros:
            num_class_with_bg = self._num_class + 1

        batch_cls_preds = batch_cls_preds.reshape(batch_size, -1, num_class_with_bg)

        batch_box_preds = self._box_coder.decode_torch(batch_box_preds,
                                                       batch_anchors) ## TODO: double check

        if self._use_direction_classifier:
            batch_dir_preds = preds_dict["dir_cls_preds"]
            batch_dir_preds = batch_dir_preds.reshape(batch_size, -1, 2)
        else:
            batch_dir_preds = [None] * batch_size

        predictions_dicts = []
        for box_preds, cls_preds, dir_preds, rect, Trv2c, P2, img_idx, a_mask in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds, batch_rect,
                batch_Trv2c, batch_P2, batch_imgidx, batch_anchors_mask
        ):
            if a_mask is not None:

                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]

            if self._use_direction_classifier:
                if a_mask is not None:
                    dir_preds = dir_preds[a_mask]
                # print(dir_preds.shape)
                dir_labels = np.max(dir_preds, dim=-1)[1]

            if self._encode_background_as_zeros:
                # this don't support softmax
                assert self._use_sigmoid_score is True
                total_scores = 1 / (1 + np.exp(-cls_preds))## TODO: double check total_scores = torch.sigmoid(cls_preds)

            else:
                # encode background as first element in one-hot vector
                if self._use_sigmoid_score:
                    total_scores = 1 / (1 + np.exp(-cls_preds))[..., 1:]
                else:
                    total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]

            if self._use_rotate_nms:
                nms_func = box_caffe_ops.rotate_nms
            else:
                nms_func = box_caffe_ops.nms
            selected_boxes = None
            selected_labels = None
            selected_scores = None
            selected_dir_labels = None

            if self._multiclass_nms:
                # curently only support class-agnostic boxes.
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                if not self._use_rotate_nms:
                    box_preds_corners = box_torch_ops.center_to_corner_box2d(
                        boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                        boxes_for_nms[:, 4])
                    boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                        box_preds_corners)
                boxes_for_mcnms = boxes_for_nms.unsqueeze(1)
                selected_per_class = box_torch_ops.multiclass_nms(
                    nms_func=nms_func,
                    boxes=boxes_for_mcnms,
                    scores=total_scores,
                    num_class=self._num_class,
                    pre_max_size=self._nms_pre_max_size,
                    post_max_size=self._nms_post_max_size,
                    iou_threshold=self._nms_iou_threshold,
                    score_thresh=self._nms_score_threshold,
                )
                selected_boxes, selected_labels, selected_scores = [], [], []
                selected_dir_labels = []
                for i, selected in enumerate(selected_per_class):
                    if selected is not None:
                        num_dets = selected.shape[0]
                        selected_boxes.append(box_preds[selected])
                        selected_labels.append(
                            torch.full([num_dets], i, dtype=torch.int64))
                        if self._use_direction_classifier:
                            selected_dir_labels.append(dir_labels[selected])
                        selected_scores.append(total_scores[selected, i])
                if len(selected_boxes) > 0:
                    selected_boxes = torch.cat(selected_boxes, dim=0)
                    selected_labels = torch.cat(selected_labels, dim=0)
                    selected_scores = torch.cat(selected_scores, dim=0)
                    if self._use_direction_classifier:
                        selected_dir_labels = torch.cat(
                            selected_dir_labels, dim=0)

            else:
                # get highest score per prediction, than apply nms
                # to remove overlapped box.
                if num_class_with_bg == 1:
                    top_scores = np.squeeze(total_scores, axis=-1)
                    top_labels = np.zeros(total_scores.shape[0])
                else:
                    top_scores, top_labels = np.max(total_scores, axis=-1)

                if self._nms_score_threshold > 0.0:
                    thresh = self._nms_score_threshold
                    top_scores_keep = top_scores >= thresh

                    top_scores = top_scores[top_scores_keep]

                if top_scores.shape[0] != 0:
                    if self._nms_score_threshold > 0.0:
                        box_preds = box_preds[top_scores_keep]
                        if self._use_direction_classifier:
                            dir_labels = dir_labels[top_scores_keep]
                        top_labels = top_labels[top_scores_keep]
                    boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]

                    if not self._use_rotate_nms:
                        box_preds_corners = box_caffe_ops.center_to_corner_box2d(
                            boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                            boxes_for_nms[:, 4])
                        boxes_for_nms = box_caffe_ops.corner_to_standup_nd(
                            box_preds_corners)
                    # the nms in 3d detection just remove overlap boxes.
                    selected = nms_func(
                        boxes_for_nms,
                        top_scores,
                        pre_max_size=self._nms_pre_max_size,
                        post_max_size=self._nms_post_max_size,
                        iou_threshold=self._nms_iou_threshold,
                    )
                    # print("[debug] selected", selected)

                else:
                    selected = None
                if selected is not None:
                    selected_boxes = box_preds[selected]
                    if self._use_direction_classifier:
                        selected_dir_labels = dir_labels[selected]
                    selected_labels = top_labels[selected]
                    selected_scores = top_scores[selected]
                    # print("[debug] selected_scores", selected_scores)

            # finally generate predictions.

            if selected_boxes is not None:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                if self._use_direction_classifier:
                    dir_labels = selected_dir_labels
                    opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.byte()
                    box_preds[..., -1] += torch.where(
                        opp_labels,
                        torch.tensor(np.pi).type_as(box_preds),
                        torch.tensor(0.0).type_as(box_preds))
                    # box_preds[..., -1] += (
                    #     ~(dir_labels.byte())).type_as(box_preds) * np.pi
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds

                final_box_preds_camera = box_caffe_ops.box_lidar_to_camera( ## TODO: double check
                    final_box_preds, rect, Trv2c)
                locs = final_box_preds_camera[:, :3]
                dims = final_box_preds_camera[:, 3:6]
                angles = final_box_preds_camera[:, 6]
                camera_box_origin = [0.5, 1.0, 0.5]
                box_corners = box_caffe_ops.center_to_corner_box3d(
                    locs, dims, angles, camera_box_origin, axis=1)
                box_corners_in_image = box_caffe_ops.project_to_image( ## TODO: double check!!!!!!!!
                    box_corners, P2)
                # box_corners_in_image: [N, 8, 2]
                minxy = np.amin(box_corners_in_image, axis=1)
                maxxy = np.amax(box_corners_in_image, axis=1)

                box_2d_preds = np.concatenate([minxy, maxxy], axis=1)


                predictions_dict = {
                    "bbox": box_2d_preds,
                    "box3d_camera": final_box_preds_camera,
                    "box3d_lidar": final_box_preds,
                    "scores": final_scores,
                    "label_preds": label_preds,
                    "image_idx": img_idx,
                }
            else:
                predictions_dict = {
                    "bbox": None,
                    "box3d_camera": None,
                    "box3d_lidar": None,
                    "scores": None,
                    "label_preds": None,
                    "image_idx": img_idx,
                }
            predictions_dicts.append(predictions_dict)

        self.dt_annos += self.predict_kitti_to_anno(predictions_dicts, self.class_names, self.center_limit_range, self.lidar_input)

        if len(self.dt_annos) == len(self.gt_annos):
            result, mAPbbox, mAPbev, mAP3d, mAPaos = get_official_eval_result(self.gt_annos, self.dt_annos, self.class_names,
                                                                                  return_data=True)
            print(result)

            result = get_coco_eval_result(self.gt_annos, self.dt_annos, self.class_names)

            print(result)

            self.dt_annos = []
            print("[info] empty self.dt_annos :> dt.annos len : ", len(self.dt_annos))

        top[0].reshape(1)
        top[0].data[...]=1
        pass
    def backward(self, top, propagate_down, bottom):
        pass
    def load_generator(self):

        model_dir = pathlib.Path(self.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        config_file_bkp = "pipeline.config"
        config = pipeline_pb2.TrainEvalPipelineConfig()

        with open(self.config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
        shutil.copyfile(self.config_path, str(model_dir / config_file_bkp))

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
        box_coder = box_coder_builder_caffe.build(model_cfg.box_coder)
        target_assigner_cfg = model_cfg.target_assigner
        target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                        bv_range, box_coder)

        #for evaluation
        self.class_names = list(input_cfg.class_names)
        self.center_limit_range = model_cfg.post_center_limit_range
        self.lidar_input = model_cfg.lidar_input

        eval_dataset = input_reader_builder.build(
            eval_input_cfg,
            model_cfg,
            training=False,
            voxel_generator=voxel_generator,
            target_assigner=target_assigner)
        gt_annos = [
            info["annos"] for info in eval_dataset.kitti_infos
        ]

        return target_assigner, gt_annos
    def predict_kitti_to_anno(self,
                            predictions_dicts,
                            class_names,
                            center_limit_range=None,
                            lidar_input=False,
                            global_set=None):
        annos = []
        for i, preds_dict in enumerate(predictions_dicts):
            image_shape = self.batch_image_shape[i]
            img_idx = preds_dict["image_idx"]
            if preds_dict["bbox"] is not None:
                box_2d_preds = preds_dict["bbox"]
                box_preds = preds_dict["box3d_camera"]
                scores = preds_dict["scores"]
                box_preds_lidar = preds_dict["box3d_lidar"]
                # write pred to file
                label_preds = preds_dict["label_preds"]
                # label_preds = np.zeros([box_2d_preds.shape[0]], dtype=np.int32)
                anno = kitti.get_start_result_anno()
                num_example = 0
                for box, box_lidar, bbox, score, label in zip(
                        box_preds, box_preds_lidar, box_2d_preds, scores,
                        label_preds):
                    if not lidar_input:
                        if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                            continue
                        if bbox[2] < 0 or bbox[3] < 0:
                            continue
                    # print(img_shape)
                    if center_limit_range is not None:
                        limit_range = np.array(center_limit_range)
                        if (np.any(box_lidar[:3] < limit_range[:3])
                                or np.any(box_lidar[:3] > limit_range[3:])):
                            continue
                    bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    anno["name"].append(class_names[int(label)])
                    anno["truncated"].append(0.0)
                    anno["occluded"].append(0)
                    anno["alpha"].append(-np.arctan2(-box_lidar[1], box_lidar[0]) +
                                         box[6])
                    anno["bbox"].append(bbox)
                    anno["dimensions"].append(box[3:6])
                    anno["location"].append(box[:3])
                    anno["rotation_y"].append(box[6])
                    if global_set is not None:
                        for i in range(100000):
                            if score in global_set:
                                score -= 1 / 100000
                            else:
                                global_set.add(score)
                                break
                    anno["score"].append(score)

                    num_example += 1
                if num_example != 0:
                    anno = {n: np.stack(v) for n, v in anno.items()}
                    annos.append(anno)
                else:
                    annos.append(kitti.empty_result_anno())
            else:
                annos.append(kitti.empty_result_anno())
            num_example = annos[-1]["name"].shape[0]
            annos[-1]["image_idx"] = np.array(
                [img_idx] * num_example, dtype=np.int64)
        return annos

class EvalLayer_v2(caffe.Layer):

    def setup(self, bottom, top):

        params= eval(self.param_str)
        self.model_dir = params['model_dir']
        self.config_path = params['config_path']

        self.target_assigner, self.gt_annos = self.load_generator()
        self._box_coder = self.target_assigner.box_coder

        self._num_class = 1
        self._use_direction_classifier = False
        self._use_sigmoid_score = True
        self._multiclass_nms = False
        self._nms_score_threshold = 0.05
        self._nms_pre_max_size = 1000
        self._nms_post_max_size = 300
        self._nms_iou_threshold = 0.5

        self.dt_annos = []

    def reshape(self, bottom, top):
        top[0].reshape(1)
        top[1].reshape(1)
        top[2].reshape(1)
        top[3].reshape(1)
        top[4].reshape(1)
        top[5].reshape(1)
    def forward(self, bottom, top):

        batch_box_preds = torch.from_numpy(bottom[0].data)
        batch_cls_preds = torch.from_numpy(bottom[1].data)

        batch_box_preds = batch_box_preds.permute(0, 2, 3, 1).contiguous()
        batch_cls_preds = batch_cls_preds.permute(0, 2, 3, 1).contiguous()

        batch_size = torch.from_numpy(bottom[2].data).shape[0]
        batch_anchors = torch.from_numpy(bottom[2].data)

        batch_rect = torch.from_numpy(bottom[3].data)
        batch_Trv2c = torch.from_numpy(bottom[4].data)
        batch_P2 = torch.from_numpy(bottom[5].data)

        batch_anchors_mask = torch.from_numpy(bottom[6].data)
        batch_anchors_mask = batch_anchors_mask.type(torch.uint8)

        batch_imgidx = torch.from_numpy(bottom[7].data)
        self.batch_image_shape = bottom[8].data

        batch_box_preds = batch_box_preds.view(batch_size, -1, self._box_coder.code_size)
        num_class_with_bg = self._num_class

        batch_cls_preds = batch_cls_preds.view(batch_size, -1, num_class_with_bg)
        batch_box_preds = self._box_coder.decode_torch(batch_box_preds, batch_anchors)

        if self._use_direction_classifier:
            batch_dir_preds = preds_dict["dir_cls_preds"]
            batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)
        else:
            batch_dir_preds = [None] * batch_size

        predictions_dicts = []
        for box_preds, cls_preds, dir_preds, rect, Trv2c, P2, img_idx, a_mask in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds, batch_rect,
                batch_Trv2c, batch_P2, batch_imgidx, batch_anchors_mask
        ):
            if a_mask is not None:
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]

            if self._use_direction_classifier:
                if a_mask is not None:
                    dir_preds = dir_preds[a_mask]
                dir_labels = torch.max(dir_preds, dim=-1)[1]

                # this don't support softmax

            assert self._use_sigmoid_score is True
            total_scores = torch.sigmoid(cls_preds)

            # Apply NMS in birdeye view
            nms_func = box_torch_ops.nms

            selected_boxes = None
            selected_labels = None
            selected_scores = None
            selected_dir_labels = None

            # get highest score per prediction, than apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = total_scores.squeeze(-1)
                top_labels = torch.zeros(
                    total_scores.shape[0],
                    device=total_scores.device,
                    dtype=torch.long)
            else:
                top_scores, top_labels = torch.max(total_scores, dim=-1)

            if self._nms_score_threshold > 0.0:
                thresh = torch.tensor(
                    [self._nms_score_threshold],
                    device=total_scores.device).type_as(total_scores)
                top_scores_keep = (top_scores >= thresh)
                top_scores = top_scores.masked_select(top_scores_keep)
            if top_scores.shape[0] != 0:
                if self._nms_score_threshold > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    if self._use_direction_classifier:
                        dir_labels = dir_labels[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]

                box_preds_corners = box_torch_ops.center_to_corner_box2d(
                    boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                    boxes_for_nms[:, 4])
                boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                    box_preds_corners)

                selected = nms_func(
                    boxes_for_nms,
                    top_scores,
                    pre_max_size=self._nms_pre_max_size,
                    post_max_size=self._nms_post_max_size,
                    iou_threshold=self._nms_iou_threshold,
                )

            else:
                selected = None

            if selected is not None:
                selected_boxes = box_preds[selected]
                if self._use_direction_classifier:
                    selected_dir_labels = dir_labels[selected]
                selected_labels = top_labels[selected]
                selected_scores = top_scores[selected]
            # finally generate predictions.

            if selected_boxes is not None:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                if self._use_direction_classifier:
                    dir_labels = selected_dir_labels
                    opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.byte()
                    box_preds[..., -1] += torch.where(
                        opp_labels,
                        torch.tensor(np.pi).type_as(box_preds),
                        torch.tensor(0.0).type_as(box_preds))
                    # box_preds[..., -1] += (
                    #     ~(dir_labels.byte())).type_as(box_preds) * np.pi
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                # print("final_label_preds shape", final_labels.shape)
                final_box_preds_camera = box_torch_ops.box_lidar_to_camera(
                    final_box_preds, rect, Trv2c)
                locs = final_box_preds_camera[:, :3]
                dims = final_box_preds_camera[:, 3:6]
                angles = final_box_preds_camera[:, 6]
                camera_box_origin = [0.5, 1.0, 0.5]
                box_corners = box_torch_ops.center_to_corner_box3d(
                    locs, dims, angles, camera_box_origin, axis=1)
                box_corners_in_image = box_torch_ops.project_to_image(
                    box_corners, P2)
                # box_corners_in_image: [N, 8, 2]
                minxy = torch.min(box_corners_in_image, dim=1)[0]
                maxxy = torch.max(box_corners_in_image, dim=1)[0]
                # minx = torch.min(box_corners_in_image[..., 0], dim=1)[0]
                # maxx = torch.max(box_corners_in_image[..., 0], dim=1)[0]
                # miny = torch.min(box_corners_in_image[..., 1], dim=1)[0]
                # maxy = torch.max(box_corners_in_image[..., 1], dim=1)[0]
                # box_2d_preds = torch.stack([minx, miny, maxx, maxy], dim=1)
                box_2d_preds = torch.cat([minxy, maxxy], dim=1)
                # predictions
                predictions_dict = {
                    "bbox": box_2d_preds,
                    "box3d_camera": final_box_preds_camera,
                    "box3d_lidar": final_box_preds,
                    "scores": final_scores,
                    "label_preds": label_preds,
                    "image_idx": img_idx,
                }
            else:
                predictions_dict = {
                    "bbox": None,
                    "box3d_camera": None,
                    "box3d_lidar": None,
                    "scores": None,
                    "label_preds": None,
                    "image_idx": img_idx,
                }
            predictions_dicts.append(predictions_dict)

        self.dt_annos += self.predict_kitti_to_anno(predictions_dicts, self.class_names, self.center_limit_range, self.lidar_input)

        if len(self.dt_annos) == len(self.gt_annos):

            ## Log
            log_path = self.model_dir+'/log.txt'
            logf = open(log_path, 'a')
            logf.write("\n")

            print("#################################")
            print("#################################", file=logf)
            print("# EVAL")
            print("# EVAL", file=logf)
            print("#################################")
            print("#################################", file=logf)
            print("Generate output labels...")
            print("Generate output labels...", file=logf)

            result, mAPbbox, mAPbev, mAP3d, mAPaos = get_official_eval_result(self.gt_annos, self.dt_annos, self.class_names,
                                                                                  return_data=True)
            print(result)
            print(result, file=logf)

            result = get_coco_eval_result(self.gt_annos, self.dt_annos, self.class_names)

            print(result)
            print(result, file=logf)

            self.dt_annos = []

            logf.close()
            # print("[info] empty self.dt_annos :> dt.annos len : ", len(self.dt_annos))

            top[0].data[...] = mAP3d[:,0,0] # 0.7 easy
            top[1].data[...] = mAP3d[:,1,0] # 0.7 moderate
            top[2].data[...] = mAP3d[:,2,0] # 0.7 hard
            top[3].data[...] = mAP3d[:,0,1] # 0.7 easy
            top[4].data[...] = mAP3d[:,1,1] # 0.5 moderate
            top[5].data[...] = mAP3d[:,2,1] # 0.5 moderate

    def backward(self, top, propagate_down, bottom):
        pass
    def load_generator(self):

        model_dir = pathlib.Path(self.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        config_file_bkp = "pipeline.config"
        config = pipeline_pb2.TrainEvalPipelineConfig()

        with open(self.config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
        shutil.copyfile(self.config_path, str(model_dir / config_file_bkp))

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

        #for evaluation
        self.class_names = list(input_cfg.class_names)
        self.center_limit_range = model_cfg.post_center_limit_range
        self.lidar_input = model_cfg.lidar_input

        eval_dataset = input_reader_builder.build(
            eval_input_cfg,
            model_cfg,
            training=False,
            voxel_generator=voxel_generator,
            target_assigner=target_assigner)
        gt_annos = [
            info["annos"] for info in eval_dataset.kitti_infos
        ]

        return target_assigner, gt_annos
    def predict_kitti_to_anno(self,
                            predictions_dicts,
                            class_names,
                            center_limit_range=None,
                            lidar_input=False,
                            global_set=None):
        annos = []
        for i, preds_dict in enumerate(predictions_dicts):
            image_shape = self.batch_image_shape[i]
            img_idx = preds_dict["image_idx"]
            if preds_dict["bbox"] is not None:
                box_2d_preds = preds_dict["bbox"].detach().cpu().numpy()
                box_preds = preds_dict["box3d_camera"].detach().cpu().numpy()
                scores = preds_dict["scores"].detach().cpu().numpy()
                box_preds_lidar = preds_dict["box3d_lidar"].detach().cpu().numpy()
                # write pred to file
                label_preds = preds_dict["label_preds"].detach().cpu().numpy()
                # label_preds = np.zeros([box_2d_preds.shape[0]], dtype=np.int32)

                anno = kitti.get_start_result_anno()
                num_example = 0
                for box, box_lidar, bbox, score, label in zip(
                        box_preds, box_preds_lidar, box_2d_preds, scores,
                        label_preds):
                    if not lidar_input:
                        if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                            continue
                        if bbox[2] < 0 or bbox[3] < 0:
                            continue
                    # print(img_shape)
                    if center_limit_range is not None:
                        limit_range = np.array(center_limit_range)
                        if (np.any(box_lidar[:3] < limit_range[:3])
                                or np.any(box_lidar[:3] > limit_range[3:])):
                            continue
                    bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    anno["name"].append(class_names[int(label)])
                    anno["truncated"].append(0.0)
                    anno["occluded"].append(0)
                    anno["alpha"].append(-np.arctan2(-box_lidar[1], box_lidar[0]) +
                                         box[6])
                    anno["bbox"].append(bbox)
                    anno["dimensions"].append(box[3:6])
                    anno["location"].append(box[:3])
                    anno["rotation_y"].append(box[6])
                    if global_set is not None:
                        for i in range(100000):
                            if score in global_set:
                                score -= 1 / 100000
                            else:
                                global_set.add(score)
                                break
                    anno["score"].append(score)

                    num_example += 1
                if num_example != 0:
                    anno = {n: np.stack(v) for n, v in anno.items()}
                    annos.append(anno)
                else:
                    annos.append(kitti.empty_result_anno())
            else:
                annos.append(kitti.empty_result_anno())
            num_example = annos[-1]["name"].shape[0]
            annos[-1]["image_idx"] = np.array(
                [img_idx] * num_example, dtype=np.int64)
        return annos

class LogLayer(caffe.Layer):

    def setup(self, bottom, top):
        in1 = bottom[0].data
        top[0].reshape(*in1.shape)

    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        in1 = bottom[0].data
        top[0].reshape(*in1.shape)
        top[0].data[...] = in1
        pass

    def backward(self, top, propagate_down, bottom):
        diff = top[0].diff
        bottom[0].diff[...]=diff
        pass

class GlobalPooling(caffe.Layer):
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        n, c, h, w = bottom[0].data.shape
        self.max_loc = bottom[0].data.reshape(n, c, h*w).argmax(axis=2)
        top[0].data[...] = bottom[0].data.max(axis=(2, 3), keepdims=True)

    def backward(self, top, propagate_down, bottom):
        n, c, h, w = top[0].diff.shape
        nn, cc = np.ix_(np.arange(n), np.arange(c))
        bottom[0].diff[...] = 0
        bottom[0].diff.reshape(n, c, -1)[nn, cc, self.max_loc] = top[0].diff.sum(axis=(2, 3))

class ProbRenorm(caffe.Layer):
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        clipped = bottom[0].data * bottom[1].data
        self.sc = 1.0 / (np.sum(clipped, axis=1, keepdims=True) + 1e-10)
        top[0].data[...] = clipped * self.sc

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = top[0].diff * bottom[1].data * self.sc

class Permute(caffe.Layer):
    def setup(self, bottom, top):
        self.dims = [int(v) for v in self.param_str.split('_')]
        self.dims_ind = list(np.argsort(self.dims))

    def reshape(self, bottom, top):
        old_shape = bottom[0].data.shape
        new_shape = [old_shape[d] for d in self.dims]
        top[0].reshape(*new_shape)

    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data.transpose(*self.dims)

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = top[0].diff.transpose(*self.dims_ind)

class LossHelper(caffe.Layer):
    def setup(self, bottom, top):
        self.old_shape = bottom[0].data.shape

    def reshape(self, bottom, top):
        new_shape = (self.old_shape[0] * self.old_shape[3], self.old_shape[1], 1, 1)
        top[0].reshape(*new_shape)

    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data.transpose(0, 3, 1, 2).reshape(*top[0].data.shape)

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = top[0].diff.reshape(self.old_shape[0], self.old_shape[3], self.old_shape[1], 1
                                                  ).transpose(0, 2, 3, 1)

class LogLoss(caffe.Layer):
    def setup(self, bottom, top):
        self.n, self.c, _, self.s = bottom[0].data.shape
        self.inds = np.ix_(np.arange(self.n), np.arange(self.c), np.arange(1), np.arange(self.s))

    def reshape(self, bottom, top):
        top[0].reshape(1, 1, 1, 1)

    def forward(self, bottom, top):
        self.valid = bottom[0].data[self.inds[0], bottom[1].data.astype(int), self.inds[2], self.inds[3]]
        top[0].data[:] = -np.mean(np.log(self.valid + 1e-10))

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[:] = 0.0
        bottom[0].diff[self.inds[0], bottom[1].data.astype(int), self.inds[2], self.inds[3]] = \
            -1.0 / ((self.valid + 1e-10) * (self.n * self.s))

class PickAndScale(caffe.Layer):
    def setup(self, bottom, top):
        self.nch_out = len(self.param_str.split('_'))
        self.dims = []
        for f in self.param_str.split('_'):
            if f.find('*') >= 0:
                self.dims.append((int(f[:f.find('*')]), float(f[f.find('*') + 1:])))
            elif f.find('/') >= 0:
                self.dims.append((int(f[:f.find('/')]), 1.0 / float(f[f.find('/') + 1:])))

            else:
                self.dims.append((int(f), 1.0))

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].data.shape[0], self.nch_out, bottom[0].data.shape[2], bottom[0].data.shape[3])

    def forward(self, bottom, top):
        for i, (j, s) in enumerate(self.dims):
            top[0].data[:, i, :, :] = bottom[0].data[:, j, :, :] * s

    def backward(self, top, propagate_down, bottom):
        pass  # TODO NOT_YET_IMPLEMENTED

class SigmoidCrossEntropyWeightLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check for all inputs
        params = eval(self.param_str)
        self.cls_weight = float(params["cls_weight"])
        # if len(bottom) != 2:
        #     raise Exception("Need two inputs (scores and labels) to compute sigmoid crossentropy loss.")

    def reshape(self, bottom, top):
        # check input dimensions match between the scores and labels
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference would be the same shape as any input
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # layer output would be an averaged scalar loss
        top[0].reshape(1)

    def forward(self, bottom, top):
        score=bottom[0].data
        label=bottom[1].data
        cls_weights=bottom[2].data

        first_term = (label-1)*score
        second_term = ((1-self.cls_weight)*label - 1)*np.log(1+np.exp(-score))


        cls_weight = np.expand_dims(cls_weights, axis = 0)

        top[0].data[...] = -np.sum((first_term + second_term)*cls_weights)

        sig = -1.0/(1.0+np.exp(-score))
        self.diff = ((self.cls_weight-1)*label+1)*sig + self.cls_weight*label
        if np.isnan(top[0].data):
                exit()

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...]=self.diff
