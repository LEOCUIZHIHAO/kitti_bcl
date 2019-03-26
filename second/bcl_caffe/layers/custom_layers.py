import numpy as np
import caffe
import pathlib
import shutil
import time
from functools import partial

from second.bcl_caffe.utils import get_paddings_indicator_caffe
from second.builder import target_assigner_builder, voxel_builder
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                      lr_scheduler_builder, optimizer_builder,
                                      second_builder)
from second.protos import pipeline_pb2, losses_pb2, second_pb2
from google.protobuf import text_format #make prototxt work
from collections import defaultdict # for merge data to batch
from enum import Enum

from builder import losses_builder

import torch

class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"


class InputKittiData(caffe.Layer):

    def setup(self, bottom, top):

        print("#### PREPARE INPUT -custom_layers.py")
        params = dict(subset='train', batch_size=2)
        params.update(eval(self.param_str))

        self.model_dir = params['model_dir']
        self.config_path = params['config_path']
        self.batch_size = params['batch_size']
        self.top_names = ['data', 'label']

        ########################################################################
        ## TODO:  pass by param
        point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
        voxel_size = [0.16, 0.16, 4]
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]

        ########################################################################
        example = self.prep_and_read_kitti()
        self.voxels = example['voxels']
        self.coors = example['coordinates']
        self.num_points = example['num_points']
        self.labels = example['labels']
        self.reg_targets =example['reg_targets']

        points_mean = np.sum(self.voxels[:, :, :3], axis=1, keepdims=True) / self.num_points.reshape(-1,1,1)
        f_cluster = self.voxels[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center = self.voxels[:, :, :2]
        f_center[:, :, 0] = f_center[:, :, 0] - (np.expand_dims(self.coors[:, 3].astype(float), axis=1) * self.vx + self.x_offset)
        f_center[:, :, 1] = f_center[:, :, 1] - (np.expand_dims(self.coors[:, 2].astype(float), axis=1) * self.vy + self.y_offset)

        features_ls = [self.voxels, f_cluster, f_center]
        self.features = np.concatenate(features_ls, axis=-1) #[num_voxles, points_num, features]

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.

        # TODO:  double check !!!!!!!!!!!!
        points_per_voxels = self.features.shape[1]
        mask = get_paddings_indicator_caffe(self.num_points, points_per_voxels, axis=0)
        #mask = torch.unsqueeze(mask, -1).type_as(features)
        mask = np.expand_dims(mask, axis=-1)
        self.features *= mask
        ########################################################################

        # voxel_num = self.voxels.shape[0]
        # max_points_in_voxels = self.voxels.shape[1]
        #self.features = self.features.reshape(1, voxel_num, max_points_in_voxels, -1,).transpose(0,3,1,2)
        #self.data = features
        #top[0].reshape(*features.shape)

    def reshape(self, bottom, top):
        print("flag - reshape")
        voxel_num = self.voxels.shape[0]
        max_points_in_voxels = self.voxels.shape[1]
        self.features = self.features.reshape(1, voxel_num, max_points_in_voxels, -1).transpose(0,3,1,2)
        top[0].reshape(*self.features.shape) #[1,9,7000,100]

        # self.coors = self.coors.reshape(1,1,voxel_num,-1) #[1, 1, 7000, 4]
        top[1].reshape(*self.coors.shape) #[7000,4]
        top[2].reshape(*self.labels.shape) #[2 107136]
        top[3].reshape(*self.reg_targets.shape) #[]
        pass

    def forward(self, bottom, top):
        print("[flag -forward]")
        top[0].data[...] = self.features
        top[1].data[...] = self.coors
        top[2].data[...] = self.labels
        top[3].data[...] = self.reg_targets

        print("[flag -forward done]")

    def backward(self, top, propagate_down, bottom):
        pass

    def prep_and_read_kitti(self):

        create_folder = False
        result_path = None
        print("[config_path]", self.config_path)

        if create_folder:
            if pathlib.Path(self.model_dir).exists():
                model_dir = torchplus.train.create_folder(self.model_dir)

        model_dir = pathlib.Path(self.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        eval_checkpoint_dir = model_dir / 'eval_checkpoints'
        eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if result_path is None:
            result_path = model_dir / 'results'
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
        for _ in range(self.batch_size):
            example = next(data_iter)
            print("[debug] image_idx" , example["image_idx"])
            example_batch.append(example)

        ret = self.merge_second_batch(example_batch)



        return ret

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
        num_input_features = param['num_input_features']

        self.ny = output_shape[2]
        self.nx = output_shape[3]
        self.nchannels = num_input_features
        self.batch_size = 2 # TODO: pass batch to here

        voxel_features = bottom[0].data
        voxel_features = np.transpose(np.squeeze(voxel_features))
        coords = bottom[1].data

        self.batch_canvas = []
        for batch_itt in range(self.batch_size):
            # Create the canvas for this sample
            canvas = np.zeros(shape=(self.nchannels, self.nx * self.ny))

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.astype(int)
            voxels = voxel_features[batch_mask, :]
            voxels = np.transpose(voxels)

            _coors = coords[batch_mask, :]
            _coors = np.transpose(_coors)

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            self.batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        self.batch_canvas = np.stack(self.batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        self.batch_canvas = self.batch_canvas.reshape(self.batch_size, self.nchannels, self.ny, self.nx)

    def reshape(self, bottom, top):
        top[0].reshape(*self.batch_canvas.shape)

    def forward(self, bottom, top):
        print("[flag -scater forward start]")
        top[0].data[...] = self.batch_canvas
        print("[flag -scater forward done]")

    def backward(self, top, propagate_down, bottom):
        pass


class Reshape(caffe.Layer):
    def setup(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def reshape(self, bottom,top):
        pass

    def forward(self, bottom, top):
        label = bottom[0].data
        print("reshape label.shape: ", label.shape)
        predict = bottom[1].data
        print("reshape predict.shape: ", predict.shape)
        predict = predict.reshape(label.shape)
        print("reshape predit.shape: ", predict.shape)
        top[0].data[...] = predict
        # exit()

    def backward(self, bottom, top):
        pass


class CreateLoss(caffe.Layer):

    def setup(self, bottom, top):
        print("####Loss setup")
        ####read proto
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open('configs/pointpillars/car/xyres_16.proto', "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
        loss_config = config.model.second.loss
        #######################################
        '''stay in setup'''
        self.cls_loss_ftor, self.loc_loss_ftor, self.cls_loss_weight, self.loc_loss_weight, _ = losses_builder._build_loss(loss_config)
        """
        task breakdown:
        Get cls_weights, reg_weights, cared from prepareloss;
        Get labels and reg_targets
        """
        # Should be in forward

    def reshape(self, bottom, top):
        print("####Loss reshape")
        pass

    def forward(self, bottom, top):
        reg_targets = bottom[0].data
        labels = bottom[1].data
        box_preds = bottom[2].data
        cls_preds = bottom[3].data

        # should be in forward
        cls_weights, reg_weights, cared = self.prepare_loss_weights(labels)

        cls_targets = labels * cared
        cls_targets = np.expand_dims(cls_targets, -1).astype(int)

        cls_losses, loc_losses = self.create_loss(
                                    self.loc_loss_ftor,
                                    self.cls_loss_ftor,
                                    box_preds=box_preds,
                                    cls_preds=cls_preds,
                                    cls_targets=cls_targets,
                                    cls_weights=cls_weights,
                                    reg_targets=reg_targets,
                                    reg_weights=reg_weights,
                                    num_class=1, ## TODO:
                                    encode_rad_error_by_sin= True, # TODO:  pass #self._encode_rad_error_by_sin,
                                    encode_background_as_zeros= True, #self._encode_background_as_zeros,
                                    box_code_size= 7) #self._box_coder.code_size,)
        pos_cls_weight = 1
        neg_cls_weight = 1
        batch_size = 2 ## TODO:  pass param to here

        loc_loss_reduced = loc_losses.sum() / batch_size
        loc_loss_reduced *= self.loc_loss_weight
        cls_pos_loss, cls_neg_loss = self.get_pos_neg_loss(cls_losses, labels)
        cls_pos_loss /= pos_cls_weight
        cls_neg_loss /= neg_cls_weight
        cls_loss_reduced = cls_losses.sum() / batch_size
        cls_loss_reduced *= self.cls_loss_weight
        self.loss = loc_loss_reduced + cls_loss_reduced
        top[0].reshape(*self.loss.shape)
        top[0].data[...] = self.loss
        print("####Loss forward")

    def backward(self, top, propagate_down, bottom):
        print("####Loss backward")
        bottom[0].diff[...] = self.loss
        bottom[1].diff[...] = self.loss


    def add_sin_difference(self, boxes1, boxes2):
        rad_pred_encoding = np.sin(boxes1[..., -1:]) * np.cos(
            boxes2[..., -1:])
        rad_tg_encoding = np.cos(boxes1[..., -1:]) * np.sin(boxes2[..., -1:])
        boxes1 = np.concatenate([boxes1[..., :-1], rad_pred_encoding], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :-1], rad_tg_encoding], axis=-1)
        return boxes1, boxes2

    def prepare_loss_weights(self, labels,
                            pos_cls_weight=1.0, # TODO: pass params here
                            neg_cls_weight=1.0,
                            loss_norm_type=LossNormType.NormByNumPositives,
                            dtype="float32"):
        """get cls_weights and reg_weights from labels.
        """
        cared = labels >= 0
        # cared: [N, num_anchors]
        positives = labels > 0
        negatives = labels == 0
        negative_cls_weights = negatives.astype(dtype) * neg_cls_weight
        cls_weights = negative_cls_weights + pos_cls_weight * positives.astype(dtype)
        reg_weights = positives.astype(dtype)
        if loss_norm_type == LossNormType.NormByNumExamples:
            num_examples = cared.astype(dtype).sum(1, keepdims=True)
            num_examples = np.clip(num_examples, a_min=1.0, a_max=None)
            cls_weights /= num_examples
            bbox_normalizer = np.sum(positives, 1, keepdims=True).astype(dtype)
            reg_weights /= np.clip(bbox_normalizer, a_min=1.0, a_max=None)
        elif loss_norm_type == LossNormType.NormByNumPositives:  # for focal loss # TODO: double check
            pos_normalizer = np.sum(positives, 1, keepdims=True).astype(dtype)
            reg_weights /= np.clip(pos_normalizer, a_min=1.0, a_max=None)
            cls_weights /= np.clip(pos_normalizer, a_min=1.0, a_max=None)
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

    def create_loss(self,
                    loc_loss_ftor,
                    cls_loss_ftor,
                    box_preds,
                    cls_preds,
                    cls_targets,
                    cls_weights,
                    reg_targets,
                    reg_weights,
                    num_class=1,
                    encode_rad_error_by_sin=True,
                    encode_background_as_zeros=True,
                    box_code_size=7,
                    ):

        batch_size = int(box_preds.shape[0])
        box_preds = box_preds.reshape(batch_size, -1, box_code_size)
        # print("cls_preds shape", cls_preds.shape)
        if encode_background_as_zeros:
            cls_preds = cls_preds.reshape(batch_size, -1, num_class)
        else:
            cls_preds = cls_preds.reshape(batch_size, -1, num_class + 1)
        cls_targets = np.squeeze(cls_targets, -1) # cls_targets.squeeze(-1)
        # print("cls_preds shape after", cls_preds.shape)

        # print("cls_targets shape", cls_targets.shape)
        one_hot_targets = np.eye(num_class+1)[cls_targets]   #One_hot label -- make sure one hot class is <num_class+1>
        # print("one_hot_targets shape", one_hot_targets.shape)

        if encode_background_as_zeros:
            one_hot_targets = one_hot_targets[..., 1:]
        if encode_rad_error_by_sin:
            # sin(a - b) = sinacosb-cosasinb
            # TODO: double check
            box_preds, reg_targets = self.add_sin_difference(box_preds, reg_targets)


        loc_losses = loc_loss_ftor(
            box_preds, reg_targets, weights=reg_weights)  # [N, M]
        cls_losses = cls_loss_ftor(
            cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]

        return cls_losses, loc_losses

    def get_pos_neg_loss(self, cls_loss, labels):
        # cls_loss: [N, num_anchors, num_class]
        # labels: [N, num_anchors]
        batch_size = cls_loss.shape[0]
        if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
            cls_pos_loss = (labels > 0).astype(cls_loss.dtype) * cls_loss.reshape(
                batch_size, -1)
            cls_neg_loss = (labels == 0).astype(cls_loss.dtype) * cls_loss.reshape(
                batch_size, -1)
            cls_pos_loss = cls_pos_loss.sum() / batch_size
            cls_neg_loss = cls_neg_loss.sum() / batch_size
        else:
            cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
            cls_neg_loss = cls_loss[..., 0].sum() / batch_size
        return cls_pos_loss, cls_neg_loss


class WeightedSmoothL1LocalizationLoss(caffe.Layer):

    def setup(self, bottom, top):
        reg_targets = bottom[0].data
        box_preds = bottom[1].data
        labels = bottom[2].data

        encode_rad_error_by_sin=True
        box_code_size=7

        _, reg_weights, cared = self.prepare_loss_weights(labels)

        batch_size = int(box_preds.shape[0])
        box_preds = box_preds.reshape(batch_size, -1, box_code_size)

        if encode_rad_error_by_sin:
            # sin(a - b) = sinacosb-cosasinb
            # TODO: double check
            box_preds, reg_targets = self.add_sin_difference(box_preds, reg_targets)

        loc_losses = loc_loss_ftor(
             box_preds, reg_targets, weights=reg_weights)  # [N, M]

    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        pass
    def backward(self, top, propagate_down, bottom):
        pass
    def loc_loss_ftor(self, box_preds, reg_targets, weights=None):
        pass
    def add_sin_difference(self, boxes1, boxes2):
        rad_pred_encoding = np.sin(boxes1[..., -1:]) * np.cos(
            boxes2[..., -1:])
        rad_tg_encoding = np.cos(boxes1[..., -1:]) * np.sin(boxes2[..., -1:])
        boxes1 = np.concatenate([boxes1[..., :-1], rad_pred_encoding], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :-1], rad_tg_encoding], axis=-1)
        return boxes1, boxes2

    def prepare_loss_weights(self, labels,
                            pos_cls_weight=1.0, # TODO: pass params here
                            neg_cls_weight=1.0,
                            loss_norm_type=LossNormType.NormByNumPositives,
                            dtype="float32"):
        """get cls_weights and reg_weights from labels.
        """
        cared = labels >= 0
        # cared: [N, num_anchors]
        positives = labels > 0
        negatives = labels == 0
        negative_cls_weights = negatives.astype(dtype) * neg_cls_weight
        cls_weights = negative_cls_weights + pos_cls_weight * positives.astype(dtype)
        reg_weights = positives.astype(dtype)
        if loss_norm_type == LossNormType.NormByNumExamples:
            num_examples = cared.astype(dtype).sum(1, keepdims=True)
            num_examples = np.clip(num_examples, a_min=1.0, a_max=None)
            cls_weights /= num_examples
            bbox_normalizer = np.sum(positives, 1, keepdims=True).astype(dtype)
            reg_weights /= np.clip(bbox_normalizer, a_min=1.0, a_max=None)
        elif loss_norm_type == LossNormType.NormByNumPositives:  # for focal loss # TODO: double check
            pos_normalizer = np.sum(positives, 1, keepdims=True).astype(dtype)
            reg_weights /= np.clip(pos_normalizer, a_min=1.0, a_max=None)
            cls_weights /= np.clip(pos_normalizer, a_min=1.0, a_max=None)
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


class SigmoidFocalClassificationLoss(caffe.Layer):

    def setup(self, bottom, top):
        labels = bottom[0].data
        cls_preds = bottom[1].data

        num_class=1  # TODO: pass params to here
        encode_background_as_zeros=True

        cls_weights, _, cared = self.prepare_loss_weights(labels)

        cls_targets = labels * cared
        cls_targets = np.expand_dims(cls_targets, -1).astype(int)

        batch_size = int(cls_preds.shape[0])
        # print("cls_preds shape", cls_preds.shape)
        if encode_background_as_zeros:
            cls_preds = cls_preds.reshape(batch_size, -1, num_class)
        else:
            cls_preds = cls_preds.reshape(batch_size, -1, num_class + 1)
        cls_targets = np.squeeze(cls_targets, -1) # cls_targets.squeeze(-1)
        # print("cls_preds shape after", cls_preds.shape)

        # print("cls_targets shape", cls_targets.shape)
        one_hot_targets = np.eye(num_class+1)[cls_targets]   #One_hot label -- make sure one hot class is <num_class+1>
        # print("one_hot_targets shape", one_hot_targets.shape)

        if encode_background_as_zeros:
            one_hot_targets = one_hot_targets[..., 1:]

        cls_losses = cls_loss_ftor(
            cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]

    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        pass
    def backward(self, top, propagate_down, bottom):
        pass
    def cls_loss_ftor(self, cls_preds, one_hot_targets, weights=None):
        pass
    def prepare_loss_weights(self, labels,
                            pos_cls_weight=1.0, # TODO: pass params here
                            neg_cls_weight=1.0,
                            loss_norm_type=LossNormType.NormByNumPositives,
                            dtype="float32"):
        """get cls_weights and reg_weights from labels.
        """
        cared = labels >= 0
        # cared: [N, num_anchors]
        positives = labels > 0
        negatives = labels == 0
        negative_cls_weights = negatives.astype(dtype) * neg_cls_weight
        cls_weights = negative_cls_weights + pos_cls_weight * positives.astype(dtype)
        reg_weights = positives.astype(dtype)
        if loss_norm_type == LossNormType.NormByNumExamples:
            num_examples = cared.astype(dtype).sum(1, keepdims=True)
            num_examples = np.clip(num_examples, a_min=1.0, a_max=None)
            cls_weights /= num_examples
            bbox_normalizer = np.sum(positives, 1, keepdims=True).astype(dtype)
            reg_weights /= np.clip(bbox_normalizer, a_min=1.0, a_max=None)
        elif loss_norm_type == LossNormType.NormByNumPositives:  # for focal loss # TODO: double check
            pos_normalizer = np.sum(positives, 1, keepdims=True).astype(dtype)
            reg_weights /= np.clip(pos_normalizer, a_min=1.0, a_max=None)
            cls_weights /= np.clip(pos_normalizer, a_min=1.0, a_max=None)
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
