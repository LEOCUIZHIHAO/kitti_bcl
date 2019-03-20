import time
from enum import Enum

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import sparseconvnet as scn
import torchplus
from torchplus.nn import Empty, GroupNorm, Sequential
from torchplus.tools import change_default_args

class RPN(nn.Module):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 num_filters=[128, 128, 256],
                 upsample_strides=[1, 2, 4],
                 num_upsample_filters=[256, 256, 256],
                 num_input_filters=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_bev=False,
                 box_code_size=7,
                 name='rpn'):
        super(RPN, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        block2_input_filters = num_filters[0]
        if use_bev:
            self.bev_extractor = Sequential(
                Conv2d(6, 32, 3, padding=1),
                BatchNorm2d(32),
                nn.ReLU(),
                # nn.MaxPool2d(2, 2),
                Conv2d(32, 64, 3, padding=1),
                BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
            block2_input_filters += 64

        self.block1 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                num_input_filters, num_filters[0], 3, stride=layer_strides[0]),
            BatchNorm2d(num_filters[0]),
            nn.ReLU(),
        )
        for i in range(layer_nums[0]):
            self.block1.add(
                Conv2d(num_filters[0], num_filters[0], 3, padding=1))
            self.block1.add(BatchNorm2d(num_filters[0]))
            self.block1.add(nn.ReLU())
        self.deconv1 = Sequential(
            ConvTranspose2d(
                num_filters[0],
                num_upsample_filters[0],
                upsample_strides[0],
                stride=upsample_strides[0]),
            BatchNorm2d(num_upsample_filters[0]),
            nn.ReLU(),
        )
        self.block2 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                block2_input_filters,
                num_filters[1],
                3,
                stride=layer_strides[1]),
            BatchNorm2d(num_filters[1]),
            nn.ReLU(),
        )
        for i in range(layer_nums[1]):
            self.block2.add(
                Conv2d(num_filters[1], num_filters[1], 3, padding=1))
            self.block2.add(BatchNorm2d(num_filters[1]))
            self.block2.add(nn.ReLU())
        self.deconv2 = Sequential(
            ConvTranspose2d(
                num_filters[1],
                num_upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1]),
            BatchNorm2d(num_upsample_filters[1]),
            nn.ReLU(),
        )
        self.block3 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(num_filters[1], num_filters[2], 3, stride=layer_strides[2]),
            BatchNorm2d(num_filters[2]),
            nn.ReLU(),
        )
        for i in range(layer_nums[2]):
            self.block3.add(
                Conv2d(num_filters[2], num_filters[2], 3, padding=1))
            self.block3.add(BatchNorm2d(num_filters[2]))
            self.block3.add(nn.ReLU())
        self.deconv3 = Sequential(
            ConvTranspose2d(
                num_filters[2],
                num_upsample_filters[2],
                upsample_strides[2],
                stride=upsample_strides[2]),
            BatchNorm2d(num_upsample_filters[2]),
            nn.ReLU(),
        )
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        self.conv_box = nn.Conv2d(
            sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * 2, 1)

    def forward(self, x, bev=None):
        x = self.block1(x)
        up1 = self.deconv1(x)
        if self._use_bev:
            bev[:, -1] = torch.clamp(
                torch.log(1 + bev[:, -1]) / np.log(16.0), max=1.0)
            x = torch.cat([x, self.bev_extractor(bev)], dim=1)
        x = self.block2(x)
        up2 = self.deconv2(x)
        x = self.block3(x)
        up3 = self.deconv3(x)
        x = torch.cat([up1, up2, up3], dim=1)
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
        return ret_dict

class RPNV2(nn.Module):
    """Compare with RPN, RPNV2 support arbitrary number of stage.
    """
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 num_filters=[128, 128, 256],
                 upsample_strides=[1, 2, 4],
                 num_upsample_filters=[256, 256, 256],
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_bev=False,
                 box_code_size=7,
                 use_rc_net=False,
                 name='rpn'):
        super(RPNV2, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        self._use_rc_net = use_rc_net
        # assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        """
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        """
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        in_filters = [num_input_features, *num_filters[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(layer_nums):
            block = Sequential(
                nn.ZeroPad2d(1),
                Conv2d(
                    in_filters[i], num_filters[i], 3, stride=layer_strides[i]),
                BatchNorm2d(num_filters[i]),
                nn.ReLU(),
            )
            for j in range(layer_num):
                block.add(
                    Conv2d(num_filters[i], num_filters[i], 3, padding=1))
                block.add(BatchNorm2d(num_filters[i]))
                block.add(nn.ReLU())
            blocks.append(block)
            deblock = Sequential(
                ConvTranspose2d(
                    num_filters[i],
                    num_upsample_filters[i],
                    upsample_strides[i],
                    stride=upsample_strides[i]),
                BatchNorm2d(num_upsample_filters[i]),
                nn.ReLU(),
            )
            deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        self.conv_box = nn.Conv2d(
            sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * 2, 1)

        if self._use_rc_net:
            self.conv_rc = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * box_code_size,
                1)

    def forward(self, x, bev=None):
        # t = time.time()
        # torch.cuda.synchronize()
        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            ups.append(self.deblocks[i](x))
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
        if self._use_rc_net:
            rc_preds = self.conv_rc(x)
            rc_preds = rc_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["rc_preds"] = rc_preds
        # torch.cuda.synchronize()
        # print("rpn forward time", time.time() - t)
        return ret_dict

class SparseRPN(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_class=2,
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 num_filters=[128, 128, 256],
                 upsample_strides=[1, 2, 4],
                 num_upsample_filters=[256, 256, 256],
                 num_input_features=64,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_bev=False,
                 box_code_size=7,
                 name='rpn'):
        super(SparseRPN, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        if use_norm:
            if use_groupnorm:
                BatchNormReLU = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNormReLU = change_default_args(
                    eps=1e-3, momentum=0.01)(scn.BatchNormReLU)

            Convolution = change_default_args(bias=False)(scn.Convolution)
            Deconvolution = change_default_args(bias=False)(
                scn.Deconvolution)
            SubmanifoldConvolution = change_default_args(bias=False)(
                scn.SubmanifoldConvolution)
        # else:
        #     BatchNormReLU = Empty
        #     Convolution = change_default_args(bias=True)(scn.Convolution)
        #     Deconvolution = change_default_args(bias=True)(
        #         scn.Deconvolution)

        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        # block2_input_filters = num_filters[0]
        # if use_bev:
        #     self.bev_extractor = Sequential(
        #         Conv2d(6, 32, 3, padding=1),
        #         BatchNorm2d(32),
        #         nn.ReLU(),
        #         # nn.MaxPool2d(2, 2),
        #         Conv2d(32, 64, 3, padding=1),
        #         BatchNorm2d(64),
        #         nn.ReLU(),
        #         nn.MaxPool2d(2, 2),
        #     )
        #     block2_input_filters += 64
        sparse_shape = np.array(output_shape)[2:4]
        self.scn_input = scn.InputLayer(2, sparse_shape.tolist())

        self.block1 = scn.Sequential(
            # nn.ZeroPad2d(1),
            Convolution(2, num_input_features, num_filters[0], 2, layer_strides[0], False), # dimension, nIn, nOut, filter_size, filter_stride, bias
            BatchNormReLU(num_filters[0]))

        for i in range(layer_nums[0]):
            self.block1.add(
                SubmanifoldConvolution(2, num_filters[0], num_filters[0], 3, False)) # dimension, nIn, nOut, filter_size, bias
            self.block1.add(BatchNormReLU(num_filters[0]))

        # dimension, nIn, nOut, filter_size, filter_stride, bias
        # self.deconv1 = scn.Sequential(
        #     # scn.SparseToDense(2, num_filters[0]),
        #     Deconvolution(
        #         2,
        #         num_filters[0],
        #         num_upsample_filters[0],
        #         upsample_strides[0],
        #         upsample_strides[0],
        #         False),
        #     BatchNormReLU(num_upsample_filters[0]),
        #
        #     # scn.OutputLayer(3) # not sure
        #     scn.SparseToDense(2, num_upsample_filters[0]) # not sure
        #     # scn.OutputLayer(2)
        # )

        self.deconv1 = scn.SparseToDense(2, num_filters[0])
        self.norm_dec1 = Sequential(
            nn.ConvTranspose2d(
                    num_filters[0],
                    num_upsample_filters[0],
                    upsample_strides[0],
                    stride=upsample_strides[0], bias=False),
            nn.BatchNorm2d(num_upsample_filters[0]),
            nn.ReLU()
        )

        ###########################Block 2######################################
        self.block2 = scn.Sequential(
            # nn.ZeroPad2d(0),
            Convolution(2, num_filters[0], num_filters[1], 2, layer_strides[1], False), # dimension, nIn, nOut, filter_size, filter_stride, bias
            BatchNormReLU(num_filters[1]))

        for i in range(layer_nums[1]):
            self.block2.add(
                SubmanifoldConvolution(2, num_filters[1], num_filters[1], 3, False)) # dimension, nIn, nOut, filter_size, bias
            self.block2.add(BatchNormReLU(num_filters[1]))

        # dimension, nIn, nOut, filter_size, filter_stride, bias
        # self.deconv2 = scn.Sequential(
        #     # scn.SparseToDense(2, num_filters[1]),
        #     Deconvolution(
        #         2,
        #         num_filters[1],
        #         num_upsample_filters[1],
        #         upsample_strides[1],
        #         upsample_strides[1],
        #         False),
        #     BatchNormReLU(num_upsample_filters[1]),
        #     scn.SparseToDense(2, num_upsample_filters[1])
        # )
        #
        self.deconv2 = scn.SparseToDense(2, num_filters[1])
        self.norm_dec2 = Sequential(
            nn.ConvTranspose2d(
                    num_filters[1],
                    num_upsample_filters[1],
                    upsample_strides[1],
                    stride=upsample_strides[1], bias=False),
            nn.BatchNorm2d(num_upsample_filters[1]),
            nn.ReLU()
        )

        ###########################Block 3######################################

        self.block3 = scn.Sequential(
            # nn.ZeroPad2d(0),
            Convolution(2, num_filters[1], num_filters[2], 2, layer_strides[2], False), # dimension, nIn, nOut, filter_size, filter_stride, bias
            BatchNormReLU(num_filters[2]))

        for i in range(layer_nums[2]):
            self.block3.add(
                SubmanifoldConvolution(2, num_filters[2], num_filters[2], 3, False)) # dimension, nIn, nOut, filter_size, bias
            self.block3.add(BatchNormReLU(num_filters[2]))

        # dimension, nIn, nOut, filter_size, filter_stride, bias
        # self.deconv3 = scn.Sequential(
        #     # scn.SparseToDense(2, num_filters[2]),
        #     Deconvolution(
        #         2,
        #         num_filters[2],
        #         num_upsample_filters[2],
        #         upsample_strides[2],
        #         upsample_strides[2],
        #         False),
        #     BatchNormReLU(num_upsample_filters[2]),
        #     scn.SparseToDense(2, num_upsample_filters[2])
        # )

        self.deconv3 = scn.SparseToDense(2, num_filters[2])
        self.norm_dec3 = Sequential(
            nn.ConvTranspose2d(
                    num_filters[2],
                    num_upsample_filters[2],
                    upsample_strides[2],
                    stride=upsample_strides[2], bias=False),
            nn.BatchNorm2d(num_upsample_filters[2]),
            nn.ReLU()
        )

        self.post = Sequential(
                    nn.Conv2d(
                        sum(num_upsample_filters),
                        128,
                        3,
                        stride=1,padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(
                        128,
                        64,
                        3,
                        stride=1,padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),

        ) # [1, 200, 176]

        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)

        # self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        # self.conv_box = nn.Conv2d(
        #     sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1)
        # if use_direction_classifier:
        #     self.conv_dir_cls = nn.Conv2d(
        #         sum(num_upsample_filters), num_anchor_per_loc * 2, 1)

        self.conv_cls = nn.Conv2d(64, num_cls, 1)
        self.conv_box = nn.Conv2d(
            64, num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                64, num_anchor_per_loc * 2, 1)

    def forward(self, voxel_features, coors, batch_size, bev=None):
        coors = coors.int()[:,[2,3,0]]
        # print(coors)
        # print(len((coors, voxel_features, batch_size)))
        # print("============================")
        sx = self.scn_input((coors.cpu(), voxel_features, batch_size))
        # print(sx.features.shape)
        x1 = self.block1(sx)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        # print("block-1",x1.features.shape)
        # print("block-2",x2.features.shape)
        # print("block-3",x3.features.shape)
        up1 = self.deconv1(x1)
        up1 = self.norm_dec1(up1)
        # print("up-1",up1.shape)
        up2 = self.deconv2(x2)
        up2 = self.norm_dec2(up2)
        # print("up-2",up2.shape)
        up3 = self.deconv3(x3)
        up3 = self.norm_dec3(up3)
        # print("up-3",up3.shape)

        # x = self.block1(x)
        # print("block-1",x.features.shape)
        # up1 = self.deconv1(x)
        # x = self.block2(x)
        # print("block-2",x.features.shape)
        # up2 = self.deconv2(x)
        # x = self.block3(x)
        # print("block-3",x.features.shape)
        # up3 = self.deconv3(x)
        #
        # print("up-1",up1.shape)
        # print("up-2",up2.shape)
        # print("up-3",up3.shape)

        x = torch.cat([up1,up2,up3], dim=1)
        x = self.post(x)
        # print("concat shape", x.shape)
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
        return ret_dict
