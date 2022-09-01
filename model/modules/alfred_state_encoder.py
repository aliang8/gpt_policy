import os
import json
import numpy as np
import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ALFREDStateEncoder(nn.Module):
    def __init__(self, hparams, dataset_info=None):
        super().__init__()
        self.hparams = hparams

        if self.hparams.input_type == "resnet_feats":
            input_shape = dataset_info["feat_shape"][1:]
            layers, activation_shape = self.init_cnn(
                input_shape, channels=[256, 64], kernels=[1, 1], paddings=[0, 0]
            )
            layers += [
                Flatten(),
                nn.Linear(np.prod(activation_shape), self.hparams.enc_dim),
            ]
            self.layers = nn.Sequential(*layers)

        if self.hparams.enc_method == "mlp":
            # do nothing
            pass
        elif self.hparams.enc_method == "graph":
            pass
        elif self.hparams.enc_method == "transformer":
            pass

    def init_cnn(self, input_shape, channels, kernels, paddings):
        layers = []
        planes_in, spatial = input_shape[0], input_shape[-1]
        for planes_out, kernel, padding in zip(channels, kernels, paddings):
            # do not use striding
            stride = 1
            layers += [
                nn.Conv2d(
                    planes_in,
                    planes_out,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                ),
                nn.BatchNorm2d(planes_out),
                nn.ReLU(inplace=True),
            ]
            planes_in = planes_out
            spatial = (spatial - kernel + 2 * padding) // stride + 1
        activation_shape = (planes_in, spatial, spatial)
        return layers, activation_shape

    def forward(self, state):
        if self.hparams.input_type == "resnet_feats":
            out = self.layers(state)
        else:
            out = state
        return out
