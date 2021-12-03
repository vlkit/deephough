import torch
from . import hough_ext
import numpy as np
import matplotlib.pyplot as plt
import time

class deephough(torch.autograd.Function):
    @staticmethod 
    def forward(ctx, feat, numangle, numrho):
        N, C, _, _ = feat.size()
        out = torch.zeros(N, C, numangle, numrho).type_as(feat).cuda()
        out = hough_ext.forward(feat, out, numangle, numrho)
        outputs = out[0]
        ctx.save_for_backward(feat)
        ctx.numangle = numangle
        ctx.numrho = numrho
        return outputs
        
    @staticmethod
    def backward(ctx, grad_output):
        feat = ctx.saved_tensors[0]
        numangle = ctx.numangle
        numrho = ctx.numrho
        out = torch.zeros_like(feat).type_as(feat).cuda()
        out = hough_ext.backward(grad_output.contiguous(), out, feat, numangle, numrho)
        grad_in = out[0]
        return grad_in, None, None


class DeepHough(torch.nn.Module):
    def __init__(self, num_angle, num_bias):
        super().__init__()
        self.num_angle = num_angle
        self.num_bias = num_bias
    
    def forward(self, feat):
        return deephough.apply(feat, self.num_angle, self.num_bias)
