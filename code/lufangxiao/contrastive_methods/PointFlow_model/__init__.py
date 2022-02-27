"""
Network Initializations
"""

import logging
import importlib
import torch
import numpy as np
import cv2
from .pointflow_resnet_with_max_avg_pool import *

def get_model(network, num_classes, pretrained=True):
    """
    Fetch Network Function Pointer
    """
    module = network[:network.rfind('.')]
    model = network[network.rfind('.') + 1:]
    mod = importlib.import_module(module)
    net_func = getattr(mod, model)
    if model == 'DeepR101_PF_maxavg_deeply' or model == 'DeepR50_PF_maxavg_deeply':
        net = net_func(num_classes=num_classes, reduce_dim=64,
                       max_pool_size=9, avgpool_size=9, edge_points=32, pretrained=pretrained)
    else:
        net = net_func(num_classes=num_classes)
    return net

def get_boundary(mask, thicky=8):
    assert len(mask.size()) == 3
    edge = torch.zeros_like(mask).to(mask.device)
    B = mask.size(0)
    for i in range(B):
        tmp = mask[i].squeeze().cpu().data.numpy().astype('uint8')
        contour, _ = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boundary = np.zeros_like(tmp)
        boundary = cv2.drawContours(boundary, contour, -1, 1, thicky)
        boundary = boundary.astype(np.float)
        edge[i] = torch.from_numpy(boundary)
        del tmp, contour, boundary
    return edge

def get_body(mask, edge):
    edge_valid = edge == 1
    body = mask.clone()
    body[edge_valid] = 0
    return body

def build_model(num_classes, pretrained=True):
    model = DeepR50_PF_maxavg_deeply(num_classes=num_classes, reduce_dim=64,
                       max_pool_size=9, avgpool_size=9, edge_points=32, pretrained=pretrained)
    return model

if __name__ == '__main__':
    model = get_model('pointflow_resnet_with_max_avg_pool.DeepR50_PF_maxavg_deeply', 2).eval()
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params / 1e6)
    input = torch.autograd.Variable(torch.randn(2, 3, 1024, 1024))
    mask = torch.autograd.Variable(torch.randn(2, 1, 1024, 1024))
    x = model(input)
    print(x[0].size())
    for e in x[1]:
        print(e.size())
    e = get_boundary(mask)
    print(e.size())