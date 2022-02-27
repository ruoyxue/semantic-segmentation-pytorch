from code.models.D_LinkNet import DLinkNet
from code.models.DeepLabv3plus import DeepLabV3Plus
from code.models.LinkNet import LinkNet
from code.models.ResUNet import ResUNet
from code.models.SegHRNet import SegHRNet
from code.models.SegHRNet_OCR import SegHRNet_OCR
from code.models.SegHR_LinkNet import SegHR_LinkNet
from code.models.SegHR_LinkNet_SUA import SegHR_LinkNet_SUA
from code.models.Swin_LinkNet import Swin_LinkNet
from code.models.D_LinkNet_SUA import DLinkNet_SUA
from code.models.Swin_LinkNet_SUA import Swin_LinkNet_SUA

from .contrastive_methods import DiResNet_model, PointFlow_model
from ..models.road_extraction import GAMSNet_model, JointNet_model, SIINet_model, BT_RoadNet_model


def build_model(in_ch,
                n_classes, 
                model_key='', 
                backbone='resnet34', 
                pretrained_flag=False, 
                resume_path='',
                vis=False):
    assert model_key in ['DLinkNet', 'DeepLabV3Plus', 'LinkNet', 'ResUNet', 'SegHRNet', 'SegHRNet_OCR', 'SegHR_LinkNet', 
                         'Swin_LinkNet', 'SIINet', 'BT_RoadNet', 'GAMSNet', 'DiResNet', 'JointNet', 'PointFlow', 'SegHR_LinkNet_SUA', 'DLinkNet_SUA', 'Swin_LinkNet_SUA'], '%s not registered in models' % model_key
    if model_key == 'DLinkNet':
        assert backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101']
        model = DLinkNet(n_classes, 
                         in_ch, 
                         backbone=backbone, 
                         pretrained_flag=pretrained_flag)
    elif model_key == 'DeepLabV3Plus':
        assert backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101']
        model = DeepLabV3Plus(in_ch, 
                              n_classes, 
                              atrous_rates=[6, 12, 18], 
                              multi_grids=[1, 2, 4],
                              output_stride=16,
                              backbone=backbone)
        if pretrained_flag:
            model.init_weights(resume_path)
    elif model_key == 'LinkNet':
        assert backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101']
        model = LinkNet(n_classes, 
                        in_ch, 
                        backbone=backbone, 
                        pretrained_flag=pretrained_flag)
    elif model_key == 'ResUNet':
        assert backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101']
        model = ResUNet(in_ch, 
                        n_classes, 
                        backbone=backbone, 
                        pretrained_flag=pretrained_flag)
    elif model_key == 'SegHRNet':
        assert backbone in ['hr-w18', 'hr-w32', 'hr-w48']
        model = SegHRNet(in_ch, 
                         n_classes)
        if pretrained_flag:
            model.init_weights(resume_path)
    elif model_key == 'SegHRNet_OCR':
        assert backbone in ['hr-w18', 'hr-w32', 'hr-w48']
        model = SegHRNet_OCR(in_ch, 
                             n_classes)
        if pretrained_flag:
            model.init_weights(resume_path)
    elif model_key == 'SegHR_LinkNet':
        assert backbone in ['hr-w18', 'hr-w32', 'hr-w48']
        model = SegHR_LinkNet(in_ch, 
                              n_classes)
        if pretrained_flag:
            model.init_weights(resume_path)
    elif model_key == 'SegHR_LinkNet_SUA':
        assert backbone in ['hr-w18', 'hr-w32', 'hr-w48']
        model = SegHR_LinkNet_SUA(in_ch, 
                                  n_classes,
                                  backbone,
                                  vis)
        if pretrained_flag:
            model.init_weights(resume_path)
    elif model_key == 'Swin_LinkNet':
        assert backbone in ['swin-t', 'swin-s', 'swin-b', 'swin-l']
        if pretrained_flag:
            model = Swin_LinkNet(n_classes, in_ch, resume_path, backbone=backbone)
        else:
            model = Swin_LinkNet(n_classes, in_ch, backbone=backbone)
    elif model_key == 'SIINet':
        assert backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101']
        assert in_ch == 3
        model = SIINet_model.build_model(num_classes=n_classes, net_num='44330020', pretrained_flag=pretrained_flag)
    elif model_key == 'BT_RoadNet':
        model = BT_RoadNet_model.build_model(in_ch, 5, n_classes)
    elif model_key == 'GAMSNet':
        model = GAMSNet_model.build_model(n_classes)
    elif model_key == 'DiResNet':
        model = DiResNet_model.build_model(in_ch, n_classes, pretrained=pretrained_flag)
    elif model_key == 'JointNet':
        model = JointNet_model.build_model(in_ch, n_classes)
    elif model_key == 'PointFlow':
        model = PointFlow_model.build_model(n_classes, pretrained=pretrained_flag)
    elif model_key == 'DLinkNet_SUA':
        assert backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101']
        model = DLinkNet_SUA(n_classes, 
                             in_ch, 
                             backbone=backbone, 
                             pretrained_flag=pretrained_flag)
    elif model_key == 'Swin_LinkNet_SUA':
        assert backbone in ['swin-t', 'swin-s', 'swin-b', 'swin-l']
        if pretrained_flag:
            model = Swin_LinkNet_SUA(n_classes, in_ch, resume_path, backbone=backbone)
        else:
            model = Swin_LinkNet_SUA(n_classes, in_ch, backbone=backbone)
    else:
        pass

    return model

from code.models.SegHRNet_DA import SegHRNet_DA
from code.models.Swin_UperNet_DA import Swin_UperNet_DA
from code.models.SegHR_UperNet_OCR_DA import SegHR_UperNet_OCR_DA

def build_uda_model(in_ch, 
                    n_classes, 
                    model_key='', 
                    backbone='resnet34', 
                    pretrained_flag=False, 
                    resume_path=''):
    assert model_key in ['SegHRNet_DA', 'Swin_UperNet_DA', 'SegHR_UperNet_OCR_DA'], '%s not registered in models' % model_key
    if model_key == 'SegHRNet_DA':
        assert backbone in ['hr-w18', 'hr-w32', 'hr-w48']
        model = SegHRNet_DA(in_ch, 
                            n_classes,
                            backbone)
        if pretrained_flag:
            model.init_weights(resume_path)
    elif model_key == 'Swin_UperNet_DA':
        assert backbone in ['swin-t', 'swin-s', 'swin-b', 'swin-l']
        if pretrained_flag:
            model = Swin_UperNet_DA(n_classes, in_ch, resume_path, backbone=backbone)
        else:
            model = Swin_UperNet_DA(n_classes, in_ch, backbone=backbone)
    elif model_key == 'SegHR_UperNet_OCR_DA':
        assert backbone in ['hr-w18', 'hr-w32', 'hr-w48']
        model = SegHR_UperNet_OCR_DA(in_ch, 
                                     n_classes,
                                     backbone)
        if pretrained_flag:
            model.init_weights(resume_path)
    else:
        pass

    return model