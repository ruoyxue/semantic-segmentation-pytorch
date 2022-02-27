from .JointNet import JointNet

def build_model(in_ch, out_ch):
    return JointNet(in_ch, out_ch)