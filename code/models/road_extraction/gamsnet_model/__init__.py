from .gamsnet import GAMSNet

def build_model(out_ch):
    return GAMSNet(out_ch)