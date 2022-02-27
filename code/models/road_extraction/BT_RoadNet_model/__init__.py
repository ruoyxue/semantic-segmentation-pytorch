from .bt_roadnet import BT_RoadNet

def build_model(in_ch, k, out_ch):
    return BT_RoadNet(in_ch, k, out_ch)