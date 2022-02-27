import torch

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, base_pyramid_size):
    H = W = int(torch.sqrt(torch.tensor(windows.size(1)))) * base_pyramid_size
    B = int(windows.size(0) / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def create_scale_mask(pixels, channels, value, suppress_diag=None):
    assert len(pixels) == len(channels)

    def side_mask(mask, start_level, end_level, value):
        assert start_level < end_level
        x_cord_start = start_point[end_level]
        y_cord_start = start_point[start_level]
        for p in range(pixels[end_level]):
            x_cord = x_cord_start + p * channels[end_level]
            y_cord = y_cord_start + p * channels[start_level] * 4 ** (end_level - start_level)
            mask[y_cord: y_cord + channels[start_level] * 4 ** (end_level - start_level),
            x_cord: x_cord + channels[end_level]] = value

    def create_diag_block_mask(mask, level, value):
        cord_start = start_point[level]
        for i in range(pixels[level]):
            mask[cord_start + i * channels[level]: cord_start + (i + 1) * channels[level],
            cord_start + i * channels[level]: cord_start + (i + 1) * channels[level]] = value

    # initial mask
    dim = sum([pixels[i] * channels[i] for i in range(len(pixels))])
    mask = torch.zeros((dim, dim))

    # create start points
    start_point = [0]
    for i in range(len(pixels) - 1):
        start_point.append(pixels[i] * channels[i])
    for i in range(len(start_point)):
        if i > 0:
            start_point[i] += start_point[i - 1]

    # create side-mask
    for i in range(len(pixels)):
        for j in range(i):
            side_mask(mask, j, i, value)

    mask_t = mask.transpose(0, 1)
    f_mask = torch.triu(mask) + torch.tril(mask_t)

    # suppress the diagonal value
    if suppress_diag is not None:
        if suppress_diag == 'Self':
            values = [value for _ in range(dim)]
            diag = torch.diag(torch.tensor(values, dtype=torch.float))
            f_mask += diag
        elif suppress_diag == 'Block':
            for i in range(len(pixels)):
                create_diag_block_mask(f_mask, i, value)
        elif suppress_diag == 'Layer':
            for i in range(len(start_point) - 1):
                f_mask[start_point[i]: start_point[i + 1], start_point[i]: start_point[i + 1]] = value
            f_mask[start_point[-1]:, start_point[-1]:] = value
        else:
            raise ValueError("Invalid indicator for 'suppress_diag', which should be 'Self', 'Block' or 'Layer'.")

    return f_mask

def create_spatial_mask(num_pixels, value):

    # initial mask
    mask = torch.zeros((num_pixels, num_pixels))

    # create diag mask
    values = [value for _ in range(num_pixels)]
    diag = torch.diag(torch.tensor(values, dtype=torch.float))
    mask += diag
    return mask

if __name__ == '__main__':
    device = torch.device('cpu')
    l1 = torch.autograd.Variable(torch.randn(2, 64, 128, 128)).to(device)
    l2 = torch.autograd.Variable(torch.randn(2, 64, 64, 64)).to(device)
    l3 = torch.autograd.Variable(torch.randn(2, 64, 32, 32)).to(device)
    l4 = torch.autograd.Variable(torch.randn(2, 64, 16, 16)).to(device)

    w1 = window_partition(l1.permute(0,2,3,1), 8)
    w2 = window_partition(l2.permute(0,2,3,1), 4)
    w3 = window_partition(l3.permute(0,2,3,1), 2)
    w4 = window_partition(l4.permute(0,2,3,1), 1)
    print(w1.size())
    print(w2.size())
    print(w3.size())
    print(w4.size())
    print(l1[0, 0, 0:8, 0:8])
    print(w1[0, :, 0])
    print(l2[0, 0, 0:4, 0:4])
    print(w2[0, :, 0])
    print(l3[0, 0, 0:2, 0:2])
    print(w3[0, :, 0])
    print(l4[0, 0, 0:1, 0:1])
    print(w4[0, :, 0])
    w = torch.cat([w1, w2, w3, w4], dim=1)
    print(w.size())
    print(w[0, :, 0])