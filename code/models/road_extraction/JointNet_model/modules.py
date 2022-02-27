import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

class basic_conv(nn.Module):
    def __init__(self, in_ch, out_ch, group=8, dilation_rate=1, memory_efficient=True):
        super(basic_conv, self).__init__()
        self.memory_efficient = memory_efficient
        self.conv = nn.Conv2d(in_ch, out_ch, 3, dilation=dilation_rate, padding=dilation_rate)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.GroupNorm(group, out_ch, eps=0.1)
    
    def bn_function(self, inputs):
        concated_features = torch.cat(inputs, dim=1)
        bottleneck_output = self.norm(self.relu(self.conv(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        pass
    
    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, torch.Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        return bottleneck_output

class Dense_Atrous_Convs_Block(nn.Module):
    def __init__(self, in_ch, growth_rate, group=8, downsample=True):
        super(Dense_Atrous_Convs_Block, self).__init__()
        self.basic_conv_list = nn.ModuleList([])
        self.basic_conv_list.append(basic_conv(in_ch, growth_rate, group=group, dilation_rate=1))
        self.basic_conv_list.append(basic_conv(in_ch + growth_rate, growth_rate, group=group, dilation_rate=2))
        self.basic_conv_list.append(basic_conv(in_ch + growth_rate * 2, growth_rate, group=group, dilation_rate=5))
        self.basic_conv_list.append(basic_conv(in_ch + growth_rate * 3, growth_rate, group=group, dilation_rate=1))
        self.basic_conv_list.append(basic_conv(in_ch + growth_rate * 4, growth_rate, group=group, dilation_rate=2))
        self.basic_conv_list.append(basic_conv(in_ch + growth_rate * 5, growth_rate, group=group, dilation_rate=5))

        self.conv_f = nn.Sequential(
            nn.Conv2d(in_ch + growth_rate * 6, growth_rate * 4, 1),
            nn.ReLU(inplace=True)
        )

        self.downsample = downsample
        if self.downsample:
            self.conv_d = nn.Sequential(
                nn.Conv2d(growth_rate * 4, growth_rate, 1, stride=2),
                nn.ReLU(inplace=True)
            )
    
    def dense_forward(self, input):
        features = [input]
        for layer in self.basic_conv_list:
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

    def forward(self, input):
        out = self.conv_f(self.dense_forward(input))
        if self.downsample:
            out_d = self.conv_d(out)
            return out, out_d
        else:
            return out