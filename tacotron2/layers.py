import torch


class TruncatedSVDLinear(torch.nn.Module):
    def __init__(self, layer, use_bias=True):
        """Applies SVD to the trained layer given as an input for class constructor."""
        self.use_bias = use_bias
        
        bias = layer.linear_layer.bias.data if self.use_bias else None
        W = layer.linear_layer.weight.data
        
        U, s, V = torch.svd(W)
        self.US = torch.nn.Parameter((U @ torch.diag(S)).transpose(1, 0))
        self.V = torch.nn.Parameter(V)
        self.bias = torch.Parameter(bias) if self.use_bias else None

    def forward(self, x):
        return x @ self.V @ self.US + (self.bias if self.use_bias else 0)
    

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))
        
    def forward(self, x):
        return self.linear_layer(x)
    

class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal
