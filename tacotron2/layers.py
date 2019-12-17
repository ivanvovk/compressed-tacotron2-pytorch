import torch
from torch.nn import Parameter, ParameterList

import tntorch as tn
    

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


class TruncatedSVDLinear(torch.nn.Module):
    def __init__(self, layer, explained_variance=0.7):
        """Applies SVD to the trained layer given as an input for class constructor."""
        super(TruncatedSVDLinear, self).__init__()
        
        self.bias = layer.bias
        W = layer.weight.data
        
        U, s, V = torch.svd(W)
        
        self.rank = (torch.cumsum(s / s.sum(), dim=-1) < explained_variance).int().sum()
        U, s, V = U[:, :self.rank], s[:self.rank], V[:, :self.rank]
        
        self.US = torch.nn.Parameter((U @ torch.diag(s)).transpose(1, 0))
        self.V = torch.nn.Parameter(V)

    def forward(self, x):
        return x @ self.V @ self.US \
            + (self.bias if type(self.bias).__name__ != 'NoneType' else 0)


from enum import IntEnum

class DIMS(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class TTLSTM(torch.nn.Module):
    def __init__(self, lstm_layer, ranks_tt=10):
        """LSTM class wrapper with tensor-trained weights."""
        super(TTLSTM, self).__init__()
        
        self.input_size = lstm_layer.input_size
        self.hidden_size = lstm_layer.hidden_size
        self.num_layers = lstm_layer.num_layers
        self.bias = lstm_layer.bias
        self.bidirectional = lstm_layer.bidirectional

        self.weight_ih = ParameterList([
            Parameter(core) \
                for core in tn.Tensor(lstm_layer.weight_ih_l0.data.T,
                    ranks_tt=ranks_tt).cores
        ])
        self.weight_hh = ParameterList([
            Parameter(core) \
                for core in tn.Tensor(lstm_layer.weight_hh_l0.data.T,
                    ranks_tt=ranks_tt).cores
        ])
        if self.bias:
            self.bias_ih = Parameter(lstm_layer.bias_ih_l0.data)
            self.bias_hh = Parameter(lstm_layer.bias_hh_l0.data)

        if self.bidirectional:
            self.weight_ih_reverse = ParameterList([
                Parameter(core) \
                    for core in tn.Tensor(lstm_layer.weight_ih_l0_reverse.data.T,
                        ranks_tt=ranks_tt).cores
            ])
            self.weight_hh_reverse = ParameterList([
                Parameter(core) \
                    for core in tn.Tensor(lstm_layer.weight_hh_l0_reverse.data.T,
                        ranks_tt=ranks_tt).cores
            ])
            if self.bias:
                self.bias_ih_reverse = Parameter(lstm_layer.bias_ih_l0_reverse.data)
                self.bias_hh_reverse = Parameter(lstm_layer.bias_hh_l0_reverse.data)

    def nparams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def flatten_parameters(self):
        pass

    def __restore_weight(self, cores):
        return torch.einsum('amc,cna->mn', *cores)
     
    def __impl(self, x, init_states=None, backward=False):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(self.hidden_size).to(x.device), 
                        torch.zeros(self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]

            # batch the computations into a single matrix multiplication
            weight_ih = self.weight_ih if not backward else self.weight_ih_reverse
            weight_hh = self.weight_hh if not backward else self.weight_hh_reverse
            bias_ih = self.bias_ih if not backward else self.bias_ih_reverse
            bias_hh = self.bias_hh if not backward else self.bias_hh_reverse

            gates = x_t @ self.__restore_weight(weight_ih) + bias_ih \
                + h_t @ self.__restore_weight(weight_hh) + bias_hh
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(DIMS.batch))

        hidden_seq = torch.cat(hidden_seq, dim=DIMS.batch)

        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(DIMS.batch, DIMS.seq).contiguous()
        return hidden_seq, (h_t, c_t)

    def forward(self, x, init_states=None):
        outputs_forward, (h_n_forward, c_n_forward) = self.__impl(
            x, init_states=init_states, backward=False
        )
        if self.bidirectional:
            outputs_backward, (h_n_backward, c_n_backward) = self.__impl(
                x[:, range(x.shape[1] - 1, -1, -1), :], init_states=init_states, backward=True
            )
            return torch.cat([outputs_forward, outputs_backward], -1), (
                torch.cat([h_n_forward, h_n_backward], 0),
                torch.cat([c_n_forward, c_n_backward], 0)
            )
        return outputs_forward, (h_n_forward, c_n_forward)
