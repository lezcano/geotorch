"""
Basic example of usage of GeoTorch

Implements a constrained RNN to learn a synthetic regression problem which asks to recall
some inputs and output them. From a dictionary of 9 numbers, the input-output looks like follows:

Input:   14221----------:----
Output:  ---------------14221

This class should converge to 0% error. When at 0% error, sometimes there are some instabilities.

The GeoTorch code happens in `ExpRNNCell.__init__`, `ExpRNNCell.reset_parameters` and line 107.
The rest of the code is normal PyTorch.
Lines 146-167 shows how to assign different learning rates to parametrized weights.

This file also implements in lines 152 and 180 Riemannian Gradient Descent (RGD). As shown, RGD
dynamics account for using SGD as the optimizer calling `update_basis()` after every optization step.
"""

import torch
from torch import nn
import torch.nn.functional as F

import geotorch

batch_size = 128
hidden_size = 190
iterations = 4000  # Training iterations
L = 1000  # Length of sequence before asking to remember
S = 10  # Length of sequence to remember
alphabet_size = 8
lr = 1e-3
lr_orth = 2e-4
device = torch.device("cuda")
# When RGD == True we perform Riemannian gradient descent
# This is to demonstrate how one may implement RGD with
# just one extra line of code.
# RGD does not perform very well in these problems though.
RGD = False
if RGD:
    print(
        "Optimizing using RGD. The perfomance will be _much_ worse than with Adam or RMSprop."
    )


class modrelu(nn.Module):
    def __init__(self, features):
        super(modrelu, self).__init__()
        self.features = features
        self.b = nn.Parameter(torch.Tensor(self.features))
        self.reset_parameters()

    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)

    def forward(self, inputs):
        norm = torch.abs(inputs)
        biased_norm = norm + self.b
        magnitude = nn.functional.relu(biased_norm)
        phase = torch.sign(inputs)

        return phase * magnitude


class ExpRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ExpRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_kernel = nn.Linear(hidden_size, hidden_size, bias=False)
        self.input_kernel = nn.Linear(input_size, hidden_size)
        self.nonlinearity = modrelu(hidden_size)

        # Make recurrent_kernel orthogonal
        geotorch.orthogonal(self.recurrent_kernel, "weight")

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.input_kernel.weight.data, nonlinearity="relu")
        # The manifold class is under `layer.parametrizations.tensor_name[0]`
        M = self.recurrent_kernel.parametrizations.weight[0]
        # Every manifold has a convenience sample method, but you can use your own initializer
        self.recurrent_kernel.weight = M.sample("torus")

    def default_hidden(self, input_):
        return input_.new_zeros(input_.size(0), self.hidden_size, requires_grad=False)

    def forward(self, input_, hidden):
        input_ = self.input_kernel(input_)
        hidden = self.recurrent_kernel(hidden)
        out = input_ + hidden
        return self.nonlinearity(out)


class Model(nn.Module):
    def __init__(self, alphabet_size, hidden_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = ExpRNNCell(alphabet_size + 2, hidden_size)
        self.lin = nn.Linear(hidden_size, alphabet_size + 1)
        self.loss_func = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.lin.weight.data, nonlinearity="relu")
        nn.init.constant_(self.lin.bias.data, 0)

    def forward(self, inputs):
        out_rnn = self.rnn.default_hidden(inputs[:, 0, ...])
        outputs = []
        with geotorch.parametrize.cached():
            for input in torch.unbind(inputs, dim=1):
                out_rnn = self.rnn(input, out_rnn)
                outputs.append(self.lin(out_rnn))
        return torch.stack(outputs, dim=1)

    def loss(self, logits, y):
        return self.loss_func(logits.view(-1, 9), y.view(-1))

    def accuracy(self, logits, y):
        return torch.eq(torch.argmax(logits, dim=2), y).float().mean()


def copy_data(batch_size):
    # Generates some random synthetic data
    # Example of input-output sequence
    # 14221----------:----
    # ---------------14221
    # Numbers go from 1 to 8
    # We generate S of them and we have to recall them
    # L is the waiting between the last number and the
    # signal to start outputting the numbers
    # We codify `-` as a 0 and `:` as a 9.

    seq = torch.randint(
        1, alphabet_size + 1, (batch_size, S), dtype=torch.long, device=device
    )
    zeros1 = torch.zeros((batch_size, L), dtype=torch.long, device=device)
    zeros2 = torch.zeros((batch_size, S - 1), dtype=torch.long, device=device)
    zeros3 = torch.zeros((batch_size, S + L), dtype=torch.long, device=device)
    marker = torch.full(
        (batch_size, 1), alphabet_size + 1, dtype=torch.long, device=device
    )

    x = torch.cat([seq, zeros1, marker, zeros2], dim=1)
    y = torch.cat([zeros3, seq], dim=1)

    return x, y


def main():
    model = Model(alphabet_size, hidden_size).to(device)

    p_orth = model.rnn.recurrent_kernel
    orth_params = p_orth.parameters()
    non_orth_params = (
        p for p in model.parameters() if p not in set(p_orth.parameters())
    )

    if RGD:
        # Implement Stochstic Riemannian Gradient Descent via SGD
        optim = torch.optim.SGD(
            [{"params": non_orth_params}, {"params": orth_params, "lr": lr_orth}], lr=lr
        )
    else:
        # These recurrent models benefit of slightly larger mixing constants
        # on the adaptive term. They also work with beta_2 = 0.999, but they
        # give better results with beta_2 \in [0.9, 0.99]
        optim = torch.optim.Adam(
            [
                {"params": non_orth_params},
                {"params": orth_params, "lr": lr_orth, "betas": (0.9, 0.95)},
            ],
            lr=lr,
        )

    model.train()
    for step in range(iterations):
        batch_x, batch_y = copy_data(batch_size)
        x_onehot = F.one_hot(batch_x, num_classes=alphabet_size + 2).float()
        logits = model(x_onehot)
        loss = model.loss(logits, batch_y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if RGD:
            # Updating the base after every step and using SGD gives us
            # Riemannian Gradient Descent. More on this in Section 5
            # https://arxiv.org/abs/1909.09501
            geotorch.update_base(model.rnn.recurrent_kernel, "weight")

        with torch.no_grad():
            accuracy = model.accuracy(logits, batch_y)

        print("Iter {} Loss: {:.6f}, Accuracy: {:.5f}".format(step, loss, accuracy))

    # The evaluation in this model is not quite necessary, as we do not repeat any
    # element of the training batch


if __name__ == "__main__":
    main()
