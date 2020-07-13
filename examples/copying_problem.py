import torch
from torch import nn
import torch.nn.functional as F

import geotorch
import geotorch.parametrize as P

batch_size = 128
hidden_size = 190
iterations = 4000  # Training iterations
L = 1000  # Length of sequence before asking to remember
K = 10  # Length of sequence to remember
n_classes = 9  # Number of possible classes
lr = 1e-3
lr_orth = 2e-4
device = torch.device("cuda")
# When RGD == True we perform Riemannian gradient descent
# This is to demonstrate how one may implement RGD with
# just one extra line of code.
# RGD does not perform very well in these problems though.
RGD = False


class modrelu(nn.Module):
    def __init__(self, features):
        # For now we just support square layers
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
        # Initialize the recurrent kernel
        self.recurrent_kernel.parametrizations.weight.torus_init_()

    def default_hidden(self, input):
        return input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)

    def forward(self, input, hidden):
        input = self.input_kernel(input)
        hidden = self.recurrent_kernel(hidden)
        out = input + hidden
        out = self.nonlinearity(out)

        return out, out


class Model(nn.Module):
    def __init__(self, n_classes, hidden_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = ExpRNNCell(n_classes + 1, hidden_size)
        self.lin = nn.Linear(hidden_size, n_classes)
        self.loss_func = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.lin.weight.data, nonlinearity="relu")
        nn.init.constant_(self.lin.bias.data, 0)

    def forward(self, inputs):
        state = self.rnn.default_hidden(inputs[:, 0, ...])
        outputs = []
        with P.cached():
            for input in torch.unbind(inputs, dim=1):
                out_rnn, state = self.rnn(input, state)
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
    # We generate K of them and we have to recall them
    # L is the waiting between the last number and the
    # signal to start outputting the numbers
    # We codify `-` as a 0 and `:` as a 9.

    seq = torch.randint(1, n_classes, (batch_size, K), dtype=torch.long, device=device)
    zeros1 = torch.zeros((batch_size, L), dtype=torch.long, device=device)
    zeros2 = torch.zeros((batch_size, K - 1), dtype=torch.long, device=device)
    zeros3 = torch.zeros((batch_size, K + L), dtype=torch.long, device=device)
    marker = torch.full((batch_size, 1), n_classes, dtype=torch.long, device=device)

    x = torch.cat([seq, zeros1, marker, zeros2], dim=1)
    y = torch.cat([zeros3, seq], dim=1)

    return x, y


def main():
    model = Model(n_classes, hidden_size).to(device)

    p_orth = model.rnn.recurrent_kernel
    orth_params = p_orth.parameters()
    non_orth_params = (
        param for param in model.parameters() if param not in set(p_orth.parameters())
    )

    if RGD:
        # Implement Stochstic Riemannian Gradient Descent via SGD
        optim = torch.optim.SGD(
            [{"params": non_orth_params}, {"params": orth_params, "lr": lr_orth}], lr=lr
        )
    else:
        # These recurrent models benefit of slightly larger mixing constants
        # on the adaptive term. They also work with beta_2 = 0.999, but they
        # give better results with beta_2 = 0.99 or even 0.95
        optim = torch.optim.Adam(
            [
                {"params": non_orth_params},
                {"params": orth_params, "lr": lr_orth, "betas": (0.9, 0.99)},
            ],
            lr=lr,
        )

    model.train()
    for step in range(iterations):
        batch_x, batch_y = copy_data(batch_size)
        x_onehot = F.one_hot(batch_x, num_classes=n_classes + 1).float()
        logits = model(x_onehot)
        loss = model.loss(logits, batch_y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if RGD:
            # Updating the base after every step and using SGD gives us
            # Riemannian Gradient Descent. More on this in Section 5
            # https://arxiv.org/abs/1909.09501
            model.rnn.recurrent_kernel.parametrizations.weight.update_base()

        with torch.no_grad():
            accuracy = model.accuracy(logits, batch_y)

        print("Iter {} Loss: {:.6f}, Accuracy: {:.5f}".format(step, loss, accuracy))

    # The evaluation in this model is not quite necessary, as we do not repeat any
    # element of the training batch, but we leave it for completeness
    model.eval()
    with torch.no_grad():
        test_x, test_y = copy_data(batch_size)
        x_onehot = F.one_hot(test_x, num_classes=n_classes + 1).float()
        logits = model(x_onehot)
        loss = model.loss(logits, test_y)
        accuracy = model.accuracy(logits, test_y)
        print("Test result. Loss: {:.6f}, Accuracy: {:.5f}".format(loss, accuracy))


if __name__ == "__main__":
    main()
