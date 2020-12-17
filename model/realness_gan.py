# coding: utf-8
from torch import nn


class Maxout(nn.Module):
    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)

    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m

class Discriminator(nn.Module):
    def __init__(self, config: dict):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            Maxout(config['feature_dim'], config['D_h_size'], 5),
            Maxout(config['D_h_size'], config['D_h_size'], 5),
            Maxout(config['D_h_size'], config['D_h_size'], 5),
            nn.Linear(config['D_h_size'], config['num_outcomes'])
        )
        self.config = config

    def forward(self, x):
        discriminator_output = self.model(x).view(-1,self.config['num_outcomes'])
        return discriminator_output


class Generator(nn.Module):
    def __init__(self, config: dict):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(config['G_z_dim'], config['G_h_size']),
            nn.BatchNorm1d(config['G_h_size']),
            nn.ReLU(),
            nn.Linear(config['G_h_size'], config['G_h_size']),
            nn.BatchNorm1d(config['G_h_size']),
            nn.ReLU(),
            nn.Linear(config['G_h_size'], config['G_h_size']),
            nn.BatchNorm1d(config['G_h_size']),
            nn.ReLU(),
            nn.Linear(config['G_h_size'], config['G_h_size']),
            nn.BatchNorm1d(config['G_h_size']),
            nn.ReLU(),
            nn.Linear(config['G_h_size'], config['feature_dim'])
        )

    def forward(self, z):
        # [batch, feature_dim]
        feature_vector = self.model(z)
        return feature_vector


if __name__ == '__main__':
    D_config = {'feature_dim': 768, 'Wf_dim': 512, 'n_class': 2}
    D = Discriminator(D_config)
    print(D)

    G_config = {'feature_dim': 768, 'z_dim': 2}
    G = Generator(G_config)
    print(G)
