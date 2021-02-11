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
        self.config = config
        self.model = nn.Sequential(
            Maxout(config['feature_dim'], config['D_h_size'], 5),
            Maxout(config['D_h_size'], config['D_h_size'], 5),
            Maxout(config['D_h_size'], config['D_h_size'], 5),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(config['D_h_size'], config['num_outcomes'])
        )

        self.classifier = nn.Sequential(
            nn.Linear(config['D_h_size'], config['n_class']),
        )

    def forward(self, x, return_feature=False):
        # [batch, D_h_size]
        f_vector = self.model(x)
        # [batch, num_outcomes]
        discriminator_output = self.discriminator(f_vector)
        discriminator_output = discriminator_output.view(-1,self.config['num_outcomes'])
        # [batch, n_class]
        classification_output = self.classifier(f_vector)
        if return_feature:
            return f_vector, discriminator_output, classification_output
        return discriminator_output, classification_output

    def detect_only(self, x, return_feature=False):
        """只进行OOD判别"""
        # [batch, D_h_size]
        f_vector = self.model(x)
        # [batch, num_outcomes]
        discriminator_output = self.discriminator(f_vector)
        if return_feature:
            return f_vector, discriminator_output
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


