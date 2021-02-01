import torch
import torch.nn as nn
import numpy as np

class Autorec(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(Autorec, self).__init__()

        self.args = args
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_units = args.hidden_units
        self.lambda_value = args.lambda_value

        self.encoder = nn.Sequential(
            nn.Linear(self.num_items, self.hidden_units),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_units, self.num_items),
        )

    def forward(self, torch_input):
        encoder = self.encoder(torch_input)
        decoder = self.decoder(encoder)

        return decoder

    def loss(self, decoder, input, optimizer, mask_input):
        cost = 0
        temp2 = 0

        cost += ((decoder - input) * mask_input).pow(2).sum()
        rmse = cost

        for i in optimizer.param_groups:
            for j in i['params']:
                # print(type(j.data), j.shape,j.data.dim())
                if j.data.dim() == 2:
                    temp2 += torch.t(j.data).pow(2).sum()

        cost += temp2 * self.lambda_value * 0.5
        return cost, rmse

    def saveModel(self, location):
        torch.save(self.state_dict(), location)

    def loadModel(self, path, map_location):
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict, strict=False)

    def recommend_user(self, r_u):
        p_rate_dict = dict()
        predict = self.forward(torch.from_numpy(r_u).float())
        predict = predict.detach().numpy()
        for i in range(len(predict)):
            p_rate_dict[i] = predict[i]

        return p_rate_dict
