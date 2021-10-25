import torch
import torch.nn as nn # Neural Network implementations in PyTorch
from torch_scatter import scatter_mean # Fast computation of node group mean features
from torch_geometric.nn import MetaLayer

class GNNModel(nn.Module):
    def __init__(self, node_input, edge_input, node_output=64, edge_output=64, leakiness=0.1, num_mp=3):
        super(GNNModel, self).__init__()

        # Initialize the updaters
        self.message_passing = nn.ModuleList()

        # Update the node and edge feature N times (number of message passings, here = 3)
        self.num_mp = num_mp # Number of message passings

        for i in range(self.num_mp):
            self.message_passing.append(
                MetaLayer(
                    edge_model = EdgeLayer(node_input, edge_input, edge_output, leakiness=leakiness),
                    node_model = NodeLayer(node_input, node_output, edge_output, leakiness=leakiness)
                )
            )
            node_input = node_output
            edge_input = edge_output

        # Reduce the number of node and edge features edge, as we are performing a simple classification
        self.node_predictor = nn.Linear(node_output, 2)
        self.edge_predictor = nn.Linear(edge_output, 2)

    def forward(self, data):

        # Loop over message passing steps, pass data through the updaters
        x = data.x
        e = data.edge_attr
        if e is None:
            e = torch.empty((data.edge_index.shape[1],0)).to(x.device)
        for i in range(self.num_mp):
            x, e, _ = self.message_passing[i](x, data.edge_index, e, batch=data.batch)

        # Reduce output features to 2 each
#         x_pred = self.node_predictor(x)
        e_pred = self.edge_predictor(e)

        # Return
        res = {
#             'node_pred': x_pred,
            'edge_pred': e_pred
            }

        return res

class EdgeLayer(nn.Module):
    def __init__(self, node_in, edge_in, edge_out, leakiness=0.0):
        super(EdgeLayer, self).__init__()
        self.edge_mlp = nn.Sequential(
            nn.BatchNorm1d(2 * node_in + edge_in),
            nn.Linear(2 * node_in + edge_in, edge_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(edge_out),
            nn.Linear(edge_out, edge_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(edge_out),
            nn.Linear(edge_out, edge_out)
        )

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        out = torch.cat([src, dest, edge_attr], dim=1)
        return self.edge_mlp(out)

class NodeLayer(nn.Module):
    def __init__(self, node_in, node_out, edge_in, leakiness=0.0):
        super(NodeLayer, self).__init__()

        self.node_mlp_1 = nn.Sequential(
            nn.BatchNorm1d(node_in + edge_in),
            nn.Linear(node_in + edge_in, node_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(node_out),
            nn.Linear(node_out, node_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(node_out),
            nn.Linear(node_out, node_out)
        )

        self.node_mlp_2 = nn.Sequential(
            nn.BatchNorm1d(node_in + node_out),
            nn.Linear(node_in + node_out, node_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(node_out),
            nn.Linear(node_out, node_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(node_out),
            nn.Linear(node_out, node_out)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1) # Aggregating neighboring node with connecting edges
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0)) # Building mean messages
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)
