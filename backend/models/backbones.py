import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, GraphNorm, InstanceNorm


class GraphSAGEBackbone(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, out_channels: int = 2,
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self._init_weights()

    def _init_weights(self):
        for conv in self.convs:
            if hasattr(conv, 'lin_l'):
                nn.init.xavier_uniform_(conv.lin_l.weight)
            if hasattr(conv, 'lin_r'):
                nn.init.xavier_uniform_(conv.lin_r.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def forward_with_dropout(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=True)
        x = self.convs[-1](x, edge_index)
        return x


class GATBackbone(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, out_channels: int = 2,
                 num_layers: int = 3, dropout: float = 0.3, heads: int = 4):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))
        self.norms.append(GraphNorm(hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))
            self.norms.append(GraphNorm(hidden_channels * heads))
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False))
        self._init_weights()

    def _init_weights(self):
        for conv in self.convs:
            if hasattr(conv, 'lin_src'):
                nn.init.xavier_uniform_(conv.lin_src.weight)
            if hasattr(conv, 'lin_dst') and conv.lin_dst is not None:
                nn.init.xavier_uniform_(conv.lin_dst.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def forward_with_dropout(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=True)
        x = self.convs[-1](x, edge_index)
        return x


class GCNBackbone(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, out_channels: int = 2,
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.norms.append(InstanceNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(InstanceNorm(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self._init_weights()

    def _init_weights(self):
        for conv in self.convs:
            if hasattr(conv, 'lin'):
                nn.init.uniform_(conv.lin.weight, -0.1, 0.1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def forward_with_dropout(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=True)
        x = self.convs[-1](x, edge_index)
        return x


class EDLWrapper(nn.Module):
    """Wraps a GNN backbone to output Dirichlet evidence instead of logits."""
    def __init__(self, backbone: nn.Module, num_classes: int = 2):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        last_conv = backbone.convs[-1]
        if hasattr(last_conv, 'out_channels'):
            in_features = last_conv.out_channels
        else:
            in_features = num_classes
        backbone.convs = backbone.convs[:-1]
        if hasattr(backbone, 'norms') and len(backbone.norms) >= len(backbone.convs) + 1:
            backbone.norms = backbone.norms[:len(backbone.convs)]
        self.evidence_layer = nn.Linear(in_features if in_features != num_classes else 128, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.backbone.convs):
            x = conv(x, edge_index)
            if hasattr(self.backbone, 'norms') and i < len(self.backbone.norms):
                x = self.backbone.norms[i](x)
            x = F.relu(x) if not isinstance(self.backbone, GATBackbone) else F.elu(x)
            x = F.dropout(x, p=self.backbone.dropout, training=self.training)
        evidence = F.softplus(self.evidence_layer(x))
        alpha = evidence + 1.0
        return alpha

    def forward_with_dropout(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.backbone.convs):
            x = conv(x, edge_index)
            if hasattr(self.backbone, 'norms') and i < len(self.backbone.norms):
                x = self.backbone.norms[i](x)
            x = F.relu(x) if not isinstance(self.backbone, GATBackbone) else F.elu(x)
            x = F.dropout(x, p=self.backbone.dropout, training=True)
        evidence = F.softplus(self.evidence_layer(x))
        alpha = evidence + 1.0
        return alpha
