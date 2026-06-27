import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GCNConv


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return self.ln(x)


class GraphSAGEBackbone(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, out_channels: int = 2,
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(LayerNorm(hidden_channels))
        self.residuals.append(nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else nn.Identity())
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(LayerNorm(hidden_channels))
            self.residuals.append(nn.Identity())
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(x, edge_index)
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = self.residuals[i](x) + h
        return self.convs[-1](x, edge_index)

    def forward_with_dropout(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(x, edge_index)
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=True)
            x = self.residuals[i](x) + h
        return self.convs[-1](x, edge_index)


class GATBackbone(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, out_channels: int = 2,
                 num_layers: int = 3, dropout: float = 0.3, heads: int = 4):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads, concat=True))
        self.norms.append(LayerNorm(hidden_channels))
        self.residuals.append(nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else nn.Identity())
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads, concat=True))
            self.norms.append(LayerNorm(hidden_channels))
            self.residuals.append(nn.Identity())
        self.convs.append(GATConv(hidden_channels, out_channels, heads=1, concat=False))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(x, edge_index)
            h = self.norms[i](h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = self.residuals[i](x) + h
        return self.convs[-1](x, edge_index)

    def forward_with_dropout(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(x, edge_index)
            h = self.norms[i](h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=True)
            x = self.residuals[i](x) + h
        return self.convs[-1](x, edge_index)


class GCNBackbone(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, out_channels: int = 2,
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.norms.append(LayerNorm(hidden_channels))
        self.residuals.append(nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else nn.Identity())
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(LayerNorm(hidden_channels))
            self.residuals.append(nn.Identity())
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(x, edge_index)
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = self.residuals[i](x) + h
        return self.convs[-1](x, edge_index)

    def forward_with_dropout(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(x, edge_index)
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=True)
            x = self.residuals[i](x) + h
        return self.convs[-1](x, edge_index)


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
        if hasattr(backbone, 'residuals') and len(backbone.residuals) >= len(backbone.convs) + 1:
            backbone.residuals = backbone.residuals[:len(backbone.convs)]
        self.evidence_layer = nn.Linear(in_features if in_features != num_classes else 128, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.backbone.convs):
            h = conv(x, edge_index)
            if hasattr(self.backbone, 'norms') and i < len(self.backbone.norms):
                h = self.backbone.norms[i](h)
            act = F.relu if not isinstance(self.backbone, GATBackbone) else F.elu
            h = act(h)
            h = F.dropout(h, p=self.backbone.dropout, training=self.training)
            x = (self.backbone.residuals[i](x) + h) if hasattr(self.backbone, 'residuals') and i < len(self.backbone.residuals) else h
        evidence = F.softplus(self.evidence_layer(x))
        return evidence + 1.0

    def forward_with_dropout(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.backbone.convs):
            h = conv(x, edge_index)
            if hasattr(self.backbone, 'norms') and i < len(self.backbone.norms):
                h = self.backbone.norms[i](h)
            act = F.relu if not isinstance(self.backbone, GATBackbone) else F.elu
            h = act(h)
            h = F.dropout(h, p=self.backbone.dropout, training=True)
            x = (self.backbone.residuals[i](x) + h) if hasattr(self.backbone, 'residuals') and i < len(self.backbone.residuals) else h
        evidence = F.softplus(self.evidence_layer(x))
        return evidence + 1.0
