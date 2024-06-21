from torch import nn
import torch

class ScaledParallelAdapter(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        bottleneck_dim: int,
        scaling_factor: float = 1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.bottleneck_dim = bottleneck_dim
        self.scaling_factor = scaling_factor
        
        self.down_proj = nn.Linear(embed_dim, bottleneck_dim)
        self.relu = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck_dim, embed_dim)
        
    def forward(self, x: torch.Tensor, y: torch.tensor) -> torch.Tensor:
        # x is the input to the layer that we are modifying
        # y is the output of the later that we are modifying
        adapter_out = self.up_proj(self.relu(self.down_proj(x)))
        return y + self.scaling_factor * adapter_out