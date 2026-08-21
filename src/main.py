import torch
import torch.nn as nn

# straight linear reference path
# interpolate linear
# xt any point in vector field that satisfies ODE
def interpolate_linear(x0, x1, t):
    '''Evaluates the linear interpolation path between x0 and x1 at step t.'''
    xt = ((1 - t) * x0) + (t * x1)
    return xt

def get_target_velocity(x0, x1):
    '''Get the velocity for a given pair of noise and target points.
    This is the per-pair (conditional) velocity along the straight path.
    '''
    return x1 - x0

# define flow matching model class
class FlowMatchingModel(nn.Module):
    '''
        Flow Matching Model to predict the velocity field at time t and position xt
    '''

    def __init__(self, data_dim:int, hidden_dim:int) -> None:
        super().__init__()

        # MLP
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.GELU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def forward(self, t:torch.Tensor, xt:torch.Tensor) -> torch.Tensor:
       ''' Predicts the velocity field at time t and position xt '''
       t_concat_xt: torch.Tensor = torch.cat([t, xt], dim=-1)
       return self.net(t_concat_xt)


