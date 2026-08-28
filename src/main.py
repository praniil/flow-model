# Imports and setup
import base64
import functools
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from IPython.display import HTML, display
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

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


def compute_loss(flow_matching_model:FlowMatchingModel, x0:torch.Tensor, x1:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
    '''
    Compute the loss for a single batch of (X0, X1) couplings and flow steps T
    '''
    # interpolate the data at the sampled time step
    xt = interpolate_linear(x0=x0, x1=x1, t=t)

    # get the target velocith
    v_target = get_target_velocity(x0=x0, x1=x1)

    # predict the velocity
    v_pred = flow_matching_model(t=t, xt=xt)

    # compute the loss
    loss = ((v_pred - v_target) ** 2).mean()

    return loss

# Train the flow matching model

#Hyperparameters
data_dim: int  = 1 # one D data
hidden_dim:int = 64
train_iteration = 10000
learning_rate = 1e-3
batch_size = 256

DEVICE : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(626)

# init the vector field network and optimizer
flow_matching_model = FlowMatchingModel(data_dim=data_dim, hidden_dim=hidden_dim).to(DEVICE).train()
optimizer = torch.optim.Adam(flow_matching_model.parameters(), lr=learning_rate)


def mixture_sample(size):
    """
    Sample from a 1D mixture of two Gaussian distributions.

    50% of samples come from N(-2, 0.5^2)
    50% of samples come from N(+2, 0.5^2)
    """
    # Choose which Gaussian each sample comes from
    component = np.random.rand(size) < 0.5  # random nuber list of len (size) and if number less than 0.5 -> True

    # Sample from the two Gaussians
    # np
    samples = np.where(
        component,
        np.random.normal(loc=-2.0, scale=0.5, size=size),
        np.random.normal(loc=2.0, scale=0.5, size=size)
    )

    return samples

# Training loop
losses: list[float] = []
with tqdm(range(train_iteration), desc="Training", unit="iteration") as progress_bar:
    for i in progress_bar:
        # Sample a batch of target and noise samples
        x1 = torch.from_numpy(mixture_sample(size=batch_size)).to(dtype=torch.float32, device=DEVICE).unsqueeze(-1)
        x0 = torch.randn_like(x1)
        # Sample a random time step for each sample in the batch
        t = torch.rand(x1.shape[0], device=DEVICE).unsqueeze(-1)

        # Compute the loss
        loss = compute_loss(flow_matching_model=flow_matching_model, x0=x0, x1=x1, t=t)

        # Backpropagate the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        progress_bar.set_postfix({"Loss": f"{loss.item():.2f}"})

# plot of train vs loss
fig, ax = plt.subplots(
    figsize=(12, 3),
    dpi=100
)
# Raw loss
ax.plot(
    losses,
    alpha=0.5,
    label="Loss"
)
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
ax.set_title("Training Loss Curve")
window_size = 100
smoothed_losses = np.convolve(
    losses,
    np.ones(window_size) / window_size,
    mode="valid"
)
ax.plot(
    np.arange(
        window_size - 1,
        len(losses)
    ),
    smoothed_losses,
    label="Loss (moving avg)"
)
ax.legend(loc="upper right")
ax.set_xlim(0, len(losses))
ax.grid(True)
plt.savefig('train_vs_loss.png')
plt.close(fig)


#sampling from the trained model
# illustration on how to sample x1 and x0 using the learned velocity field
nb_steps = 15
path_x = np.zeros(nb_steps + 1) # array to store the full sampled path
t_steps = np.linspace(0, 1, nb_steps + 1)

# x0 starting piont
shape = 1, 1
x0 = torch.randn(shape).to(device=DEVICE)
print("x0: ", x0)

with torch.inference_mode():
    flow_matching_model.eval()
    xt = x0
    path_x[0] = xt.squeeze().cpu().numpy()

    for i in range(nb_steps):
        t = t_steps[i]
        dt = t_steps[i + 1] - t_steps[i]
        t_batch = torch.Tensor([[t]]).to(DEVICE)
        xt = xt + flow_matching_model(t=t_batch, xt=xt) * dt    #euler integration
        path_x[i + 1] = xt.squeeze().cpu().numpy()  # store the new position

display(HTML(pd.DataFrame({"t": t_steps, "x": path_x}).transpose().to_html()))

#multiple samples
@torch.inference_mode()
def sample(
    n_samples: int,  # Number of samples to generate
    model: FlowMatchingModel,  # The flow matching model
    nb_steps: int,  # Number of Euler integration steps
) -> torch.Tensor:
    """Generates samples by integrating the learned vector field using Euler integration."""
    ts = torch.linspace(0, 1, nb_steps + 1, device=DEVICE)  #[0, ....,1] nb_steps + 1 many items of equally spaced
    x_t = torch.randn(n_samples, data_dim).to(DEVICE)  # Sample x_0 ~ N(0, I)
    for i in range(nb_steps):  # Euler integration from t=0 to t=1 (last step happens just before t=1)
        t = ts[i]  # Current step $t$
        dt = ts[i + 1] - ts[i]  # Step size
        t_batch = t.expand(n_samples).unsqueeze(-1)     # [n_samples itesm] unsqueeze adds one dimenstion 
        # Move the sample a small step dt in the direction of the velocity field
        x_t = x_t + model(t=t_batch, x_t=x_t) * dt
    return x_t  # Final sample x_1


import numpy as np
import matplotlib.pyplot as plt
import torch

nb_steps = 50
n_samples = 5000

ts = torch.linspace(0, 1, nb_steps + 1, device=DEVICE)

x_t = torch.randn(n_samples, data_dim, device=DEVICE)

paths = torch.zeros(
    nb_steps + 1,
    n_samples,
    data_dim,
    device=DEVICE
)

paths[0] = x_t

with torch.inference_mode():
    flow_matching_model.eval()

    for i in range(nb_steps):
        t = ts[i]
        dt = ts[i + 1] - ts[i]

        t_batch = t.expand(n_samples).unsqueeze(-1)

        v_pred = flow_matching_model(
            t=t_batch,
            xt=x_t
        )

        x_t = x_t + v_pred * dt

        paths[i + 1] = x_t

paths = paths.squeeze(-1).cpu().numpy()
ts_np = ts.cpu().numpy()

x0_samples = paths[0]
x1_samples = paths[-1]

target_samples = mixture_sample(n_samples)


fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(
    x0_samples,
    bins=80,
    density=True,
    alpha=0.7,
    label="x₀ ~ N(0, I)"
)

ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.set_title("Initial Distribution: x₀")
ax.legend()
ax.grid(True)

plt.savefig(
    "x0_distribution.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)


fig, ax = plt.subplots(figsize=(10, 6))

selected_steps = [0, 5, 10, 20, 30, 40, 50]

for step in selected_steps:
    ax.hist(
        paths[step],
        bins=80,
        density=True,
        histtype="step",
        linewidth=2,
        label=f"t={ts_np[step]:.2f}"
    )

ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.set_title("Distribution Evolution During Euler Integration")
ax.legend()
ax.grid(True)

plt.savefig(
    "distribution_evolution.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)


fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(
    target_samples,
    bins=80,
    density=True,
    alpha=0.5,
    label="Target x₁ distribution"
)

ax.hist(
    x1_samples,
    bins=80,
    density=True,
    alpha=0.5,
    label="Generated x₁ distribution"
)

ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.set_title("Target Distribution vs Generated x₁ Distribution")
ax.legend()
ax.grid(True)

plt.savefig(
    "x1_distribution_vs_target.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)


fig, ax = plt.subplots(figsize=(10, 6))

for step in range(0, nb_steps + 1, 5):
    ax.hist(
        paths[step],
        bins=80,
        density=True,
        histtype="step",
        alpha=0.7,
        label=f"t={ts_np[step]:.2f}"
    )

ax.hist(
    target_samples,
    bins=80,
    density=True,
    alpha=0.25,
    label="Target"
)

ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.set_title("How N(0, I) Transforms into the Target Distribution")
ax.legend(
    loc="upper right",
    fontsize=8,
    ncol=2
)
ax.grid(True)

plt.savefig(
    "noise_to_target_distribution.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)


print("Saved:")
print("x0_distribution.png")
print("distribution_evolution.png")
print("x1_distribution_vs_target.png")
print("noise_to_target_distribution.png")