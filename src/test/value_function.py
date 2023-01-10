# This code creates a neural network that maps states to values.

import torch
from torch import Tensor

class ValueFunction(torch.nn.Module):
    """
    The value function network maps states to values.
    """
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        """
        Initializes the value function network.

        Parameters
        ----------
        action_dim : int
            The number of dimensions in the state.
        hidden_dim : int
            The number of neurons in the hidden layers.
        """
        # Call the base class's constructor
        super(ValueFunction, self).__init__()
        # Create the first layer
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # Create the second layer
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        # Create the third layer
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, state: Tensor) -> Tensor:
        """
        Maps states to values.

        Parameters
        ----------
        state : Tensor
            A batch of states.

        Returns
        -------
        Tensor
            The value of each state.
        """
        # Feed the batch of states through the first layer
        x = torch.relu(self.fc1(state))
        # Feed the result through the second layer
        x = torch.relu(self.fc2(x))
        # Feed the result through the third layer
        x = self.fc3(x)
        # Return the result
        return x