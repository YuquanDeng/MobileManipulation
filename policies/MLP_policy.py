from torch import nn, Tensor
import torch.nn.functional as F

class MLP_policy(nn.Module):
    def __init__(self,
                 input_size:int, 
                 size: int,
                 output_size: int) -> None:
        super(MLP_policy, self).__init__()

        self.fc1 = nn.Linear(input_size, size)
        self.fc2 = nn.Linear(size, size)
        self.fc3 = nn.Linear(size, size)
        self.fc4 = nn.Linear(size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


        
