import torch
import torch.nn as nn
import torch.nn.functional as F

<<<<<<< HEAD

=======
>>>>>>> 544f254 (Add deployment files & PyTorch model via LFS)
class OilSpillCNN(nn.Module):
    def __init__(self):
        super(OilSpillCNN, self).__init__()

<<<<<<< HEAD

=======
>>>>>>> 544f254 (Add deployment files & PyTorch model via LFS)
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

<<<<<<< HEAD

        self.fc1 = nn.Linear(32 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, 1)


=======
        self.fc1 = nn.Linear(32 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, 1)

>>>>>>> 544f254 (Add deployment files & PyTorch model via LFS)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

<<<<<<< HEAD

=======
>>>>>>> 544f254 (Add deployment files & PyTorch model via LFS)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

<<<<<<< HEAD

=======
>>>>>>> 544f254 (Add deployment files & PyTorch model via LFS)
        return x
