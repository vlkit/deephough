import torch
from deephough import DeepHough
import matplotlib.pyplot as plt


fig, axes = plt.subplots(1, 2)

dh = DeepHough(num_angle=180, num_bias=100)
x = torch.zeros(100, 100).cuda()
x[40, :] = 1
x[60, :] = 1

x[:, [40, 50, 60]] = 1
y = dh(x.view(1, 1, 100, 100))

axes[0].imshow(x.cpu().squeeze())
axes[0].set_xlabel('H')
axes[0].set_ylabel('W')
axes[0].set(aspect='equal')
axes[1].imshow(y.squeeze().cpu())
axes[1].set_xlabel('bias')
axes[1].set_ylabel('angle')
axes[1].set_yticklabels(range(-110, 90, 20))
axes[1].set(aspect='equal')
plt.savefig('results.jpg')
