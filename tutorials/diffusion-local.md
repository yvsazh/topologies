# Train and sample a tiny diffusion model locally

This CPU-friendly example uses random in-memory grayscale images; it requires no dataset download or network access. The data are demonstration noise, not a quality benchmark.

```python
import torch
from topologies.models.diffusion_unet import DiffusionUNet
from topologies.inference_tasks.diffusion import DiffusionSchedule, sample_ddim

torch.manual_seed(7)
images = torch.rand(8, 1, 16, 16) * 2 - 1
schedule = DiffusionSchedule.create(8, "linear")
model = DiffusionUNet(1, features=(8, 16), time_dim=16)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for _ in range(3):
    t = torch.randint(0, 8, (len(images),))
    noise = torch.randn_like(images)
    noisy = schedule.add_noise(images, noise, t)
    loss = (model(noisy, t) - noise).square().mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()

samples = sample_ddim(model.eval(), (4, 1, 16, 16), schedule, steps=4,
                      device="cpu", generator=torch.Generator().manual_seed(9))
torch.save(samples, "/tmp/topologies-diffusion-samples.pt")
print(samples.shape, samples.isfinite().all().item())
```

The tiny schedule and three updates demonstrate execution only. Production workflows should persist the schedule, prediction target, conditioning, and input normalization together with model weights.
