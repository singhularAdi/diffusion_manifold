from denoising_diffusion_pytorch.classifier_free_guidance import Unet, GaussianDiffusion
import torchvision
import torch
from tqdm import tqdm

model = Unet(
    dim = 64,
    dim_mults = (1, 2),
    num_classes=10,
    channels = 1,
    cond_drop_prob = 0.5
)

diffusion = GaussianDiffusion(
    model,
    image_size = 64,
    timesteps = 1000,   # number of steps
).cuda()
transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(64),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.1307, 0.3081),
    ]
)
dataset = torchvision.datasets.MNIST("./mnist/", train=True, transform=transforms, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=2, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=8e-5)
epochs = 10
device = "cuda"

for epoch in tqdm(range(epochs)):
    for idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        loss = diffusion(images, classes=labels)
        print(f"Epoch {epoch}/{epochs} | Step {idx}/{len(dataloader)} | Loss {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for i in range(10):
        condition = torch.tensor([i]).to(device)
        sampled_img = diffusion.sample(condition)
        torchvision.utils.save_image(sampled_img, f"./conditional_results/sample-{epoch}-class={i}.jpg")
    if epoch % 5 == 0:
        torch.save(model, f"diffusion_epoch={epoch + 1}")
    
