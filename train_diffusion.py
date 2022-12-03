from denoising_diffusion_pytorch.classifier_free_guidance import Unet, GaussianDiffusion
import torchvision
import torch
from tqdm import tqdm
import os
import os.path as osp

from ema_pytorch import EMA

from accelerate import Accelerator
from pathlib import Path

def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

save_path = "./conditional_results"
os.makedirs(osp.join(save_path, "results"), exist_ok=True)
os.makedirs(osp.join(save_path, "models"), exist_ok=True)
results_folder = osp.join(save_path, "results")
model = Unet(
    dim = 64,
    dim_mults = (1, 2),
    num_classes=10,
    channels = 1,
    cond_drop_prob = 0.1
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
batch_size = 32
num_samples = 25
dataset = torchvision.datasets.MNIST("./mnist/", train=True, transform=transforms, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True)
optimizer = torch.optim.Adam(diffusion.parameters(), lr=8e-5)
epochs = 50
device = "cuda"
train_num_steps = 70000
step = 0
gradient_accumulate_every = 2
save_and_sample_every = 1000
"""for epoch in tqdm(range(epochs)):
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
        torchvision.utils.save_image(sampled_img, osp.join(save_path, "results", f"sample-{epoch}-class={i}.jpg"))
    if epoch % 5 == 0:
        torch.save(model, osp.join(save_path, "models", f"diffusion_epoch={epoch + 1}.pth"))"""
    
accelerator = Accelerator(split_batches=True)
device = accelerator.device
dataloader = accelerator.prepare(dataloader)
dataloader = cycle(dataloader)
if accelerator.is_main_process:
    ema = EMA(model, beta = 0.995, update_every = 10)

    results_folder = Path(results_folder)
    results_folder.mkdir(exist_ok = True)
model, optimizer = accelerator.prepare(model, optimizer)

with tqdm(initial = 0, total = train_num_steps, disable = not accelerator.is_main_process) as pbar:

    while step < train_num_steps:

        total_loss = 0.

        for _ in range(gradient_accumulate_every):
            data, labels = next(dataloader)
            data = data.to(device)
            labels = labels.to(device)

            with accelerator.autocast():
                loss = diffusion(data, classes=labels)
                loss = loss / gradient_accumulate_every
                total_loss += loss.item()

            accelerator.backward(loss)

        accelerator.clip_grad_norm_(diffusion.parameters(), 1.0)
        pbar.set_description(f'loss: {total_loss:.4f}')

        accelerator.wait_for_everyone()

        optimizer.step()
        optimizer.zero_grad()

        accelerator.wait_for_everyone()

        step += 1
        if accelerator.is_main_process:
            ema.to(device)
            ema.update()

            if step != 0 and step % save_and_sample_every == 0:
                ema.ema_model.eval()

                for i in range(10):
                    condition = torch.tensor([i]).to(device)
                    sampled_img = diffusion.sample(condition, cond_scale = 3.)
                    torchvision.utils.save_image(sampled_img, osp.join(save_path, "results", f"sample-{step + 1}-class={i}.jpg"))
                torch.save(model, osp.join(save_path, "models", f"diffusion_epoch={step + 1}.pth"))
                
                #torchvision.utils.save_image(all_images, osp.join(save_path, "results", f"sample-{step}-class={i}.jpg"))
                #utils.save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                #self.save(milestone)

        pbar.update(1)