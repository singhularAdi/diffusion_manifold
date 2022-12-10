#from denoising_diffusion_pytorch.classifier_free_guidance import Unet, GaussianDiffusion
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

import torchvision
import torch
from tqdm import tqdm
import os
import os.path as osp

save_path = "./conditional_results"
if not osp.exists(save_path):
    os.mkdir(osp.join(save_path, "models"))
    os.mkdir(osp.join(save_path, "results"))
    
model = Unet(
    dim = 64,
    dim_mults = (1, 2),
    #num_classes=10,
    channels = 1,
    #cond_drop_prob = 0.1
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

trainer = Trainer(
    diffusion,
    'mnist',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 70000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,                        # turn on mixed precision
    augment_horizontal_flip=False,
    results_folder="./latest_results/"
)

trainer.train()

# dataset = torchvision.datasets.MNIST("./mnist/", train=True, transform=transforms, download=True)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=2, shuffle=True)
# optimizer = torch.optim.Adam(model.parameters(), lr=8e-5)
# epochs = 50
# device = "cuda"

# for epoch in tqdm(range(epochs)):
#     for idx, (images, labels) in enumerate(dataloader):
#         images = images.to(device)
#         labels = labels.to(device)
#         loss = diffusion(images, classes=labels)
#         print(f"Epoch {epoch}/{epochs} | Step {idx}/{len(dataloader)} | Loss {loss.item()}")
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     for i in range(10):
#         condition = torch.tensor([i]).to(device)
#         sampled_img = diffusion.sample(condition)
#         torchvision.utils.save_image(sampled_img, osp.join(save_path, "results", f"sample-{epoch}-class={i}.jpg"))
#     if epoch % 5 == 0:
#         torch.save(model, osp.join(save_path, "models", f"diffusion_epoch={epoch + 1}.pth"))
    
