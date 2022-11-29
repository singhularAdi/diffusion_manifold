import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import get_loaders
from models import LeNet, LeNetPL


def main():
    lenet = LeNet()
    model = LeNetPL(lenet)

    # load data
    trainloader, testloader = get_loaders()

    checkpoint_acc = ModelCheckpoint(
        save_top_k=1,
        monitor="val_acc",
        mode="max",
        filename="amnist-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}"
    )
    checkpoint_loss = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="lmnist-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}"
    )
    # train model

    trainer = pl.Trainer(accelerator='gpu', max_epochs=50,
                         callbacks=[checkpoint_acc, checkpoint_loss])
    trainer.fit(model=model, train_dataloaders=trainloader,
                val_dataloaders=testloader)


if __name__ == '__main__':
    main()
