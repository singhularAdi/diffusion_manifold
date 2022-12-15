import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import get_loaders2
from models import LeNet, LeNetPL


def main():
    lenet = LeNet()
    model = LeNetPL(lenet)

    # load data
    trainloader, testloader = get_loaders2()

    checkpoint_loss = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="mnist64-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}"
    )
    # train model

    trainer = pl.Trainer(accelerator='gpu', max_epochs=50,
                         callbacks=[checkpoint_loss])
    trainer.fit(model=model, train_dataloaders=trainloader,
                val_dataloaders=testloader)


if __name__ == '__main__':
    main()
