import argparse
import logging
import os
from torch import optim
from tqdm import tqdm

from eval import eval_net
from model.unet_model import *
from utils.datasets import BasicDataset
from torch.utils.data import DataLoader, random_split

dir_img = 'data/train/images/'  # 原图像路径
dir_mask = 'data/train/masks/'   # mask路径
dir_checkpoint = 'checkpoints/'


def train_net(net,
              device,
              epochs=80,
              batch_size=150,
              lr=0.001,
              save_cp=True,
              img_scale=0.5):

    train_dataset = BasicDataset(dir_img, dir_mask, img_scale,mask_suffix="_m")
    n_train = len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_img = "data/val/images/"
    val_mask = "data/val/masks/"
    val_dataset = BasicDataset(val_img, val_mask, img_scale,mask_suffix="_m")

    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    global_step = 0
    criterion = nn.BCEWithLogitsLoss()
    logging.info('''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
        loss_function:   {criterion}
    ''')

    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)
    # assert not (net.n_classes > 1 and criterion == nn.BCEWithLogitsLoss())

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc='Epoch '+str(epoch + 1)+"/"+str(epochs), unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                # assert imgs.shape[1] == 3, \
                #     'Network has been defined with {net.n_channels} input channels, ' \
                #     'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                #     'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                # print(masks_pred.shape,true_masks.shape)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                # writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    val_score, Map, jaccard, recall, precision = eval_net(net, loader,device)
                    scheduler.step(val_score)
                    # writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)


                    logging.info('\n')
                    logging.info('Validation Pixel accuracy: {}'.format(Map))
                    logging.info('Validation Dice Coeff: {}'.format(val_score))
                    logging.info('Validation jaccard: {}'.format(jaccard))
                    logging.info('Validation recall: {}'.format(recall))
                    logging.info('Validation precision: {}'.format(precision))


                        # writer.add_scalar('Dice/test', val_score, global_step)

                    # writer.add_images('images', imgs, global_step)
                    # if net.n_classes == 1:
                        # writer.add_images('masks/true', true_masks, global_step)
                        # writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + '128CP_epoch.pth')
            logging.info('Checkpoint saved !')

    # writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=80 ,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=150,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.01,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=30.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = None
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(UNet(n_channels=3, n_classes=1, bilinear=True), device_ids=[0, 1])
    net.to(device=device)
    logging.info('Network:\n'
                 '\t{net.n_channels} input channels\n'
                 '\t{net.n_classes} output channels (classes)\n'
                 '\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info('Model loaded from {args.load}')

    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=0.5)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
