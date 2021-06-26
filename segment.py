import os
import time
import numpy as np
import torch
import sys

from PIL import Image, ImageOps
from argparse import ArgumentParser
from EntropyLoss import EmbeddingLoss
from iouEval import iouEval  # , getColorEntry

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
import torch.nn.functional as F

from datasets import *
# from dataset_loader import *
import transform as transforms

import importlib
from collections import OrderedDict, namedtuple

from shutil import copyfile


class load_data():

    def __init__(self, args):

        # First, a bit of setup
        dinf = namedtuple('dinf', ['name', 'n_labels', 'func'])
        self.metadata = [dinf('MNIST', 10, get_dataset),
                         dinf('EMNIST-LETTERS', 26, get_dataset),
                         #  dinf('CVD', 12, get_dataset),
                         ]

        self.num_labels = {
            entry.name: entry.n_labels for entry in self.metadata if entry.name in args.datasets}

        self.d_func = {entry.name: entry.func for entry in self.metadata}

    def __call__(self, name, train, supervised_ratio=0.2, is_grayscale=True):

        return self.d_func[name](name, train, supervised_ratio, is_grayscale)


def train(args, get_dataset, model, enc=False):
    FloatTensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if args.cuda else torch.LongTensor
    best_acc = 0

    # num_epochs = 10 if args.debug else args.num_epochs

    n_gpus = torch.cuda.device_count()
    print("\nWorking with {} GPUs".format(n_gpus))

    datasets = args.datasets
    # print(datasets)
    get_dataset = load_data(args)
    entropy = (args.alpha + args.beta) > 0
    print(entropy)
    if entropy:
        assert len(
            datasets) > 1, "Entropy Module undefined with single dataset. Exiting ... "

    NUM_LABELS = get_dataset.num_labels
    print(NUM_LABELS)

    dataset_train = {dname: get_dataset(
        dname, True, 0.3, False) for dname in datasets}
    dataset_val = {dname: get_dataset(dname, False) for dname in datasets}
    # dataset_unlabeled = {dname: get_dataset(dname, co_transform, 'train_extra' , mode='unlabeled') for dname in datasets}
    # dataset_unlabeled = {dname: get_dataset(
    #     dname, 'train', mode='unlabeled') for dname in datasets}

    if entropy:

        n_unlabeled = np.max([len(dataset_train[dname][1])
                             for dname in datasets])
        # print(n_unlabeled)

    print("Working with {} Dataset(s):".format(len(datasets)))
    for key in datasets:
        print("{}: Unlabeled images {}, Training on {} images, Validation on {} images".format(
            key, len(dataset_train[key][1]), len(dataset_train[key][0]), len(dataset_val[key])))

    # for d in datasets:
    #     if len(set(dataset_train.values())) != 1:
    #         max_train_size = np.max([len(dataset_train[dname])
    #                                 for dname in datasets])
    #         dataset_train[d].image_paths = dataset_train[d].image_paths * \
    #             int(np.ceil(float(max_train_size) /
    #                 len(dataset_train[d].image_paths)))
    #         dataset_train[d].label_paths = dataset_train[d].label_paths * \
    #             int(np.ceil(float(max_train_size) /
    #                 len(dataset_train[d].label_paths)))

    loader_train = {dname: DataLoader(dataset_train[dname][0], batch_size=args.batch_size,
                                      shuffle=True) for dname in datasets}
    loader_val = {dname: DataLoader(dataset_val[dname], batch_size=100,
                                    shuffle=True, drop_last=True) for dname in datasets}

    if entropy:
        loader_unlabeled = {dname: DataLoader(dataset_train[dname][1], batch_size=args.batch_size,
                                              shuffle=True, drop_last=True) for dname in datasets}

    if args.model == 'drnet':
        optimizer = SGD(model.optim_parameters(), args.lr, 0.9,
                        weight_decay=1e-4)  # scheduler DR-Net
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    doIou = {'train': args.iouTrain, 'val': args.iouVal}
    average_epoch_loss = {'train': np.inf, 'val': np.inf}

    label_embedding = {key: torch.randn(NUM_LABELS[key], args.em_dim).cuda(
    ) for key in datasets}  # Random Initialization

    # If provided, use label embedddings
    if args.pt_em:
        fn = torch.load(args.pt_em)
        label_embedding = {key: torch.tensor(
            fn[key], dtype=torch.float).cuda() for key in datasets}

    start_epoch = 1

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: pow(
        (1-((epoch-1)/args.num_epochs)), 0.9))  # scheduler 2
    loss_criterion = {key: torch.nn.CrossEntropyLoss(
        ignore_index=NUM_LABELS[key]-1).cuda() for key in datasets}

    if len(datasets) > 1:
        similarity_module = EmbeddingLoss(
            NUM_LABELS, args.em_dim, label_embedding, loss_criterion)
        similarity_module = torch.nn.DataParallel(similarity_module).cuda()
        torch.save(label_embedding, os.path.join(
            "./weight", "label_embedding.pkl"))

    print()
    print("========== STARTING TRAINING ===========")
    print()

    n_iters = min([len(loader_train[d]) for d in datasets])

    if entropy:
        unlabeled_iters = {
            d: len(loader_unlabeled[d])//n_iters for d in datasets}

    for epoch in range(start_epoch, num_epochs+1):

        epoch_start_time = time.time()
        usedLr = 0
        iou = {key: (0, 0) for key in datasets}

        ###### TRAIN begins  #################
        for phase in ['train']:

            eval_iou = doIou[phase]
            print("-----", phase, "- EPOCH", epoch, "-----")

            scheduler.step(epoch)
            model.train()

            for param_group in optimizer.param_groups:
                print("LEARNING RATE: ", param_group['lr'])
                usedLr = float(param_group['lr'])

            # Initialize the iterables

            labeled_iterator = {dname: iter(
                loader_train[dname]) for dname in datasets}

            if entropy:
                unlabeled_iterator = {dname: iter(
                    loader_unlabeled[dname]) for dname in datasets}

            if args.alpha:
                alpha = 1
            if args.beta:
                beta = 1

            epoch_loss = {d: [] for d in datasets}
            epoch_sup_loss = {d: [] for d in datasets}
            epoch_ent_loss = {d: [] for d in datasets}

            time_taken = []

            if (eval_iou):
                iou_data = {key: iouEval(NUM_LABELS[key]) for key in datasets}

            for itr in range(n_iters):

                optimizer.zero_grad()
                loss_sup = {d: 0 for d in datasets}
                loss_ent = {d: [0] for d in datasets}

                for d in datasets:

                    images_l, targets_l = next(labeled_iterator[d])

                    images_l = images_l.cuda()
                    targets_l = targets_l.cuda()

                    start_time = time.time()

                    images_l = Variable(images_l.type(FloatTensor))
                    dec_outputs = model(images_l, enc=False,
                                        finetune=args.finetune)
                    # print(targets_l.size())
                    # print(dec_outputs[d].size())
                    # print(dec_outputs[d])
                    # print(targets_l)
                    loss_s = loss_criterion[d](
                        dec_outputs[d], targets_l)
                    loss_s.backward()
                    loss_sup[d] = loss_s.item()

                    if entropy:

                        for _ in range(unlabeled_iters[d]):
                            images_u = next(unlabeled_iterator[d])
                            images_u = images_u.cuda()

                            images_u = Variable(images_u.type(FloatTensor))
                            _, en_outputs = model(images_u)

                            loss_e = torch.mean(similarity_module(
                                en_outputs, d, args.alpha, args.beta))  # unsupervised losses
                            loss_e /= unlabeled_iters[d]
                            loss_e.backward()
                            loss_ent[d].append(loss_e.item())

                    epoch_sup_loss[d].append(loss_sup[d])
                    epoch_ent_loss[d].extend(loss_ent[d])
                    # Already averaged over iters
                    epoch_loss[d].append(loss_sup[d] + np.sum(loss_ent[d]))

                time_taken.append(time.time() - start_time)
                optimizer.step()

                if args.steps_loss > 0 and (itr % args.steps_loss == 0 or itr == n_iters-1):
                    average = {d: np.around(
                        sum(epoch_loss[d]) / len(epoch_loss[d]), 3) for d in datasets}
                    print(f'{phase} loss: {average} (epoch: {epoch}, step: {itr})',
                          "// Avg time/img: %.4f s" % (sum(time_taken) / len(time_taken) / args.batch_size))

                average = {d: np.mean(epoch_loss[d]) for d in datasets}
                average_epoch_loss[phase] = sum(average.values())

                if entropy:
                    average_epoch_sup_loss = {d: np.mean(
                        epoch_sup_loss[d]) for d in datasets}
                    average_epoch_ent_loss = {d: np.mean(
                        epoch_ent_loss[d]) for d in datasets}

                    # Write the epoch wise supervised and total unsupervised losses.
                    # with open(loss_logpath, "a") as myfile:
                    #     if len(datasets) > 1 and (itr % args.steps_loss == 0 or itr == n_iters-1):
                    #         myfile.write("%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n" % (epoch, average_epoch_sup_loss.get(datasets[0], 0), average_epoch_sup_loss.get(
                    #             datasets[1], 0), average_epoch_ent_loss.get(datasets[0], 0), average_epoch_ent_loss.get(datasets[1], 0), average_epoch_loss[phase]))

            # Todo: A better way to close the worker threads.
            for d in datasets:
                while True:
                    try:
                        _ = next(labeled_iterator[d])
                    except StopIteration:
                        break

                if entropy:
                    while True:
                        try:
                            _ = next(unlabeled_iterator[d])
                        except StopIteration:
                            break

            iou = {key: (0, 0) for key in datasets}

            if (eval_iou):
                iou = {key: iou_data[key].getIoU() for key in datasets}

                iouStr_label = {key: '{:0.2f}'.format(
                    iou[key][0]*100) for key in datasets}
                for d in datasets:
                    print("EPOCH IoU on {} dataset: {} %".format(
                        d, iouStr_label[d]))

        ########## Train ends ###############################
        print('bbb')
        ##### Validation ###############
        if (epoch == 1) or (epoch % 5 == 0):  # validation after every 5 epoch
            for phase in ['val']:

                eval_iou = doIou[phase]
                print("-----", phase, "- EPOCH", epoch, "-----")

                model.eval()

                if (eval_iou):
                    iou_data = {d: iouEval(NUM_LABELS[d]) for d in datasets}

                epoch_val_loss = {d: [] for d in datasets}
                if args.pAcc:
                    pAcc = {d: [] for d in datasets}

                for d in datasets:
                    time_taken = []

                    for itr, (images, targets) in enumerate(loader_val[d]):

                        start_time = time.time()

                        images = images.cuda()
                        targets = targets.cuda()

                        with torch.set_grad_enabled(False):

                            seg_output = model(images, enc=False)
                            loss = loss_criterion[d](
                                seg_output[d], targets.squeeze(1))

                            if eval_iou:
                                pred = seg_output[d].argmax(1, True).data
                                iou_data[d].addBatch(pred, targets.data)
                                if args.pAcc:
                                    a = (pred == targets.data)
                                    pAcc[d].append(torch.mean(a.double()))

                            epoch_val_loss[d].append(loss.item())

                        time_taken.append(time.time() - start_time)

                        if args.steps_loss > 0 and (itr % args.steps_loss == 0 or itr == len(loader_val[d])-1):
                            average = np.around(np.mean(epoch_val_loss[d]), 3)
                            print(f'{d}: {phase} loss: {average} (epoch: {epoch}, step: {itr})',
                                  "// Avg time/img: %.4f s" % (sum(time_taken) / len(time_taken) / args.batch_size))

                average_epoch_loss[phase] = np.sum(
                    [np.mean(epoch_val_loss[d]) for d in datasets])

                if (eval_iou):
                    iou = {d: iou_data[d].getIoU() for d in datasets}

                    iouStr_label = {d: '{:0.2f}'.format(
                        iou[d][0]*100) for d in datasets}
                    for d in datasets:
                        print("EPOCH IoU on {} dataset: {} %".format(
                            d, iouStr_label[d]))
                        if args.pAcc:
                            print(f'{d}: pAcc : {np.mean(pAcc[d])*100}%')
        ############# VALIDATION ends #######################
        print('ccc')
        print("Epoch time {} s".format(time.time() - epoch_start_time))

        # remember best valIoU and save checkpoint
        if sum([iou[key][0] for key in datasets]) == 0:
            current_acc = -average_epoch_loss['val']
        else:
            # Average of the IoUs to save best model
            current_acc = sum([iou[key][0] for key in datasets])/len(datasets)

        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)

        filenameCheckpoint = savedir + '/checkpoint.pth.tar'
        filenameBest = savedir + '/model_best.pth.tar'

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)

        # SAVE MODEL AFTER EPOCH

        filename = f'{savedir}/model-{epoch:03}.pth'
        filenamebest = f'{savedir}/model_best.pth'

        if args.epochs_save > 0 and epoch > 0 and epoch % args.epochs_save == 0:
            torch.save(model.state_dict(), filename)
            print(f'save: {filename} (epoch: {epoch})')

        if (is_best):
            torch.save(model.state_dict(), filenamebest)
            print(f'save: {filenamebest} (epoch: {epoch})')

            with open(savedir + "/best.txt", "w") as myfile:
                myfile.write("Best epoch is %d\n" % (epoch))
                for d in datasets:
                    myfile.write("Val-IoU-%s= %.4f\n" % (d, iou[d][0]))

                myfile.write("\n\n")

                for d in datasets:
                    myfile.write(
                        "Classwise IoU for best epoch in %s is ... \n" % (d))
                    for values in iou[d][1]:
                        myfile.write("%.4f " % (values))
                    myfile.write("\n\n")

        # with open(automated_log_path, "a") as myfile:
        #     iouTrain = 0
        #     if len(datasets) > 1:
        #         myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (
        #             epoch, average_epoch_loss['train'], average_epoch_loss['val'], iouTrain, iouTrain, iou[datasets[0]][0], iou[datasets[1]][0], usedLr))
        #     else:
        #         myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (
        #             epoch, average_epoch_loss['train'], average_epoch_loss['val'], iouTrain, iou[datasets[0]][0], usedLr))
        print('aaaa')
    return(model)


def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print("Saving model as best")
        torch.save(state, filenameBest)


def main(args, get_dataset):
    # Load Model
    assert os.path.exists(
        args.model + ".py"), f"Error: model definition for {args.model} not found"

    model_file = importlib.import_module(args.model)
    if args.bnsync:
        model_file.BatchNorm = batchnormsync.BatchNormSync
    else:
        model_file.BatchNorm = torch.nn.BatchNorm2d

    NUM_LABELS = get_dataset.num_labels

    model = model_file.Net(NUM_LABELS, args.em_dim, args.resnet)
    # copyfile(args.model + ".py", savedir + '/' + args.model + ".py")

    # if args.cuda:
    # 	model = torch.nn.DataParallel(model).cuda()

    train_start = time.time()

    model = train(args, get_dataset, model, False)  # Train
    print("========== TRAINING FINISHED ===========")
    print(f"Took {(time.time()-train_start)/60} minutes")


def parse_args():

    parser = ArgumentParser()
    parser.add_argument('--model', default='drnet')
    # parser.add_argument('--debug', action='store_true')
    # parser.add_argument('--basedir', required=True)
    parser.add_argument('--bnsync', action='store_true')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--random-rotate', type=int, default=0)
    parser.add_argument('--random-scale', type=int, default=0)
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=128)
    # parser.add_argument('--savedir', required=True)
    parser.add_argument('--datasets', nargs='+',
                        default=['MNIST', 'EMNIST-LETTERS'])
    parser.add_argument('--em-dim', type=int, default=100)
    parser.add_argument('--K', type=float, default=1e4)
    parser.add_argument('--theta', type=float, default=0)
    # Number of samples from each dataset. If empty, consider full dataset.
    parser.add_argument('--num-samples', type=int)
    parser.add_argument('--update-embeddings', type=int, default=0)
    parser.add_argument('--pt-em')
    # Cross dataset loss term coeff.
    parser.add_argument('--alpha', type=int, default=0.1)
    # Within dataset loss term coeff.
    parser.add_argument('--beta', type=int, default=0.1)
    parser.add_argument('--resnet', default='resnet_18')
    parser.add_argument('--pAcc', action='store_true')

    ### Optional ######
    parser.add_argument('--finetune', action='store_true')
    # NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--state')
    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--height', type=int, default=512)
    # parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--steps-loss', type=int, default=50)
    # You can use this value to save model every X epochs
    parser.add_argument('--epochs-save', type=int, default=0)
    # recommended: False (takes more time to train otherwise)
    parser.add_argument('--iouTrain', action='store_true', default=False)
    parser.add_argument('--iouVal', action='store_true', default=True)
    # Use this flag to load last checkpoint for training
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    try:
        args = parse_args()

        num_epochs = args.num_epochs
        # num_epochs = 10 if args.debug else args.num_epochs

        n_gpus = torch.cuda.device_count()
        print("\nWorking with {} GPUs".format(n_gpus))

        datasets = args.datasets
        # print(datasets)
        get_dataset = load_data(args)
        entropy = (args.alpha + args.beta) > 0
        print(entropy)
        if entropy:
            assert len(
                datasets) > 1, "Entropy Module undefined with single dataset. Exiting ... "

        NUM_LABELS = get_dataset.num_labels

        dataset_train = {dname: get_dataset(
            dname, True, 0.3, False) for dname in datasets}
        print(dataset_train['MNIST'][0])

        main(args, get_dataset)
    except KeyboardInterrupt:
        sys.exit(0)
