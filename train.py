import torch
from torch.autograd import Variable
import time
import os
import sys
import cv2

from utils import *
from torchvision.utils import save_image

def extract_inputimg(opt, idx, data):
    root_path = os.path.join(opt.result_path, '{}_input_img_'.format(idx))
    #print(data.shape) # (32 , 3, 64, 112, 112)   batchsize,  cheenl,  input frame, weith, heigh
    for i in range(data.shape[0]):
        if i % 5 == 0 :
          continue
        out_path = os.path.join(root_path, '{}'.format(i))
        if not os.path.exists(out_path): 
            os.makedirs(out_path)
        for j in range(data.shape[2]):
            ab = []
            for k in range(3):
              #ab.append(data[i][k][j].numpy())
              ab.append(data[i][k][j].unsqueeze(0))
            #print(ab[0].shape)
            #imgmg = cv2.merge((ab[0],ab[1],ab[2]))
            #cv2.imwrite(out_path+'/{}.png'.format(j), b)
            save_image(ab[0], out_path+'/{}_1.png'.format(j))
 


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger, scheduler):
    print('train at epoch {}'.format(epoch))

    model.train()
    model.to('cuda')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end_time = time.time()
    scheduler.step()
    for i, (inputs, targets) in enumerate(data_loader):
        if i == 0 and epoch == 1 :
            extract_inputimg(opt, i, inputs)
        data_time.update(time.time() - end_time)
        if not opt.no_cuda:
            targets = targets.cuda()
        inputs = Variable(inputs).cuda()  
        targets = Variable(targets).cuda()
        outputs = model(inputs) 
        loss = criterion(outputs, targets)

        losses.update(loss.data, inputs.size(0))
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,5))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val.item(),
            'prec1': top1.val.item(),
            'prec5': top5.val.item(),
            'lr': optimizer.param_groups[0]['lr']
        })
        if i % 10 ==0:
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                  'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                      epoch,
                      i,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      top5=top5,
                      lr=optimizer.param_groups[0]['lr']))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'prec1': top1.avg.item(),
        'prec5': top5.avg.item(),
        'lr': optimizer.param_groups[0]['lr']
    })

    #if epoch % opt.checkpoint == 0:
    #    save_file_path = os.path.join(opt.result_path,
    #                                  'save_{}.pth'.format(epoch))
    #    states = {
    #        'epoch': epoch + 1,
    #        'arch': opt.arch,
    #        'state_dict': model.state_dict(),
    #        'optimizer': optimizer.state_dict(),
    #    }
    #    torch.save(states, save_file_path)
