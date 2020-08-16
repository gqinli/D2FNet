import datetime
import os
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.backends import cudnn

import joint_transforms
from config import dutsk_path
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir
# from model_A_levelcon import R3Net
# from model_I_backbone import R3Net
from model_II_rgbfuse_baseline import R3Net

cudnn.benchmark = True

torch.manual_seed(2018)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# torch.cuda.set_device(1)

ckpt_path = './ckpt'
# exp_name = 'rslt_A_levelcon'
# exp_name = 'rslt_I_backbone'
exp_name = 'rslt_II_rgbfuse_baseline'


args = {
    'status': 'train',
    'iter_num': 30000,
    'train_batch_size': 1,
    'last_iter': 0,
    'lr': 1e-3,
    'lr_decay': 0.1,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': ''
}

joint_transform = joint_transforms.Compose([
    joint_transforms.RandomCrop(448),
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(10)
])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
depth_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

train_set = ImageFolder(dutsk_path, args['status'],joint_transform, img_transform, target_transform,depth_transform)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=12, shuffle=True)

criterion = nn.BCEWithLogitsLoss().cuda()
edgeLoss = nn.MSELoss().cuda()
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')


def main():
    """
    RGB与Depth每一层直接融合
    """
    net = R3Net(num_class=1).cuda().train()
    # net = nn.DataParallel(net,device_ids=[0,1])

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_baseFPN.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)


def train(net, optimizer):
    curr_iter = args['last_iter']
    while True:
        total_loss_record, loss0_record, loss1_record, loss2_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']

            inputs, depth,labels,img_name = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            depth = Variable(depth).cuda()
            labels = Variable(labels).cuda()

            optimizer.zero_grad()
            salMap, sideout5, sideout4, sideout3, sideout2 = net(inputs, depth)

            loss1 = criterion(salMap, labels)
            loss5 = criterion(sideout5, labels)
            loss4 = criterion(sideout4, labels)
            loss3 = criterion(sideout3, labels)
            loss2 = criterion(sideout2, labels)

            total_loss = loss1 + loss5 + loss4 + loss3 + loss2
            total_loss.backward()
            optimizer.step()

            total_loss_record.update(total_loss.item(), batch_size)
            loss1_record.update(loss1.item(), batch_size)
            # loss1_record.update(loss1.item(), batch_size)

            curr_iter += 1

            log = '[iter %d], [total loss %.5f], [loss1 %.5f],[lr %.13f]' % \
                  (curr_iter, total_loss_record.avg, loss1_record.avg,
                   optimizer.param_groups[1]['lr'])
            print(log)
            open(log_path, 'a').write(log + '\n')

            if curr_iter == args['iter_num']:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d_res18_noCross_Gdept.pth' % curr_iter))
                return


if __name__ == '__main__':
    main()
