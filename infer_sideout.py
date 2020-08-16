# encoding=utf-8
import numpy as np
import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import glob

from config import nju_path,nlpr_path,des_path,lfsd_path,ssd_path,sip_path,stere_path
from misc import check_mkdir, AvgMeter, cal_precision_recall_mae, cal_fmeasure1
from model_A_levelcon import R3Net
# from model_B_lastlayer_trainweight import R3Net
# from model_C_eachlayer_trainweight import R3Net
# from model_I_backbone import R3Net
# from model_II_rgbfuse_baseline import R3Net
# from model_III_rgbfuse_backbone_toplayerdepth import R3Net
# from model_resnet18_d2fnet import R3Net
from collections import OrderedDict
torch.manual_seed(2018)

# set which gpu to use
# torch.cuda.set_device(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# the following two args specify the location of the file of trained model (pth extension)
# you should have the pth file in the folder './$ckpt_path$/$exp_name$'
ckpt_path = './ckpt'
exp_name = 'rslt_A_levelcon'
# exp_name = 'rslt_B_lastlayer_trainweight'
# exp_name = 'rslt_C_eachlayer_trainweight'
# exp_name = 'rslt_I_backbone'
# exp_name = 'rslt_II_rgbfuse_baseline'
# exp_name = 'rslt_III_rgbfuse_backbone_toplayerdepth'
# exp_name = 'rslt_resnet18_d2fnet'

args = {
    'snapshot': '30000_res18_noCross_Gdept',  # your snapshot filename (exclude extension name)
    'save_results': True  # whether to save the resulting masks
}

img_transform = transforms.Compose([
    transforms.Resize((448, 448), interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
depth_transform = transforms.Compose([
    transforms.Resize((448, 448), interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
to_pil = transforms.ToPILImage()

# to_test = {'NJU2000': nju_path}
# to_test = {'nlpr': nlpr_path}
# to_test = {'des': des_path}
# to_test = {'lfsd': lfsd_path}
# to_test = {'ssd': ssd_path}
# to_test = {'sip': sip_path}
to_test = {'stere': stere_path}

def main():
    net = R3Net(num_class=1).cuda().train()

    print('load snapshot \'%s\' for testing' % args['snapshot'])
    #net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'), map_location={'cuda:1':'cuda:0'}))
    # when using the linux compute then run the code
    state_dict = torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'))
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #      namekey = k[7:]  # remove 'module.'
    #      new_state_dict[namekey] = v
    #  # load params
    # net.load_state_dict(new_state_dict)
    net.load_state_dict(state_dict)
    net.eval()

    results = {}

    with torch.no_grad():

        for name, root in to_test.items():
            pre, rec = [], []

            precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
            mae_record = AvgMeter()

            if args['save_results']:
                check_mkdir(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot'])))

            hha = sorted(glob.glob(root + '/hha/*.jpg'))
            # # load nju
            # rgb = ['./data/NJU_test/LR/'+i.split('/')[4] for i in sorted(glob.glob(root + '/hha/*.jpg'))]
            # label = ['./data/NJU_test/GT/'+i.split('/')[4].split('.')[0]+'.png' for i in sorted(glob.glob(root + '/hha/*.jpg'))]
            # # load nlpr
            # rgb = ['./data/NLPR_test/RGB/'+i.split('/')[4].split('_Depth')[0]+'.jpg' for i in sorted(glob.glob(root + '/hha/*.jpg'))]
            # label = ['./data/NLPR_test/groundtruth/'+i.split('/')[4].split('_Depth')[0]+'.jpg' for i in sorted(glob.glob(root + '/hha/*.jpg'))]
            # # load des
            # rgb = ['./data/DES/image/RGBD_data_'+i.split('_')[2]+'.png' for i in sorted(glob.glob(root + '/hha/*.jpg'))]
            # label = ['./data/DES/GT/RGBD_data_'+i.split('_')[2]+'.bmp' for i in sorted(glob.glob(root + '/hha/*.jpg'))]
            # load lfsd
            # rgb = ['./data/LFSD/all_focus_images/'+i.split('/')[4] for i in sorted(glob.glob(root + '/hha/*.jpg'))]
            # label =['./data/LFSD/ground_truth/'+i.split('/')[4].split('.')[0]+'.png' for i in sorted(glob.glob(root + '/hha/*.jpg'))]
            # load ssd
            # rgb = ['./data/SSD100/RGB/' + i.split('/')[4] for i in sorted(glob.glob(root + '/hha/*.jpg'))]
            # label = ['./data/SSD100/GT/' + i.split('/')[4].split('.')[0] + '.png' for i in sorted(glob.glob(root + '/hha/*.jpg'))]
            # load sip
            # rgb = ['./data/SIP/RGB/' + i.split('/')[4] for i in sorted(glob.glob(root + '/hha/*.jpg'))]
            # label = ['./data/SIP/GT/' + i.split('/')[4].split('.')[0] + '.png' for i in sorted(glob.glob(root + '/hha/*.jpg'))]
            # load stere
            rgb = ['./data/STERE/RGB/' + i.split('/')[4] for i in sorted(glob.glob(root + '/hha/*.jpg'))]
            label = ['./data/STERE/GT/' + i.split('/')[4].split('.')[0] + '.png' for i in sorted(glob.glob(root + '/hha/*.jpg'))]

            for idx in range(len(rgb)):
                print('predicting for %s: %d / %d' % (name, idx + 1, len(rgb)))

                img = Image.open(rgb[idx]).convert('RGB')
                depth = Image.open(hha[idx]).convert('RGB')
                h, w = img.size[0], img.size[1]

                img = img_transform(img)
                depth = img_transform(depth)

                img_var = Variable(img.unsqueeze(0), volatile=True).cuda()
                depth_var = Variable(depth.unsqueeze(0), volatile=True).cuda()

                # prediction = net(img_var,depth_var)#,out3,out2,out1,out0
                # for i in range(len(tmp)):
                #    tmp_ = np.array(to_pil(tmp[i].data.squeeze(0).cpu()))
                #    Image.fromarray(tmp_).save('depth_visual/'+str(i)+'.jpg')
                sideout5,sideout4,sideout3,sideout2 = net(img_var, depth_var)
                # depthMap = F.interpolate(input=depthMap, size=(w, h), mode='bilinear', align_corners=False)
                # depthMap = np.array(to_pil(depthMap.data.squeeze(0).cpu()))
                sideout5 = F.interpolate(input=sideout5, size=(w, h), mode='bilinear', align_corners=False)
                sideout5 = np.array(to_pil(sideout5.data.squeeze(0).cpu()))
                sideout4 = F.interpolate(input=sideout4, size=(w, h), mode='bilinear', align_corners=False)
                sideout4 = np.array(to_pil(sideout4.data.squeeze(0).cpu()))
                sideout3 = F.interpolate(input=sideout3, size=(w, h), mode='bilinear', align_corners=False)
                sideout3 = np.array(to_pil(sideout3.data.squeeze(0).cpu()))
                sideout2 = F.interpolate(input=sideout2, size=(w, h), mode='bilinear', align_corners=False)
                sideout2 = np.array(to_pil(sideout2.data.squeeze(0).cpu()))
                # prediction = net(img_var, depth_var)
                # prediction = F.interpolate(input=prediction, size=(w, h), mode='bilinear', align_corners=False)
                # prediction = np.array(to_pil(prediction.data.squeeze(0).cpu()))

                # load GT
                # gt = np.array(Image.open(label[idx]).convert('L'))
                #
                # try:
                #     precision, recall, mae = cal_precision_recall_mae(prediction, gt)
                #     pre.append(precision)
                #     rec.append(recall)
                # except Exception as error:
                #     import pdb
                #     pdb.set_trace()
                # for pidx, pdata in enumerate(zip(precision, recall)):
                #     p, r = pdata
                #     precision_record[pidx].update(p)
                #     recall_record[pidx].update(r)
                # mae_record.update(mae)

                # if args['save_results']:
                #     Image.fromarray(prediction).save(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (
                #         exp_name, name, args['snapshot']), rgb[idx].split('/')[4]))
                # tmp1 = 'depthMap'
                # if not os.path.exists(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, tmp1))):
                #     os.makedirs(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, tmp1)))
                # Image.fromarray(depthMap).save(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (
                #     exp_name, name, tmp1), rgb[idx].split('/')[4]))
                tmp2 = 'sideout7543'
                if not os.path.exists(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, tmp2))):
                    os.makedirs(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, tmp2)))
                Image.fromarray(sideout2).save(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (
                    exp_name, name, tmp2), rgb[idx].split('/')[4]))
                tmp3 = 'sideout754'
                if not os.path.exists(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, tmp3))):
                    os.makedirs(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, tmp3)))
                Image.fromarray(sideout3).save(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (
                    exp_name, name, tmp3), rgb[idx].split('/')[4]))
                tmp4 = 'sideout75'
                if not os.path.exists(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, tmp4))):
                    os.makedirs(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, tmp4)))
                Image.fromarray(sideout4).save(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (
                    exp_name, name, tmp4), rgb[idx].split('/')[4]))
                tmp5 = 'sideout7'
                if not os.path.exists(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, tmp5))):
                    os.makedirs(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, tmp5)))
                Image.fromarray(sideout5).save(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (
                    exp_name, name, tmp5), rgb[idx].split('/')[4]))

            # max_fmeasure,mean_fmeasure = cal_fmeasure([precord.avg for precord in precision_record], [rrecord.avg for rrecord in recall_record])
    #         max_fmeasure, mean_fmeasure = cal_fmeasure1(pre, rec)
    #         results[name] = {'max_fmeasure': max_fmeasure,'mean_fmeasure': mean_fmeasure, 'mae': mae_record.avg}
    # print('test results:')
    # print(results)


if __name__ == '__main__':
    main()
