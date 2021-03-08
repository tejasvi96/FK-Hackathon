import argparse
from engine import *
from models import *
from coco import *
from util import *
import os
import torchvision.models as models
import torch
import torch.nn as nn
# from dataset import Flipkart2021
from torch import optim
import torch.utils.data as data
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
from dataset import *
parser = argparse.ArgumentParser(description='FK Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset (e.g. data/')
parser.add_argument('--image-size', '-i', default=448, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--device_ids', default=[0,1], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=48, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--imagesdir', default='', type=str, metavar='PATH',
                    help='path to testdir (default: none)')
parser.add_argument('--testnames', default='', type=str, metavar='PATH',
                    help='path to testnames (default: none)')
parser.add_argument('--output_file', default='', type=str, metavar='PATH',
                    help='path to testnames (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-t', '--test', dest='test', action='store_true',
                    help='test model on testing set')


class CategoryModel(nn.Module):
    def __init__(self, num_categories):
        super(CategoryModel, self).__init__()
        resnet101 = models.resnet101(pretrained=True)
#         resnet18 = models.resnet18(pretrained=True)
        self.model = nn.Sequential(*list(resnet101.children())[:-2])
        self.avgpool= nn.AdaptiveAvgPool2d(output_size=(1,1))  
        self.inp_features = resnet101.fc.in_features
        self.fc = nn.Linear(self.inp_features, num_categories)
    def forward(self, x):
#         print(self.inp_features)
        x = self.model(x)
        x = self.avgpool(x)
#         print(x.shape)
        batch_size = x.shape[0]
#         print(x.shape)
        x = x.view(batch_size, -1)
#         print(x.shape)
        x = self.fc(x)
#         print(x.shape)
        return x

def main_fk():
    torch.cuda.empty_cache()
    global args, best_prec1, use_gpu
    args = parser.parse_args()
    device = ','.join(list(map(str, args.device_ids)))
#     print(device)
    
#     os.environ['CUDA_VISIBLE_DEVICES'] = device
    params={}
    params['train_file']='./train90_modified.csv'
    params['val_file']='./extra/val_data.csv'
    params['test_file']='./extra/Test_Filenames.csv'
    params['allowed_attrs_file']='./extra/Attribute_allowedvalues.npy'
    params['category_attrs_mapping']='./extra/vertical_attributes.npy'
    params['adj_file']='./extra/p_matrix_complete.npy'
    params['train_image_dir']='./train90_images'
    params['val_image_dir']='./val_images'
#     params['image_dir']="./test_images"
    params['values_dir']="./extra/values.pkl"
    params['eval_threshold']=0.4
    params['is_training']=True
    params['cat_model_file']='./cat_model_res101_SGD.pt'

    params['test_file']=args.testnames
    params['image_dir']=args.imagesdir
    use_gpu = torch.cuda.is_available()
    # use_gpu=False
    # print(params['train_file'])
    train_dataset=Flipkart2021(params['train_image_dir'],params['train_file'],params['category_attrs_mapping'],params['allowed_attrs_file'],params['values_dir'],params['is_training'])
    val_dataset=Flipkart2021(params['val_image_dir'],params['val_file'],params['category_attrs_mapping'],params['allowed_attrs_file'],params['values_dir'],not params['is_training'])
    test_dataset=Flipkart2021_eval(params['image_dir'],params['test_file'],params['category_attrs_mapping'],params['allowed_attrs_file'],params['values_dir'],not params['is_training'])
#     test_dataset = None

    # train_dataset = COCO2014(args.data, phase='train', inp_name='data/coco/coco_glove_word2vec.pkl')
    # val_dataset = COCO2014(args.data, phase='val', inp_name='data/coco/coco_glove_word2vec.pkl')
    num_classes = 802

    model = gcn_resnet101(num_classes=num_classes, t=0.4, adj_file=params['adj_file'])

    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()

    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,'output_file':args.output_file,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes':num_classes,'device_ids':args.device_ids,'print_freq':500,'testing':args.test,'cat_model_file':params['cat_model_file']}
    state['difficult_examples'] = True
    state['save_model_path'] = 'checkpoint/flipkart/'
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    # state['device_ids'] = args.device_ids
    vertical_attribute_dict = np.load(params['category_attrs_mapping'],allow_pickle=True).tolist()
    category_list = [k for v, k in enumerate(vertical_attribute_dict.keys())]
    category_list.sort()
    category_mapping={}
    #for ind,keys in enumerate(vertical_attribute_dict.keys()): 
    #    category_mapping[ind]=keys
    category_mapping = {v:k for v, k in enumerate(category_list)}
    print(category_mapping)
    picklefile = open(params['values_dir'], 'rb')
    values_obj = pickle.load(picklefile)
#     values_obj=pickle.load(params['values_dir'])
    print(args.test)
    state['category_decoding']=category_mapping
    state['value_decoding']=values_obj
    
    state['category_attrs_mapping']=train_dataset.category_attribute_map.copy()
#     del state['category_attrs_mapping']['vertical']
    
    state['attrs_value_mapping']=train_dataset.allowed_values
    state['threshold']=0.25
    if args.evaluate:
        # need to load the val_object also if we need to decode
        state['evaluate'] = True
    params['num_categories']=26
    
    cat_model=CategoryModel(params['num_categories'])
#     print(cat_mo
#     cat_model = nn.DataParallel(cat_model, device_ids=args.device_ids)
#     cat_model.load_state_dict(torch.load(params['cat_model_file'],map_location=torch.device('cuda:0')))
#     cat_model=cat_model.cuda()
#     cat_model.eval()
#     self.state['category_model']=cat_model
    

    engine = GCNMultiLabelMAPEngine(state)
    engine.learning(model,cat_model, criterion, train_dataset, val_dataset,test_dataset, optimizer)

if __name__ == '__main__':
    main_fk()
