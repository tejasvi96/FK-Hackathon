import os
import shutil
import time
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
import torch.nn as nn
from util import *
from fk_utils import *
from loguru import logger
import ast
logger.add("./logs_testing.log")
tqdm.monitor_interval = 0
class Engine(object):
    def __init__(self, state={}):
        self.state = state
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()


        if self._state('image_size') is None:
            self.state['image_size'] = 224

        if self._state('batch_size') is None:
            self.state['batch_size'] = 64

        if self._state('workers') is None:
            self.state['workers'] = 25

        if self._state('device_ids') is None:
            self.state['device_ids'] = None

        if self._state('evaluate') is None:
            self.state['evaluate'] = False

        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 90

        if self._state('epoch_step') is None:
            self.state['epoch_step'] = []

        # meters
        self.state['meter_loss'] = tnt.meter.AverageValueMeter()
        # time measure
        self.state['batch_time'] = tnt.meter.AverageValueMeter()
        self.state['data_time'] = tnt.meter.AverageValueMeter()
        # display parameters
        if self._state('use_pb') is None:
            self.state['use_pb'] = True
        if self._state('print_freq') is None:
            self.state['print_freq'] = 0

    def _state(self, name):
        if name in self.state:
            return self.state[name]

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        loss = self.state['meter_loss'].value()[0]
        if display:
            if training:
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}'.format(self.state['epoch'], loss=loss))
            else:
                print('Test: \t Loss {loss:.4f}'.format(loss=loss))
        return loss

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        pass

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        # record loss
        
        self.state['loss_batch'] = self.state['loss'].data
#         print(self.state['loss_batch'])
#         print(self.state['loss'].data)
        self.state['meter_loss'].add(self.state['loss_batch'].cpu())
#         print(self.state['ap_meter'])
        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
#                 print('Epoch: [{0}][{1}/{2}]\t'
#                       'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
#                       'Data {data_time_current:.3f} ({data_time:.3f})\t'
#                       'Loss {loss_current:.4f} ({loss:.4f})'.format(
#                     self.state['epoch'], self.state['iteration'], len(data_loader),
#                     batch_time_current=self.state['batch_time_current'],
#                     batch_time=batch_time, data_time_current=self.state['data_time_batch'],
#                     data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))

                    logger.info('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
            else:
                
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
        testing=self.state['testing']
        input_var = torch.autograd.Variable(self.state['input'])
        if not testing:
            target_var = torch.autograd.Variable(self.state['target'])
        else:
            target_var=None
#         if not training:
#             input_var.volatile = True
#             target_var.volatile = True

        # compute output
        self.state['output'] = model(input_var)
        if not testing:
            self.state['loss'] = criterion(self.state['output'], target_var)
        else:
            self.state['loss']=0
        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            optimizer.step()

    def init_learning(self, model, criterion):

        if self._state('train_transform') is None:
            normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
            self.state['train_transform'] = transforms.Compose([
                MultiScaleCrop(self.state['image_size'], scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        if self._state('val_transform') is None:
            normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
            self.state['val_transform'] = transforms.Compose([
                Warp(self.state['image_size']),
                transforms.ToTensor(),
                normalize,
            ])

        self.state['best_score'] = 0

    def learning(self, model,model_cat, criterion, train_dataset, val_dataset,test_dataset=None, optimizer=None):

        self.init_learning(model, criterion)

        # define train and val transform
        
        
        # train_dataset.transform = self.state['train_transform']
        # train_dataset.target_transform = self._state('train_target_transform')
        # val_dataset.transform = self.state['val_transform']
        # val_dataset.target_transform = self._state('val_target_transform')

        # data loading code
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.state['batch_size'], shuffle=True,
                                                   num_workers=self.state['workers'])

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.state['batch_size'], shuffle=False,
                                                 num_workers=self.state['workers'])
        
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=self.state['batch_size'], shuffle=False,
                                                 num_workers=self.state['workers'])
        

        # optionally resume from a checkpoint
        if self._state('resume') is not None:
            if os.path.isfile(self.state['resume']):
                print("=> loading checkpoint '{}'".format(self.state['resume']))
                if self.state['use_gpu']:
                  checkpoint = torch.load(self.state['resume'])
                else:
                  checkpoint = torch.load(self.state['resume'],map_location='cpu')


                self.state['start_epoch'] = checkpoint['epoch']
#                 self.state['best_score'] = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.state['evaluate'], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.state['resume']))


        if self.state['use_gpu']:
            train_loader.pin_memory = True
            val_loader.pin_memory = True
            cudnn.benchmark = True

            list_of_devices = [torch.device(f'cuda:{i}') for i in self.state['device_ids']]
#             list_of_devices=[0,1,2,3]
#            model = torch.nn.DataParallel(model, device_ids=list_of_devices).cuda()
            
            map_location={'cuda:4':'cuda:0','cuda:5':'cuda:1','cuda:6':'cuda:2','cuda:7':'cuda:3'}
    
            pretrained_dict=torch.load(self.state['cat_model_file'],map_location=map_location)
#             pretrained_dict=torch.load(self.state['cat_model_file'],map_location='cpu')
#             pretrained_dict=torch.load(self.state['cat_model_file'])
            pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
            
            
            
            model_cat.load_state_dict(pretrained_dict)
            
            print(model)
            if not self.state['use_gpu']:
                model_cat.to('cpu').eval()
                model=model.to('cpu')
                criterion = criterion.cuda()
            else:
                model_cat = torch.nn.DataParallel(model_cat, device_ids=list_of_devices).cuda()
                model = torch.nn.DataParallel(model, device_ids=list_of_devices).cuda()
    
#             criterion = criterion.cuda()
        if self.state['testing']:
            prec1=self.evaluate(test_loader, model,model_cat, criterion)
            print("Evaluation Scores: "+str(prec1)) 
            return         
        
#         if self.state['evaluate']:
#             prec1=self.validate(val_loader, model, criterion)
#             print("Evaluation Scores: "+str(prec1)) 
#             return

        # TODO define optimizer
#         prec1 = self.validate(val_loader, model, criterion)
        prec1_old=0.0
        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            self.state['epoch'] = epoch
            lr = self.adjust_learning_rate(optimizer)
            print('lr:',lr)
            

            # train for one epoch
            self.train(train_loader, model, criterion, optimizer, epoch)
            # evaluate on validation set
            prec1 = self.validate(val_loader, model, criterion)
            
            print(prec1)
            logger.info("Epoch: "+str(epoch)+": Scores--"+str(prec1))
            # remember best prec@1 and save checkpoint
#             is_best = prec1 > self.state['best_score']
#             self.state['best_score'] = max(prec1, self.state['best_score'])
            if prec1>prec1_old or (epoch+1)%2==0:
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': self._state('arch'),
                    'state_dict': model.module.state_dict() if self.state['use_gpu'] else model.state_dict(),
    #                 'best_score': self.state['best_score'],
                    },
                    False,
                    filename=f'checkpoint_modified{epoch+1}.pth')
            
#             if  (epoch+1)%5==0:
#                 self.state['testing']=True
#                 prec2=self.evaluate(test_loader, model,model_cat, criterion,prec1)
#                 print("Evaluation Scores: "+str(prec2))
#                 self.state['testing']=False
                
            prec1_old = prec1   
            print(' *** best={best:.3f}'.format(best=self.state['best_score']))
        return self.state['best_score']

    def train(self, data_loader, model, criterion, optimizer, epoch):

        # switch to train mode
        model.train()

        self.on_start_epoch(True, model, criterion, data_loader, optimizer)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Training')

        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
#             if i==2:
#                 break
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(True, model, criterion, data_loader, optimizer)

            if self.state['use_gpu']:
                    self.state['target'] = self.state['target'].cuda()

            self.on_forward(True, model, criterion, data_loader, optimizer)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(True, model, criterion, data_loader, optimizer)
#         self.on_end_epoch(True, model, criterion, data_loader, optimizer)

#     def validate(self, data_loader, model, criterion):

#         # switch to evaluate mode
#         model.eval()

#         self.on_start_epoch(False, model, criterion, data_loader)

#         if self.state['use_pb']:
#             data_loader = tqdm(data_loader, desc='Test')

#         end = time.time()
#         for i, (input, target,cat) in enumerate(data_loader):
#             # measure data loading time
#             self.state['iteration'] = i
#             self.state['data_time_batch'] = time.time() - end
#             self.state['data_time'].add(self.state['data_time_batch'])

#             self.state['input'] = input
#             self.state['target'] = target

#             self.on_start_batch(False, model, criterion, data_loader)

#             if self.state['use_gpu']:
#                 self.state['target'] = self.state['target'].cuda()

#             self.on_forward(False, model, criterion, data_loader)

#             # measure elapsed time
#             self.state['batch_time_current'] = time.time() - end
#             self.state['batch_time'].add(self.state['batch_time_current'])
#             end = time.time()
#             # measure accuracy
#             self.on_end_batch(False, model, criterion, data_loader)

#         score = self.on_end_epoch(False, model, criterion, data_loader)

#         return score
    def validate(self, data_loader, model, criterion):

        # switch to evaluate mode
        model.eval()

        self.on_start_epoch(False, model, criterion, data_loader)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Test')

        end = time.time()
        scores=[]
        
        with torch.no_grad():
            for i, (input, target,cat) in enumerate(data_loader):
#                 if i==2:
#                     break
                attrs=[ast.literal_eval(item) for item in input[1]]
                # measure data loading time
                self.state['iteration'] = i
                self.state['data_time_batch'] = time.time() - end
                self.state['data_time'].add(self.state['data_time_batch'])
#                 print(attrs)
    #             exit()
#                 input=input.to('')
                self.state['input'] = input
                self.state['target'] = target

                self.on_start_batch(False, model, criterion, data_loader)

                if self.state['use_gpu']:
                    self.state['target'] = self.state['target'].cuda()

                self.on_forward(False, model, criterion, data_loader)     
                cats,values=decode_outputs(self.state['threshold'],
                                            self.state['value_decoding'],
                                            self.state['category_decoding'],
                                            cat,self.state['output'])
                predictions=get_outputs(self.state['category_attrs_mapping'],
                                                self.state['attrs_value_mapping'],
                                                cats,values,self.state['value_decoding'])
#                 print("+++")
#                 print(attrs)
#                 print("------")
#                 print(predictions)
#                 print("******")
#                 exit()
                score=evaluate_model_scores(attrs,predictions)
                scores.append(score)
#                 print("This is da score: ",str(score))
                # # measure elapsed time
                self.state['batch_time_current'] = time.time() - end
                self.state['batch_time'].add(self.state['batch_time_current'])

                end = time.time()
                # measure accuracy
                self.on_end_batch(False, model, criterion, data_loader)

        # score = self.on_end_epoch(False, model, criterion, data_loader)

        return sum(scores)/len(scores)
    def evaluate(self, data_loader, model,model_cat, criterion,val_loss=None):
        logger.info("In evaluate")

        # switch to evaluate mode
        model.eval()

        self.on_start_epoch(False, model, criterion, data_loader)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Test')

        end = time.time()
        scores=[]
        preds=[]
        filenames=[]
        catlist=[]
        value_list = []
        with torch.no_grad():
            for i, (input) in enumerate(data_loader):
#                 if i==2:
#                     break
                if self.state['use_gpu']:
                  input[0]=input[0].to('cuda')

    #             attrs=[ast.literal_eval(item) for item in input[1]]
                # measure data loading time
                self.state['iteration'] = i
                self.state['data_time_batch'] = time.time() - end
                self.state['data_time'].add(self.state['data_time_batch'])
    #                 print(attrs)
    #             exit()

                self.state['input'] = input
                self.state['target'] = None
#                 print(input[0].shape)
#                 input_var = torch.autograd.Variable(self.state['input'])
#                 print(input_var.shape)
                cat=model_cat(input[0])
#                 print(cat.shape)
#                 print("-----------")
#                 print(cat.shape)
                testing=self.state['testing']

                self.on_start_batch(False, model, criterion, data_loader)

#                 if self.state['use_gpu']:
#                     self.state['target'] = self.state['target'].cuda()

                self.on_forward(False, model, criterion, data_loader)     

                cats,values=decode_outputs(self.state['threshold'],
                                            self.state['value_decoding'],
                                            self.state['category_decoding'],
                                            cat,self.state['output'])
#                 catlist=[[i] for i in cats]
            
#                 catlist.append(cats)
#                 value_list.append(values)
                

#                 predictions=get_outputs(self.state['category_attrs_mapping'],
#                                                 self.state['attrs_value_mapping'],
#                                                 cats,values,self.state['value_decoding'])
                predictions=get_outputs(self.state['category_attrs_mapping'],
                                                self.state['attrs_value_mapping'],
                                                cats,values,self.state['value_decoding'])
    #                 print("+++")
    #                 print(attrs)
    #                 print("------")
    #                 print(predictions)
    #                 print("******")
    #                 exit()
#                 print(cats[0])
#                 exit()
#                 print(predictions)
#                 print(type(predictions[0]))
#                 l=list()
#                 l.append(cats[0])
                # predictions[0]['vertical']=cats
                n_items=len(predictions)

                for t in range(n_items):
                  predictions[t]['vertical']=[cats[t]]
#                 exit()
#                 predictions['vertical']=cats[0]
                preds.append(predictions)
                filenames.append(input[1])
#                 print(predictions)
#                 exit()
#                 score=evaluate_model_scores(attrs,predictions)
#                 scores.append(score)
    #                 print("This is da score: ",str(score))
                # # measure elapsed time
                self.state['batch_time_current'] = time.time() - end
                self.state['batch_time'].add(self.state['batch_time_current'])

                end = time.time()
                # measure accuracy
#                 self.on_end_batch(False, model, criterion, data_loader)

        # score = self.on_end_epoch(False, model, criterion, data_loader)
        import pandas as pd
        
        filenames = [f for sublist in filenames for f in sublist]
        preds = [f for sublist in preds for f in sublist]
#         probs = [f for sublist in value_list for f in sublist]
#         cats = [f for sublist in catlist for f in sublist]
        
        data=pd.DataFrame()
        data["filename"]=filenames
        data["predictions"]=preds
#         data['cat']=cats
#         data['probs']=probs
        file_names=self.state['output_file']
        val_loss=0.000
        file_names=file_names.split('.csv')[0]+str(round(val_loss,3)).replace('.','_')+".csv"
        data.to_csv(file_names,index=False)
        return 0

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):

        if self._state('save_model_path') is not None:
            filename_ = filename
            filename = os.path.join(self.state['save_model_path'], filename_)
            if not os.path.exists(self.state['save_model_path']):
                os.makedirs(self.state['save_model_path'])
        print('save model {filename}'.format(filename=filename))
        torch.save(state, filename)
        if is_best:
            filename_best = 'model_best.pth.tar'
            if self._state('save_model_path') is not None:
                filename_best = os.path.join(self.state['save_model_path'], filename_best)
            shutil.copyfile(filename, filename_best)
            if self._state('save_model_path') is not None:
                if self._state('filename_previous_best') is not None:
                    os.remove(self._state('filename_previous_best'))
                filename_best = os.path.join(self.state['save_model_path'], 'model_best_{score:.4f}.pth.tar'.format(score=state['best_score']))
                shutil.copyfile(filename, filename_best)
                self.state['filename_previous_best'] = filename_best

    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr_list = []
        decay = 0.1 if sum(self.state['epoch'] == np.array(self.state['epoch_step'])) > 0 else 1.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay
            lr_list.append(param_group['lr'])
        return np.unique(lr_list)


class MultiLabelMAPEngine(Engine):
    def __init__(self, state):
        Engine.__init__(self, state)
        if self._state('difficult_examples') is None:
            self.state['difficult_examples'] = False
        self.state['ap_meter'] = AveragePrecisionMeter(self.state['difficult_examples'])

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        Engine.on_start_epoch(self, training, model, criterion, data_loader, optimizer)
        self.state['ap_meter'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        map = 100 * self.state['ap_meter'].value().mean()
        loss = self.state['meter_loss'].value()[0]
        OP, OR, OF1, CP, CR, CF1 = self.state['ap_meter'].overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.state['ap_meter'].overall_topk(3)
        if display:
            if training:
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}\t'
                      'mAP {map:.3f}'.format(self.state['epoch'], loss=loss, map=map))
                print('OP: {OP:.4f}\t'
                      'OR: {OR:.4f}\t'
                      'OF1: {OF1:.4f}\t'
                      'CP: {CP:.4f}\t'
                      'CR: {CR:.4f}\t'
                      'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
            else:
                print('Test: \t Loss {loss:.4f}\t mAP {map:.3f}'.format(loss=loss, map=map))
                print('OP: {OP:.4f}\t'
                      'OR: {OR:.4f}\t'
                      'OF1: {OF1:.4f}\t'
                      'CP: {CP:.4f}\t'
                      'CR: {CR:.4f}\t'
                      'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
                print('OP_3: {OP:.4f}\t'
                      'OR_3: {OR:.4f}\t'
                      'OF1_3: {OF1:.4f}\t'
                      'CP_3: {CP:.4f}\t'
                      'CR_3: {CR:.4f}\t'
                      'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k, CR=CR_k, CF1=CF1_k))

        return map

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        testing=self.state['testing']
        if not testing:
            self.state['target_gt'] = self.state['target'].clone()
            self.state['target'][self.state['target'] == 0] = 1
            self.state['target'][self.state['target'] == -1] = 0

        input = self.state['input']
        self.state['input'] = input[0]
        self.state['name'] = input[1]

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        Engine.on_end_batch(self, training, model, criterion, data_loader, optimizer, display=False)

        # measure mAP
        self.state['ap_meter'].add(self.state['output'].data, self.state['target_gt'])

        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
#                 print('Epoch: [{0}][{1}/{2}]\t'
#                       'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
#                       'Data {data_time_current:.3f} ({data_time:.3f})\t'
#                       'Loss {loss_current:.4f} ({loss:.4f})'.format(
#                     self.state['epoch'], self.state['iteration'], len(data_loader),
#                     batch_time_current=self.state['batch_time_current'],
#                     batch_time=batch_time, data_time_current=self.state['data_time_batch'],
#                     data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
                logger.info('Epoch: [{0}][{1}/{2}]\t'
          'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
          'Data {data_time_current:.3f} ({data_time:.3f})\t'
          'Loss {loss_current:.4f} ({loss:.4f})'.format(
        self.state['epoch'], self.state['iteration'], len(data_loader),
        batch_time_current=self.state['batch_time_current'],
        batch_time=batch_time, data_time_current=self.state['data_time_batch'],
        data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))


class GCNMultiLabelMAPEngine(MultiLabelMAPEngine):
    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
        feature_var = torch.autograd.Variable(self.state['feature']).float()
        target_var = torch.autograd.Variable(self.state['target']).float()
        inp_var = torch.autograd.Variable(self.state['input']).float().detach()  # one hot
#         if not training:
#             feature_var.volatile = True
#             target_var.volatile = True
#             inp_var.volatile = True

        # compute output
        testing=self.state['testing']
        # featute_var=feature_var.to('cuda')
        # inp_var=inp_var.to('cuda')
#         print(feature_var.shape)
#         print
        self.state['output'] = model(feature_var, inp_var)
        # testing=True
        if not testing:
            self.state['loss'] = criterion(self.state['output'], target_var)
        else:
            self.state['loss']=0
                
        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            nn.utils.clip_grad_norm(model.parameters(), max_norm=10.0)
            optimizer.step()


    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
#         print(testing)
        # testing=True
        testing=self.state['testing']
        if  not testing:
            self.state['target_gt'] = self.state['target'].clone()
        # self.state['target'][self.state['target'] == 0] = 1
        # self.state['target'][self.state['target'] == -1] = 0

        input = self.state['input']
        self.state['feature'] = input[0]
        self.state['out'] = input[1]
        self.state['input'] = input[2]
