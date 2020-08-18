import torch
from torch import nn

from models import c3d, squeezenet, mobilenet, shufflenet, mobilenetv2, shufflenetv2, resnext, resnet, resnetl
import pdb

from apex.parallel import DistributedDataParallel as DDP

def generate_model(opt): 
    assert opt.model in ['c3d', 'squeezenet', 'mobilenet', 'resnext', 'resnet', 'resnetl',
                         'shufflenet', 'mobilenetv2', 'shufflenetv2']

    if opt.model == 'resnetl':
        assert opt.model_depth in [10] # 깊이는 10만 된다!

        from models.resnetl import get_fine_tuning_parameters # 전이학습을 위함.

        if opt.model_depth == 10:
            model = resnetl.resnetl10( 
                num_classes=opt.n_classes,  # 클래스 개수.
                shortcut_type=opt.resnet_shortcut,  # 디폴트 값 : 'B'
                sample_size=opt.sample_size, # 디폴트 값 : 112
                sample_duration=opt.sample_duration) # 디폴트 값 : 16 , 입력 프레임
    elif opt.model == 'resnext':
        assert opt.model_depth in [50, 101, 152]
        from models.resnext import get_fine_tuning_parameters
        if opt.model_depth == 50:
            model = resnext.resnext50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = resnext.resnext101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = resnext.resnext152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)       

    if not opt.no_cuda:

        if opt.gpus == '0':
            model = model.cuda()
        else:
            opt.gpus = opt.local_rank
            torch.cuda.set_device(opt.gpus)
            model = model.cuda()

        #model = nn.DataParallel(model, device_ids=None) # 병렬처리를 위함인데, 안쓸 것 같음.
        pytorch_total_params = sum(p.numel() for p in model.parameters() if
                               p.requires_grad) # grad를 하는 파라미터들을 모두 더한다.
        print("Total number of trainable parameters: ", pytorch_total_params) # 파라미터 값 출력
        
        if opt.pretrain_path: # 전이학습.
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path, map_location=torch.device('cpu'))
            # print(opt.arch)
            # print(pretrain['arch'])
            # assert opt.arch == pretrain['arch']
            model = modify_kernels(opt, model, opt.pretrain_modality)
            model.load_state_dict(pretrain['state_dict'])
            

            if opt.model in  ['mobilenet', 'mobilenetv2', 'shufflenet', 'shufflenetv2']:
                model.module.classifier = nn.Sequential(
                                nn.Dropout(0.5),
                                nn.Linear(model.module.classifier[1].in_features, opt.n_finetune_classes))
                model.module.classifier = model.module.classifier.cuda()
            elif opt.model == 'squeezenet':
                model.module.classifier = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Conv3d(model.module.classifier[1].in_channels, opt.n_finetune_classes, kernel_size=1),
                                nn.ReLU(inplace=True),
                                nn.AvgPool3d((1,4,4), stride=1))
                model.module.classifier = model.module.classifier.cuda()
            else:
                model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
                model.module.fc = model.module.fc.cuda()

            model = modify_kernels(opt, model, opt.modality)
        else: # 전이학습이  아닐때
            pass
            model = modify_kernels(opt, model, opt.modality)

        parameters = get_fine_tuning_parameters(model, opt.ft_portion) # 전이학습할때만 적용 지금은 그냥 파라미터 그대로 반환됨.
    return model, parameters


def _construct_depth_model(base_model):
    # modify the first convolution kernels for Depth input
    modules = list(base_model.modules())

    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d),
                                 list(range(len(modules)))))[0]
    conv_layer = modules[first_conv_idx]
    container = modules[first_conv_idx - 1]

    # modify parameters, assume the first blob contains the convolution kernels
    motion_length = 1
    params = [x.clone() for x in conv_layer.parameters()]
    kernel_size = params[0].size()
    new_kernel_size = kernel_size[:1] + (1*motion_length,  ) + kernel_size[2:]
    new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

    new_conv = nn.Conv3d(1, conv_layer.out_channels, conv_layer.kernel_size, conv_layer.stride,
                         conv_layer.padding, bias=True if len(params) == 2 else False)
    new_conv.weight.data = new_kernels
    if len(params) == 2:
        new_conv.bias.data = params[1].data # add bias if neccessary
    layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

    # replace the first convlution layer
    setattr(container, layer_name, new_conv)

    return base_model

def _construct_rgbdepth_model(base_model):
    # modify the first convolution kernels for RGB-D input
    modules = list(base_model.modules())

    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d),
                           list(range(len(modules)))))[0]
    conv_layer = modules[first_conv_idx]
    container = modules[first_conv_idx - 1]
    # modify parameters, assume the first blob contains the convolution kernels
    motion_length = 1
    params = [x.clone() for x in conv_layer.parameters()]
    kernel_size = params[0].size()
    new_kernel_size = kernel_size[:1] + (1 * motion_length,) + kernel_size[2:]
    new_kernels = torch.mul(torch.cat((params[0].data, params[0].data.mean(dim=1,keepdim=True).expand(new_kernel_size).contiguous()), 1), 0.6)
    new_kernel_size = kernel_size[:1] + (3 + 1 * motion_length,) + kernel_size[2:]
    new_conv = nn.Conv3d(4, conv_layer.out_channels, conv_layer.kernel_size, conv_layer.stride,
                         conv_layer.padding, bias=True if len(params) == 2 else False)
    new_conv.weight.data = new_kernels
    if len(params) == 2:
        new_conv.bias.data = params[1].data  # add bias if neccessary
    layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

    # replace the first convolution layer
    setattr(container, layer_name, new_conv)
    return base_model

def _modify_first_conv_layer(base_model, new_kernel_size1, new_filter_num):
    modules = list(base_model.modules())
    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d),
                                               list(range(len(modules)))))[0] # 첫번째 conv의 idx를 가져옴.
    conv_layer = modules[first_conv_idx]
    #conv_layer => resnetl10기준 Conv3d(3, 16, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
    container = modules[first_conv_idx - 1]
    # container는 모델 전체를 뜻함.
 
    new_conv = nn.Conv3d(new_filter_num, conv_layer.out_channels, kernel_size=(new_kernel_size1,7,7),
                         stride=(1,2,2), padding=(1,3,3), bias=False)
    layer_name = list(container.state_dict().keys())[0][:-7] # resnetl10 기준 'conv1'이 찍힘.

    setattr(container, layer_name, new_conv) # 기존의 conv1을 new_conv로 수정하는 함수 
    # setattr(object, name, value) : object에 존재하는 속성의 값을 바꾸거나, 새로운 속성을 생성하여 값을 부여한다.
    return base_model

def modify_kernels(opt, model, modality):
    if modality == 'RGB' and opt.model not in ['c3d', 'squeezenet', 'mobilenet','shufflenet', 'mobilenetv2', 'shufflenetv2']:
        print("[INFO]: RGB model is used for init model")
        model = _modify_first_conv_layer(model,3,3) ##### Check models trained (3,7,7) or (7,7,7)
    elif modality == 'Depth':
        print("[INFO]: Converting the pretrained model to Depth init model")
        model = _construct_depth_model(model)
        print("[INFO]: Done. Flow model ready.")
    elif modality == 'RGB-D':
        print("[INFO]: Converting the pretrained model to RGB+D init model")
        model = _construct_rgbdepth_model(model)
        print("[INFO]: Done. RGB-D model ready.")
    modules = list(model.modules()) 
    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d),
                                               list(range(len(modules)))))[0]
    #conv_layer = modules[first_conv_idx]
    #if conv_layer.kernel_size[0]> opt.sample_duration:
     #   model = _modify_first_conv_layer(model,int(opt.sample_duration/2),1)
    return model
