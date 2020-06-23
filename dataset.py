from datasets.KSL import KSL

def get_training_set(opt, spatial_transform, temporal_transform,
                     target_transform):
    assert opt.dataset in ['KSL']

    if opt.dataset == 'KSL':
        training_data = KSL(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)

    return training_data

def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform):
    assert opt.dataset in ['KSL']

    if opt.dataset == 'KSL':
        validation_data = KSL(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)

    return validation_data


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset in ['kinetics', 'jester', 'ucf101', 'egogesture', 'nvgesture']
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'
    if opt.dataset == 'kinetics':
        test_data = Kinetics(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'jester':
        test_data = Jester(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'ucf101':
        test_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'egogesture':
        test_data = EgoGesture(
            opt.video_path,
            opt.annotation_path,
            subset,
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            modality=opt.modality,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'nvgesture':
        test_data = NV(
            opt.video_path,
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality)
    return test_data

def get_online_data(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset in [ 'egogesture', 'nvgesture']
    whole_path = opt.whole_path
    if opt.dataset == 'egogesture':
        online_data = EgoGestureOnline(
            opt.annotation_path,  
            opt.video_path,
            opt.whole_path,  
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            modality="RGB-D",
            stride_len = opt.stride_len,
            sample_duration=opt.sample_duration)
    if opt.dataset == 'nvgesture':
        online_data = NVOnline(
            opt.annotation_path,  
            opt.video_path,
            opt.whole_path,  
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            modality="RGB-D",
            stride_len = opt.stride_len,
            sample_duration=opt.sample_duration)
    
    return online_data
