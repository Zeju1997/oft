from datasets.coco_dataset import CocoCannyDataset, CocoDepthDataset, CocoCanny1KDataset, CocoCanny5KDataset, CocoCanny20KDataset, CocoCanny50KDataset
from datasets.ade20k_dataset import ADE20kSegmDataset, AblationADE20kSegmDataset
from datasets.deepfashion_dataset import DeepFashionDenseposeDataset
from datasets.laion_dataset import LaionSketchDataset
from datasets.celebhq_dataset import CelebHQDataset

def return_dataset(control, is_t2i=False, full=False, n_samples=0):
    if control == 'segm':
        logger_freq = 1000
        data_name = 'ADE20K'
        print('[Control]: {}'.format(control), '[Dataset]: {}'.format(data_name))
        train_dataset = ADE20kSegmDataset(split='train', is_t2i=is_t2i, full=full)
        val_dataset = ADE20kSegmDataset(split='val', is_t2i=is_t2i, full=full)
        max_epochs = 20

    elif control == 'canny':
        data_name = 'COCO'
        logger_freq = 3000
        print('[Control]: {}'.format(control), '[Dataset]: {}'.format(data_name))
        train_dataset = CocoCannyDataset(split='train', is_t2i=is_t2i, full=full)
        val_dataset = CocoCannyDataset(split='val', is_t2i=is_t2i, full=full)
        max_epochs = 11

    elif control == 'depth':
        data_name = 'COCO'
        logger_freq = 500
        print('[Control]: {}'.format(control), '[Dataset]: {}'.format(data_name))
        train_dataset = CocoDepthDataset(split='train', is_t2i=is_t2i, full=full)
        val_dataset = CocoDepthDataset(split='val', is_t2i=is_t2i, full=full)
        max_epochs = 20

    elif control == 'canny50k':
        data_name = 'COCO'
        logger_freq = 300
        print('[Control]: {}'.format(control), '[Dataset]: {}'.format(data_name))
        train_dataset = CocoCanny50KDataset(split='train', is_t2i=is_t2i, full=full)
        val_dataset = CocoCanny50KDataset(split='val', is_t2i=is_t2i, full=full)
        max_epochs = 10
    
    elif control == 'canny20k':
        data_name = 'COCO'
        logger_freq = 300
        print('[Control]: {}'.format(control), '[Dataset]: {}'.format(data_name))
        train_dataset = CocoCanny20KDataset(split='train', is_t2i=is_t2i, full=full)
        val_dataset = CocoCanny20KDataset(split='val', is_t2i=is_t2i, full=full)
        max_epochs = 10
    
    elif control == 'canny5k':
        data_name = 'COCO'
        logger_freq = 300
        print('[Control]: {}'.format(control), '[Dataset]: {}'.format(data_name))
        train_dataset = CocoCanny5KDataset(split='train', is_t2i=is_t2i, full=full)
        val_dataset = CocoCanny5KDataset(split='val', is_t2i=is_t2i, full=full)
        max_epochs = 10

    elif control == 'canny1k':
        data_name = 'COCO'
        logger_freq = 300
        print('[Control]: {}'.format(control), '[Dataset]: {}'.format(data_name))
        train_dataset = CocoCanny1KDataset(split='train', is_t2i=is_t2i, full=full)
        val_dataset = CocoCanny1KDataset(split='val', is_t2i=is_t2i, full=full)
        max_epochs = 30

    elif control == 'sketch':
        data_name = 'LAION'
        logger_freq = 1000
        print('[Control]: {}'.format(control), '[Dataset]: {}'.format(data_name))
        train_dataset = LaionSketchDataset(split='train', is_t2i=is_t2i, full=full)
        val_dataset = LaionSketchDataset(split='val', is_t2i=is_t2i, full=full)
        max_epochs = 10

    elif control == 'densepose':
        data_name = 'DeepFashion'
        logger_freq = 300
        print('[Control]: {}'.format(control), '[Dataset]: {}'.format(data_name))
        train_dataset = DeepFashionDenseposeDataset(split='train', is_t2i=is_t2i, full=full)
        val_dataset = DeepFashionDenseposeDataset(split='val', is_t2i=is_t2i, full=full)
        max_epochs = 20

    elif control == 'landmark':
        logger_freq = 1000
        data_name = 'CelebA'
        print('[Control]: {}'.format(control), '[Dataset]: {}'.format(data_name))
        train_dataset = CelebHQDataset(split='train', is_t2i=is_t2i, full=full)
        val_dataset = CelebHQDataset(split='val', is_t2i=is_t2i, full=full)
        max_epochs = 20
    
    elif control == 'ablation_segm':
        logger_freq = 100000000000
        data_name = 'ADE20K'
        print('[Num of Samples]: {}'.format(n_samples), '[Control]: {}'.format(control), '[Control]: {}'.format(control), '[Dataset]: {}'.format(data_name))
        train_dataset = AblationADE20kSegmDataset(split='train', is_t2i=is_t2i, full=full, n_samples=n_samples)
        val_dataset = AblationADE20kSegmDataset(split='val', is_t2i=is_t2i, full=full)
        max_epochs = 1
        
    else:
        # default code
        print('unknown control!')
        sys.exit()

    return train_dataset, val_dataset, data_name, logger_freq, max_epochs