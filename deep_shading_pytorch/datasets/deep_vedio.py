import os.path
import glob
from .listdataset import ListDataset
from .util import split2list



def make_dataset(dir, split=None):
    '''Will search for triplets that go by the pattern '[name]_img1.ppm  [name]_img2.ppm    [name]_flow.flo' '''
    images = []

    #staircase dataset
    for image in sorted(glob.glob(os.path.join(dir, '*.exr'))):
        if image[-5] == 'a':
            continue

        if image[-14] == '2':
            continue
            
        image_num = image[-14:-4]
        image_1 = image
        image_2 = dir + "/" + image_num + "_a.exr"

        image_flow = dir + "/Flow/" + image_num + ".flo"
        images.append([image_1, image_2, image_flow])


    #board_game dataset
    #future structure [[input1, input2, input3, input4, input5], output]
    # for image in sorted(glob.glob(os.path.join(dir, 'Diffuse/*.exr'))):
    #     basename = image[-14:]
    #     diffuse = os.path.join(dir, 'Diffuse', basename)
    #     normal = os.path.join(dir, 'Normal', basename)
    #     position = os.path.join(dir, 'Position', basename)
    #     albedo = os.path.join(dir, 'Albedo', basename)
    #     gt = os.path.join(dir, 'GT', basename)
        
    #     flow_map = os.path.join(dir, 'Flow', basename[:10] + '.flo')

    #     next_basename = str(int(basename[:10]) + 1).zfill(10) + '.exr'
    #     next_diffuse = os.path.join(dir, 'Diffuse', next_basename)
    #     next_normal = os.path.join(dir, 'Normal', next_basename)
    #     next_position = os.path.join(dir, 'Position', next_basename)
    #     next_albedo = os.path.join(dir, 'Albedo', next_basename)
    #     next_gt = os.path.join(dir, 'GT', next_basename)

    #     if not os.path.isfile(next_diffuse):
    #         continue

    #     images.append([[diffuse, normal, position, albedo], gt, [next_diffuse, next_normal, next_position, next_albedo], next_gt, flow_map])

    return split2list(images, split, default_split=1.0)


def deep_vedio(root, transform=None, target_transform=None,
                  co_transform=None, split=None):
    train_list, test_list = make_dataset(root,split)
    train_dataset = ListDataset(root, train_list, transform, target_transform, co_transform)
    test_dataset = ListDataset(root, test_list, transform, target_transform)

    return train_dataset, test_dataset
