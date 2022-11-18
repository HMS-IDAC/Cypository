mode = 'test' # 'train', 'test', 'deploy'
num_epochs = 10 # number of train epochs
model_path = '3Dcytosegdense.pt'
dataset_path = 'F:/3Dseg/maskRCNN training patches dense/train'
train_subset_fraction = 0.97 # fraction of dataset used to train; remaining goes to 'test' subset
deploy_path_in = 'F:/3Dseg/maskRCNN training patches/deploy'
deploy_path_out = 'F:/3Dseg/maskRCNN training patches/deploy/output'

# -------------------------

# adapted from
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb
# by
# Marcelo Cicconet

import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
import math
import matplotlib.pyplot as plt
from skimage import morphology

from toolbox import listfiles, tifread, uint8Gray_to_uint8RGB, imread, Compose, RandomHorizontalFlip, ToTensor, \
    get_transform, collate_fn, reduce_dict, imshow, fileparts, imwrite

class CellsDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, load_annotations=True):
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = listfiles(root, 'Img.tif')
        self.ants = None
        if load_annotations:
            self.ants = listfiles(root, 'Ant.png')

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.imgs[idx]


        # img = uint8Gray_to_uint8RGB(tifread(img_path))
        img= tifread(img_path)
        img = np.moveaxis(img,0,-1)

        target = None
        if self.ants:
            ant_path = self.ants[idx]
            mask = imread(ant_path)
            # mask=mask-1
            obj_ids = np.unique(mask)
            # first id is the background, so remove it
            obj_ids = obj_ids[1:]


            # split the color-encoded mask into a set
            # of binary masks
            masks = mask == obj_ids[:, None, None]
            # get bounding box coordinates for each mask
            num_objs = len(obj_ids)
            boxes = []

            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if (xmin == xmax) or (ymin == ymax):
                    continue
                boxes.append([xmin, ymin, xmax, ymax])
            #
            # print(ant_path)
            # print(num_objs)
            # print(obj_ids)
            # im2=img
            # for i in range(num_objs):
            #         im2[boxes[i][1]:boxes[i][3], boxes[i][0]] = 255
            #         im2[boxes[i][1]:boxes[i][3], boxes[i][2]] = 255
            #         im2[boxes[i][1],boxes[i][0]:boxes[i][2]] = 255
            #         im2[boxes[i][3], boxes[i][0]:boxes[i][2]] = 255
            #
            # plt.imshow(im2, cmap='gray')
            # plt.show()



            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            labels = torch.ones((num_objs,), dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)

            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd


        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_instance_segmentation_model(num_classes):

    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(box_detections_per_img=300,pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model



def train_one_epoch(model, optimizer, data_loader, device):
    # model.train()
    model.to(device)

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
#         pdb.set_trace()

        # reduce losses over all GPUs for logging purposes
#         import pdb; pdb.set_trace()
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        # if not math.isfinite(loss_value):
        #     print("Loss is {}, stopping training".format(loss_value))
        #     print(loss_dict_reduced)
        #     sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
    return loss_value

# evaluate on gpu
@torch.no_grad()
def evaluate(model, data_loader, device):
    # model.eval()
    model.to(device)
    avg_dice_dataset = 0
    n_images = len(data_loader.dataset)
    rand_idx = np.random.randint(n_images)
    rand_log_img = None
    idx_image = 0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)#, targets)
        # when you call model(image,targets), it changes the size of ground truth masks in targets
        
        avg_dice_batch = 0
        for i in range(len(outputs)):
            out_i = outputs[i]
            ant_i = targets[i]
#             print(out_i.keys())
#             print(ant_i.keys())
            pred_mask, _ = torch.max(out_i['masks'],dim=0)
            pred_mask = torch.squeeze(pred_mask).cpu().numpy()
            grtr_mask, _ = torch.max(ant_i['masks'],dim=0)
            grtr_mask = grtr_mask.cpu().numpy().astype(np.float32)
            
            # import pdb; pdb.set_trace()
            # print(images[i].shape, pred_mask.shape, grtr_mask.shape)
            # pred_mask = imresizeDouble(pred_mask, list(grtr_mask.shape))
            
            # https://www.jeremyjordan.me/semantic-segmentation/
            # https://arxiv.org/pdf/1606.04797.pdf
            dice_coef = 2*np.sum(pred_mask*grtr_mask)/(np.sum(pred_mask**2)+np.sum(grtr_mask**2))
            avg_dice_dataset += dice_coef
            if idx_image == rand_idx:
                blue = np.zeros((pred_mask.shape))
                rand_log_img = np.stack([pred_mask, grtr_mask, blue],axis=2)
#                 imshow(images[i].cpu().numpy()[0,:,:])
#                 imshow(grtr_mask)
#                 print(images[i].cpu().numpy()[0,:,:].shape, grtr_mask.shape)
            idx_image += 1
    avg_dice_dataset /= n_images
#     print('avg_dice_dataset', avg_dice_dataset)
    return avg_dice_dataset, rand_log_img
            
#             print(pred_mask.shape, np.max(pred_mask), grtr_mask.shape, np.max(grtr_mask))
#             print(pred_mask.dtype, np.max(pred_mask), grtr_mask.dtype, np.max(grtr_mask))

# dice_coef, rand_img = evaluate(model, data_loader_test, device=device_train)
# print(dice_coef)
# imshow(rand_img)
if __name__ == '__main__':

    if mode == 'train' or mode == 'test':
        # use our dataset and defined transformations
        dataset = CellsDataset(dataset_path, get_transform(train=True))
        dataset_test = CellsDataset(dataset_path, get_transform(train=False))

        # split the dataset in train and test set
        torch.manual_seed(1)
        indices = torch.randperm(len(dataset)).tolist()
        # print(indices)
        n_train = int(train_subset_fraction*len(dataset))
        dataset = torch.utils.data.Subset(dataset, indices[:n_train])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[n_train:])

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=3, shuffle=True, num_workers=4,
            collate_fn=collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=3, shuffle=False, num_workers=4,
            collate_fn=collate_fn)

        print('n train', len(dataset), 'n test', len(dataset_test))

    device_train = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(device_train)

    if mode == 'train':
        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.0005,
                                    momentum=0.9, weight_decay=0.005)

        # and a learning rate scheduler which decreases the learning rate by
        # 2x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=10,
                                                       gamma=0.7)



    if mode == 'train':
        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            model.train()
            # import pdb; pdb.set_trace()
            loss_train = train_one_epoch(model, optimizer, data_loader, device_train)
            print('epoch', epoch, 'loss_train', loss_train)
            # update the learning rate
            lr_scheduler.step()

            # evaluate on the test dataset
            model.eval()
            dice_test, rand_img = evaluate(model, data_loader_test, device=device_train)
            print('epoch', epoch, 'dice_test', dice_test)
            # imshow(rand_img)

        torch.save(model.state_dict(), model_path)

    if mode == 'test':
        model.load_state_dict(torch.load(model_path))

        model.eval()
        with torch.no_grad():
            model.to(device_train)

            for img_index in range(len(dataset_test)):
                img, _ = dataset_test[img_index]

                prediction = model([img.to(device_train)])

                im = np.mean(img.numpy(),axis=0)

                p = prediction[0]['masks'][:, 0].cpu().numpy()
                p_max = np.max(p,axis=0)

                bb = prediction[0]['boxes'].cpu().numpy()
                sc = prediction[0]['scores'].cpu().numpy()

                im2 = 0.9*np.copy(im)
                fig = plt.figure(figsize=(12,6))
                for i in range(bb.shape[0]):
                    x0, y0, x1, y1 = np.round(bb[i,:]).astype(int)
                    x1 = np.minimum(x1, im2.shape[1]-1)
                    y1 = np.minimum(y1, im2.shape[0]-1)
                    if sc[i] > 0.5:
                        im2[y0:y1,x0] = 1
                        im2[y0:y1,x1] = 1
                        im2[y0,x0:x1] = 1
                        im2[y1,x0:x1] = 1
                plt.subplot(1,2,1)
                plt.imshow(im2,cmap='gray')
                plt.axis('off')
                plt.subplot(1,2,2)
                plt.imshow(p_max)
                plt.axis('off')
                plt.show()

    if mode == 'deploy':
        model.load_state_dict(torch.load(model_path))
        model.eval()
        with torch.no_grad():
            model.to(device_train)

            dataset_deploy = CellsDataset(deploy_path_in, get_transform(train=False), False)

            for img_index in range(len(dataset_deploy)):
                file_path = dataset_deploy.imgs[img_index]
                _, file_name, _ = fileparts(file_path)
                print('processing image', file_name)

                img1, _ = dataset_deploy[img_index]
                img= img1[:,0:256,0:256]

                prediction = model([img.to(device_train)])

                im = np.mean(img.numpy(),axis=0)

                p = prediction[0]['masks'][:, 0].cpu().numpy()
                p_max = np.max(p,axis=0)

                bb = prediction[0]['boxes'].cpu().numpy()
                sc = prediction[0]['scores'].cpu().numpy()
                im2 = 0 * np.copy(im)
                for i in range(bb.shape[0]):
                    if sc[i] > 0.3:
                        mask = morphology.remove_small_holes(morphology.remove_small_objects(p[i,:,:]>0.5,10), 100)
                        im2[mask==1] = i
                imwrite(im2, deploy_path_out+'/'+file_name+'_label.tif')
                # imwrite(p,deploy_path_out+'/'+file_name+'_labelStack.tif')

                im2 = 0.9*np.copy(im)
                for i in range(bb.shape[0]):
                    x0, y0, x1, y1 = np.round(bb[i,:]).astype(int)
                    x1 = np.minimum(x1, im2.shape[1]-1)
                    y1 = np.minimum(y1, im2.shape[0]-1)
                    if sc[i] > 0.3:
                        im2[y0:y1,x0] = 1
                        im2[y0:y1,x1] = 1
                        im2[y0,x0:x1] = 1
                        im2[y1,x0:x1] = 1

                imwrite(np.uint8(255*im2), deploy_path_out+'/'+file_name+'_bb.png')
                # imwrite(np.uint8(255*p_max), deploy_path_out+'/'+file_name+'_pm.png')