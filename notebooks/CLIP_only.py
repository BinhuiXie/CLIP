import os.path

import numpy as np
import torch
import clip
from tqdm import tqdm
from pkg_resources import packaging
from torchvision.datasets import ImageFolder

from collections import defaultdict, OrderedDict
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

print("Torch version:", torch.__version__)

import clip

datasets = {
    'domainnet':
        {
            'classes': ['The_Eiffel_Tower', 'The_Great_Wall_of_China', 'The_Mona_Lisa', 'aircraft_carrier', 'airplane',
                        'alarm_clock', 'ambulance', 'angel', 'animal_migration', 'ant', 'anvil', 'apple', 'arm',
                        'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball_bat',
                        'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt',
                        'bench', 'bicycle', 'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 'book',
                        'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom',
                        'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar',
                        'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot',
                        'castle', 'cat', 'ceiling_fan', 'cell_phone', 'cello', 'chair', 'chandelier', 'church',
                        'circle', 'clarinet', 'clock', 'cloud', 'coffee_cup', 'compass', 'computer', 'cookie', 'cooler',
                        'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise_ship', 'cup', 'diamond',
                        'dishwasher', 'diving_board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill',
                        'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye',
                        'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 'fire_hydrant', 'fireplace',
                        'firetruck', 'fish', 'flamingo', 'flashlight', 'flip_flops', 'floor_lamp', 'flower',
                        'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan', 'garden', 'garden_hose', 'giraffe',
                        'goatee', 'golf_club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp',
                        'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey_puck',
                        'hockey_stick', 'horse', 'hospital', 'hot_air_balloon', 'hot_dog', 'hot_tub', 'hourglass',
                        'house', 'house_plant', 'hurricane', 'ice_cream', 'jacket', 'jail', 'kangaroo', 'key',
                        'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'light_bulb',
                        'lighter', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop',
                        'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave',
                        'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug',
                        'mushroom', 'nail', 'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl',
                        'paint_can', 'paintbrush', 'palm_tree', 'panda', 'pants', 'paper_clip', 'parachute', 'parrot',
                        'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup_truck',
                        'picture_frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police_car', 'pond', 'pool',
                        'popsicle', 'postcard', 'potato', 'power_outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain',
                        'rainbow', 'rake', 'remote_control', 'rhinoceros', 'rifle', 'river', 'roller_coaster',
                        'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school_bus', 'scissors',
                        'scorpion', 'screwdriver', 'sea_turtle', 'see_saw', 'shark', 'sheep', 'shoe', 'shorts',
                        'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping_bag', 'smiley_face', 'snail',
                        'snake', 'snorkel', 'snowflake', 'snowman', 'soccer_ball', 'sock', 'speedboat', 'spider',
                        'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo',
                        'stethoscope', 'stitches', 'stop_sign', 'stove', 'strawberry', 'streetlight', 'string_bean',
                        'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword', 'syringe', 't-shirt',
                        'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis_racquet', 'tent', 'tiger',
                        'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 'tractor',
                        'traffic_light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 'umbrella',
                        'underwear', 'van', 'vase', 'violin', 'washing_machine', 'watermelon', 'waterslide', 'whale',
                        'wheel', 'windmill', 'wine_bottle', 'wine_glass', 'wristwatch', 'yoga', 'zebra', 'zigzag'],
            'domains': ["painting", "real", "sketch", "clipart", "infograph", "quickdraw", ],
            'split': [150, 50, 145]

        },
    'visda':  # 6, 3, 3
        {
            'classes': ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person', 'plant',
                        'skateboard', 'train', 'truck'],
            'domains': ['validation'],
            'split': [6, 3, 3]
        },
    'office31':  # 10, 11, 10
        {
            'classes': ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair',
                        'desk_lamp', 'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer',
                        'letter_tray', 'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone',
                        'printer', 'projector', 'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler',
                        'tape_dispenser', 'trash_can'],
            'domains': ["amazon", "webcam", "dslr"],
            'split': [10, 11, 10]
        },
    'home':  # 10, 5, 50
        {
            'classes': ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator',
                        'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp',
                        'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork',
                        'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker',
                        'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil',
                        'Postit_Notes', 'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors',
                        'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', 'Spoon', 'TV', 'Table',
                        'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'Webcam'],
            'domains': ["Art", "Clipart", "Product", "RealWorld"],
            'split': [10, 5, 50]
        },
}

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


# imagenet_templates = [
#     'a photo of a {}.'
# ]


def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


for dataset_name in datasets:  # all domains
    print(f"{len(datasets[dataset_name]['classes'])} classes, {len(imagenet_templates)} templates")
    dataset_classes = datasets[dataset_name]['classes']
    for domain_name in datasets[dataset_name]['domains']:
        domain_path = os.path.join('../data', dataset_name, domain_name)

        for model_name in clip.available_models():
            model, preprocess = clip.load(model_name)

            if dataset_name == 'domainnet':
                images = ImageList(open(os.path.join('../data', dataset_name, domain_name + '_test.txt')).readlines(),
                                   transform=preprocess)
            else:
                images = ImageFolder(domain_path, transform=preprocess)

            loader = torch.utils.data.DataLoader(images, batch_size=32, num_workers=2)

            zeroshot_weights = zeroshot_classifier(dataset_classes, imagenet_templates, model)

            with torch.no_grad():
                top1, top5, n = 0., 0., 0.
                total_pred, total_gt = None, None
                for i, (images, target) in enumerate(tqdm(loader)):
                    images = images.cuda()
                    target = target.cuda()

                    # predict
                    image_features = model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    logits = 100. * image_features @ zeroshot_weights

                    # measure accuracy
                    acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                    top1 += acc1
                    top5 += acc5
                    n += images.size(0)

                    if total_gt is None:
                        total_gt = target.cpu()
                        total_pred = logits.max(1)[1].cpu()
                    else:
                        total_gt = torch.cat((total_gt, target.cpu()))
                        total_pred = torch.cat((total_pred, logits.max(1)[1].cpu()))

            per_class_acc = [0. for _ in range(len(dataset_classes))]
            target_class_acc = [0. for _ in
                                range(datasets[dataset_name]['split'][0] + datasets[dataset_name]['split'][2])]
            print('-' * 30, dataset_name, '\t', domain_name, '\t', model_name, '-' * 30)
            for cls in range(len(dataset_classes)):
                if (total_gt == cls).sum():
                    per_class_acc[cls] = ((total_pred == cls) & (total_pred == total_gt)).sum() / (
                                total_gt == cls).sum() * 100.
                    print('class:', dataset_classes[cls], f'acc:{per_class_acc[cls]:.2f}')
                else:
                    print('class:', dataset_classes[cls], 'does not exist')

                if not (datasets[dataset_name]['split'][0] <= cls < datasets[dataset_name]['split'][0] +
                        datasets[dataset_name]['split'][1]):
                    index = cls if cls < datasets[dataset_name]['split'][0] else cls - datasets[dataset_name]['split'][
                        1]
                    if (total_gt == cls).sum():
                        target_class_acc[index] = ((total_pred == cls) & (total_pred == total_gt)).sum() / (
                                    total_gt == cls).sum() * 100.
                    else:
                        print('class:', dataset_classes[cls], 'target private class does not exist')

            print('mean class acc:', np.mean(per_class_acc))

            print('target mean class acc:', np.mean(target_class_acc))

            top1 = (top1 / n) * 100
            top5 = (top5 / n) * 100

            print(f"Top-1 accuracy: {top1:.2f}")
            print(f"Top-5 accuracy: {top5:.2f}")
            print('-' * 100)
