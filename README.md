# ISFPN
```
Pytorch Implementation of "ISFPN: A medical detection model for ischemic stroke based on Feature Pyramid Network and Swin Transformer
You can star this repository to keep track of the project if it's helpful for you, thank you for your support.
```

# Environment
```
OS: Windows 10
Python: python3.7 with torch==1.2.0, torchvision==0.4.0
```

# Trained models
```
You could get the trained models reported above at 
https://github.com/Thanostitan1022/MyModel_2
```


#### Train
```
usage: train.py [-h] --datasetname DATASETNAME]
                [--checkpointspath CHECKPOINTSPATH]
optional arguments:
  -h, --help            show this help message and exit
  --datasetname DATASETNAME
                        dataset for training.
  --checkpointspath CHECKPOINTSPATH
                        checkpoints you want to use.
                        
parser.add_argument("--start_epoch", type=int, default=0, help="epoch to start training from last time")
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=80, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0003, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=3, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=64, help="size of image height")
parser.add_argument("--img_width", type=int, default=64, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in model")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
parser.add_argument('--model', type=str, default='', help='model checkpoint file')                        


cmd example:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --datasetname ISLES
```
#### Test
```
usage: test.py [-h] --datasetname DATASETNAME [--annfilepath ANNFILEPATH]
               [--datasettype DATASETTYPE]
               --checkpointspath CHECKPOINTSPATH [--nmsthresh NMSTHRESH]
optional arguments:
  -h, --help            show this help message and exit
  --datasetname DATASETNAME
                        dataset for testing.
  --annfilepath ANNFILEPATH
                        used to specify annfilepath.
  --datasettype DATASETTYPE
                        used to specify datasettype.
  --checkpointspath CHECKPOINTSPATH
                        checkpoints you want to use.
  --nmsthresh NMSTHRESH
                        thresh used in nms.
cmd example:
CUDA_VISIBLE_DEVICES=0 python test.py --checkpointspath fpn_res50_trainbackup/mymodel_20.pth --datasetname ISLES
```


#### Demo
```
import torch

# Model
model = torch.load("save", "mymodel")

# Images
img = "Dataset/images/test_iamges/0001.jpg"

# Inference
results = model(img)

# Results
results.print()
```
