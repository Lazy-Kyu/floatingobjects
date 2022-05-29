python code/main.py train --results-dir /data/floatingobjects/models/prototypevitp64 --model prototypevit --data-path /data/floatingobjects/data --weight-decay 0.01 --tensorboard /data/floatingobjects/tensorboard/prototypevitp64 --device cuda --epochs 100 --workers 32 --cache-to-numpy --batch-size 24 --augmentation-intensity 1

# didnt work very well...
#python code/main.py train --results-dir /data/floatingobjects/models/unet_with_negoutlierloss --model unet --data-path /data/floatingobjects/data --tensorboard /data/floatingobjects/tensorboard/unet_with_negoutlierloss \
#  --device cuda --workers 32 --cache-to-numpy --batch-size 256 --neg_outlier_loss_border 19 --neg_outlier_loss_num_pixel 100 --neg_outlier_loss_penalty_factor 3

#python code/main.py train --results-dir /data/floatingobjects/models/fcnresnet --model fcnresnet --data-path /data/floatingobjects/flobsdata/data --tensorboard /data/floatingobjects/tensorboard/fcnresnet --device cuda --workers 32 --cache-to-numpy --batch-size 256 --ignore_border_from_loss_kernelsize 0
#python code/main.py train --results-dir /data/floatingobjects/models/uresnet_sametorchversions --model uresnet --data-path /data/floatingobjects/flobsdata/data --tensorboard /data/floatingobjects/tensorboard/uresnet_sametorchversions --device cuda --workers 32 --cache-to-numpy --batch-size 256 --ignore_border_from_loss_kernelsize 0

# python code/main.py train --results-dir /data/floatingobjects/models/resnetunet --model resnetunet --data-path /data/floatingobjects/data --tensorboard /data/floatingobjects/tensorboard/resnetunet --device cuda --workers 32 --cache-to-numpy --batch-size 256 --ignore_border_from_loss_kernelsize 0
