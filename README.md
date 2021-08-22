# PyTorch MultiGPU Template 

This project contains logging, model saving, resuming, ema, mix precision, some augmentations, etc.

The code is based on the structure from the official implement of [Improving Contrastive Learning by Visualizing Feature Transformation](https://github.com/DTennant/CL-Visualizing-Feature-Transformation)

### Prerequisites
- Python3
- NVIDIA GPUs + CUDA CuDNN
- PyTorch, torchvision, tensorboard_logger, etc. 
- (Optional) install [apex](https://github.com/NVIDIA/apex) if you would like to 
try mixed precision training. If you do not want to take a look at the 
[apex](https://github.com/NVIDIA/apex) repo, the installing commands are (assuming pytorch 
and CUDA are availabel):
    ```
    cd ~
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
	```
  
### prepartion
- **Dataset** update dataset/dataset.py and build_custom_loader in dataset/utils.py as your wish.
- **model** update build_backbone.py
- **trainer** update custom_trainer.py

### Training
Now it only supports **DistributedDataParallel** training (single-node multi-GPU or 
multi-node multi-GPU). It has predefined the configurations for several methods. 
For example, a training command with 2 GPUs:
```
CUDA_VISIBLE_DEVICES=0,1
python main.py \
  --method *** \
  ... \
  --cosine \
  --data_folder /path/to/data \
  --multiprocessing-distributed --world-size 1 --rank 0 \
```

(optional) If you want to use mixed precision training, please appending the following options:  
```
--amp --opt_level O1
```

(Optional) If you would like to use multi-node for training, the example command is:
```
# node 1
python main.py --method *** --batch_size 512 -j 40 --learning_rate 0.06 --multiprocessing-distributed --dist-url 'tcp://10.128.0.4:12345' --world-size 2 --rank 0
# node 2
python main.py --method *** --batch_size 512 -j 40 --learning_rate 0.06 --multiprocessing-distributed --dist-url 'tcp://10.128.0.4:12345' --world-size 2 --rank 1
```
where the `--batch_size` means global batch size and `-j` indicates number of workers on each node. 

### One more thing
This code is an experimental work and may contain errors. 
Feel free to raise an issue if you find one.
