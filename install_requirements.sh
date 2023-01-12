# It's highly recommended to install these inside a virtual environment

pip install fgvcdata
pip install hydra-colorlog
pip install hydra-core>=1.1.1
pip install numba>=0.53.1
pip install numpy>=1.20.2
pip install omegaconf>=2.1.0
pip install opencv-python>=4.5.1.48
pip install Pillow>=8.1.0
pip install tqdm 
pip install python-dotenv

# Choose the appropriate version of pytorch for your system
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
pip install pytorch-lightning>=1.4.0 torchmetrics>=0.4.1 segmentation-models-pytorch
