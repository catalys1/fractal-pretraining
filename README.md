<div align="center">

# Improving Fractal Pre-training

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
[![Paper](http://img.shields.io/badge/paper-arxiv.2110.03091-B31B1B.svg)](https://arxiv.org/abs/2110.03091)
[![Conference](http://img.shields.io/badge/WACV-2022-4b44ce.svg)](https://wacv2022.thecvf.com/home)

</div>

This is the official PyTorch code for Improving Fractal Pre-training ([arXiv](https://arxiv.org/abs/2110.03091)).

```
@article{anderson2021fractal,
  author  = {Connor Anderson and Ryan Farrell},
  title   = {Improving Fractal Pre-training},
  journal = {arXiv preprint arXiv:2110.03091},
  year    = {2021},
}
```

<div style="color: red">This README is incomplete (work in progress).</div>

### Setup

The code uses [PyTorch-Lightning](https://www.pytorchlightning.ai/) for training and [Hydra](https://hydra.cc/) for configuration. Other required packages are listed in `install_requirements.sh`.
```yaml
# clone project
git clone https://github.com/catalys1/fractal-pretraining.git
cd fractal-pretraining

# [RECOMMENDED] setup a virtual environment
python3 -m venv venv
source venv/bin/activate

# install requirements
bash install_requirements.sh
# install project in editable mode
pip install -e fractal_learning
```

<br>
