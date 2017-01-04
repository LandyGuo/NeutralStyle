##  NeutralStyle
This repository provides a simple Implementation of NeutralStyle.
See the original paper:[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576).
The implementation refers to [Anish Athalye's implementation](https://github.com/anishathalye/neural-style).

## Requirements
* tensorflow 0.11 or newer
* numpy
* scipy
* A well trained VGG19 Model,you can download it [here](https://pan.baidu.com/s/1bo2ojKV)

## Examples
![input.jpg](https://github.com/LandyGuo/NeutralStyle/blob/master/examples/input.jpg)
![style1.jpg](https://github.com/LandyGuo/NeutralStyle/blob/master/examples/style1.jpg)
![style1-out.jpg](https://github.com/LandyGuo/NeutralStyle/blob/master/examples/style1_out.jpg)


## How to realize
* Input content Image-->VGG19 `Relu_42` feature extraction--> content feature
* Input style Image--> VGG19 `Relu1_1`,`Relu2_1`,`Relu3_1`,`Relu4_1`,`Relu5_1`--> style feature
* generate Image with `content_loss` + `style_loss` + `variation_denoise_loss` 

##How to use
Modify your `INPUT_IMAGE_PATH` and `STYLE_IMAGE_PATH` in `NeuralStyle.py`, then Just Run It!
