# Handwriting-number-recognition
a PyTorch implementation - handwriting  number recognition with GAN model using MNIST Dataset

## Project Goal 
The goal is to generate new handwritten digits using a generative adversarial network (GAN) trained on the MNIST dataset.
## GAN(Generative Adversarial Network)
GANs were first reported on in 2014 from Ian Goodfellow and others in Yoshua Bengio's lab. Since then, GANs have exploded in popularity. 
The idea behind GANs is using two networks, a generator  G  and a discriminator  D , competing against each other. The generator makes "fake" data to pass to the discriminator. The discriminator also sees real training data and predicts if the data it's received is real or fake.

The generator is trained to fool the discriminator, it wants to output data that looks as close as possible to real, training data.
The discriminator is a classifier that is trained to figure out which data is real and which is fake.
What ends up happening is that the generator learns to make data that is indistinguishable from real data to the discriminator.

## Reference 
* [2014 Generative Adversarial Network](https://arxiv.org/abs/1406.2661)
* [Pix2Pix]
* [CycleGAN & Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
* [A list of generative models](https://github.com/wiseodd/generative-models)
* [Image-to-Image Demo](https://affinelayer.com/pixsrv/)
