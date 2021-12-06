# easygans
GANs implemented using keras

## Implementation

- [__PIX2PIX__](https://arxiv.org/abs/1611.07004)
```python
from easygans.pix2pix import *

pix2pix = Pix2Pix(dataset_path='data_path', img_rows=256, img_cols=256, gf=32, df=32)
pix2pix.train(epochs=100, batch_size=100)
predictor = Predictor(model='saved_model.h5', imgpath='working/image.jpg')
predictor.predict(img_res=(256, 256))
```

## Roadmap

- [__GAN__]
- [__PIX2PIX__](https://arxiv.org/abs/1611.07004)
- [__SRGAN__](https://arxiv.org/abs/1609.04802)
- [__DCGAN__]
- [__DiscoGAN__]
- [__CycleGAN__]
- [__InfoGAN__]
