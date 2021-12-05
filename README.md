# easygans
GANs implemented using keras

## Implementation

- __PIX2PIX__
```python
from easygans.pix2pix import *

pix2pix = Pix2Pix(dataset_path='data_path', img_rows=256, img_cols=256, gf=32, df=32)
pix2pix.train(epochs=100, batch_size=100)
predictor = Predictor(model='saved_model.h5', imgpath='working/image.jpg')
predictor.predict(img_res=(256, 256))
```

- More models will be added soon
