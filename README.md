# Cloud:cloud:
This is an implementation of the following paper:
Masatoshi, Kobayashi., Kohei, Miyaguchi., Shin, Matsushima. [Detection of Unobserved Common Cause in Discrete Data Based on the MDL Principle]()


## Requirement
- Python 3.8+
- `numpy`
- `sklearn`

## Install
Clone this repository, then run
```
$ python setup.py install
```
## Usage & Demo
Here is a simple example to use Cloud:

```
import numpy as np
import pandas as pd
from src import Cloud

# generate data from X causes Y
x = np.random.randint(0, 5, 1000) # 5 cyclic
y = (x + np.random.randint(0, 8, 1000)) % 8 # 8 cyclic
                                            # Of course, you do not need modulo operation

# pass the data to Cloud
result = Cloud(x, y,
    n_model_candiates = 4, # select a set of model candidates
    is_print=True # print out inferred causal direction 
)
```

You should see output like this:

```
Cloud Inference Result:: X ⫫ Y   Δ=3.85
```
`result` is like this:
```
[(5355.004242943733, 'indep'),
 (5358.854384890658, 'to'),
 (5363.219221043212, 'gets'),
 (5412.560143294632, 'confounder')]
```

## Licence
[MIT](https://github.com/Matsushima-lab/Cloud/blob/main/LICENSE)

## Reference
[tcnksm](https://github.com/tcnksm)
