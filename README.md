# Cloud:cloud:
This is an implementation of the following paper:  
[1] Masatoshi, Kobayashi., Kohei, Miyaguchi., Shin, Matsushima. (2022) [Detection of Unobserved Common Cause in Discrete Data Based on the MDL Principle]()


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
x = np.random.randint(0, 5, 10000) # 5 cyclic
y = (x + np.random.randint(0, 8, 10000)) % 8 # 8 cyclic
                                             # modulo operation is optional

# pass the data to Cloud
result = Cloud(
    X=x, 
    Y=y,
    n_model_candiates=4, # select a set of model candidates
    is_print=True # print out inferred causal direction 
)
```

You should see output like this:

```
Cloud Inference Result:: X ⇒ Y    Δ=2.13
```

`result` is like this:

```
[(53272.67451834934, 'to'),
 (53274.80253272231, 'indep'),
 (53277.3410945044, 'gets'),
 (53365.39411832997, 'confounder')]
```

## Licence
[MIT](https://github.com/Matsushima-lab/Cloud/blob/main/LICENSE)

## Reference
[kbudhath](https://github.molgen.mpg.de/EDA/cisc)

Our implementation is greatly inspired by his.
