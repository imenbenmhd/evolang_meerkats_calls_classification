# Meerkats Call-Type Classification

The structure of this directory is organized as the following:

```
├── __init__.py     # Init file
├── README.md       # This file
├── src             # Source code
    ├── utils       # scripts
    └── features_extraction
    ├── models      # Model architectures              
    └── data        # dataloaders file and any other mapping 
  

## Notes:
- Regular PyTorch code can be left behind for PyTorch lightening code.
- The files under `src/models` can be used with regular PyTorch or through PyTorch Lightening thanks to the `lit.py` wrapper file. 

Example:

Regular PyTorch:
```python

from src.models.PalazCNN import palazcnn

model = palazcnn(n_input=n_input, n_output=num_classes)
```

PyTorch Lightning :zap: :
```python
from src.models.lit import Lit
from src.models.PalazCNN import palazcnn

model = Lit(palazcnn(n_input=n_input, n_output=num_classes))
```
