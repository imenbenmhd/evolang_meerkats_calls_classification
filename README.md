# Meerkats Call-Type Classification
The goal of this project is to see if there is a difference of structure between the calls that meerkats make in the differents situations that they face in the wildlife : Alarm, sunning, grooming etc

Project start : Febuary 2022
Project end : July 2023

The structure of this directory is organized as the following:

```
├── environment.yml # File to re-create conda environment 
├── __init__.py     # Init file
├── data            # contains the wavs path of the data
├── env.sh          # to set the python path
├── lib             # gitsubmodules used
├── README.md       # This file
├── experiments     # result of every experiment with the model checkpoint
│   ├── data        # Dataloader files and any other mapping
│   ├── models      # Model architectures              
│   └── utils       # Training scripts for (mostly for regular PyTorch)
├── figs            # useful figures
├── LICENSES        # licenses of modules used
├── meerkats        # 
├── scripts         # main scripts to submit and notebooks
└── SSL             # self-supervised learning scripts
```

## Usage
```
conda env create -f environment.yml -n meerkats # Create env
conda activate meerkats                         # Activate env
python                             # Run training script
```


