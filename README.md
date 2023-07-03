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
├── figs            # useful figures
├── LICENSES        # licenses of modules used
├── meerkats        # 
├── scripts         # main scripts to submit and notebooks
└── SSL             # self-supervised learning scripts
```

## Usage

Create and activate conda environment:

```
conda env create -f environment.yml -n meerkats # Create env
conda activate meerkats                         # Activate env
```
### CNN Approach
Run the script to use the end-to-end CNN approach with pytorch-lightening:

```
python scripts/train_lit.py -dir info_file.csv -s sampling_rate -b batch_size -lr learning_rate # Run CNN classification script 
```
Structure of info_file.csv must be:
	
	|path	       |labels|other_columns|..|
	|segment_1.wav |al    |value1	    |..|
	|segment_2.wav |gr    |value2       |..|
        |...	       |..    |..           |..|


A class_to_index.json file should be created for your data to correspond a label to an index, for example:

```json
{ "alarm":	0,
  "grooming":	1,
  "sunning":	2
 }
```

### RF or SVM on feature set:

```
python scripts/train_svm.py -m rf_or_svm -p path_to_feature.csv -n name_of_features -d exp_folder
```

To have more information about the arguments, run:
```
python scripts/train_svm.py --help
```

## Contact:

If you have a question, or an issue, please contact the [author](mailto:imen.benmhd@gmail.com)


