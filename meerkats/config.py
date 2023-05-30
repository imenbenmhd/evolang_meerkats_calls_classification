# -*- coding: utf-8 -*-

import os 

configDir =os.path.abspath(os.path.dirname(__file__))

GITROOT=os.path.join(configDir + "/..")
#GITROOT="/idiap/project/evolang/meerkats_imen/evolang_meerkats_calls_classification"


DATADIR= GITROOT + "/data/"

DATAMARTA =DATADIR + "info_file_marta.csv"
DATAMARA = DATADIR + "info_file_mara.csv"
DATAISABEL=DATADIR + "info_file_isabel.csv"
DATASSL="/idiap/temp/ibmahmoud/s3prl/"
FEATURES_DIR= DATADIR + "features/"
