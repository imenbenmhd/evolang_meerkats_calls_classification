jman -vv submit -q sgpu -n wavlmfeats /idiap/temp/esarkar/miniconda/envs/s3prl/bin/python s3prl_featureExtractor_Imen.py -u wavlm_base --l -1 -w wav_list_mara.list --ids ids_mara.list -n wavlm-mara -p /idiap/temp/ibmahmoud/s3prl/ --device cuda --npickle 10 -f wavlm-base-mara --cache_dir /idiap/temp/ibmahmoud/s3prl/CacheDir/

