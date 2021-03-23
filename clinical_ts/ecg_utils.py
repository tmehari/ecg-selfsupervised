__all__ = ['get_available_channels', 'channel_stoi_default', 'resample_data', 'get_filename_out', 'prepare_data_ptb_xl',
           'filter_ptb_xl','prepare_data_cinc', 'prepare_data_zheng', 'prepare_data_ribeiro_test']

# Cell
import wfdb

import scipy.io

import numpy as np
import pandas as pd

from skimage import transform
from scipy.ndimage import zoom
from tqdm.auto import tqdm
from pathlib import Path

from .stratify import stratify# ,stratify_batched

#ribeiro
import h5py
import datetime

from .timeseries_utils import *

channel_stoi_default = {"i": 0, "ii": 1, "v1":2, "v2":3, "v3":4, "v4":5, "v5":6, "v6":7, "iii":8, "avr":9, "avl":10, "avf":11, "vx":12, "vy":13, "vz":14}

def get_available_channels(channel_labels, channel_stoi):
    if(channel_stoi is None):
        return range(len(channel_labels))
    else:
        return sorted([channel_stoi[c] for c in channel_labels if c in channel_stoi.keys()])

def resample_data(sigbufs, channel_labels, fs, target_fs, channels=8, channel_stoi=None,skimage_transform=True,interpolation_order=3):
    channel_labels = [c.lower() for c in channel_labels]
    #https://github.com/scipy/scipy/issues/7324 zoom issues
    factor = target_fs/fs
    timesteps_new = int(len(sigbufs)*factor)
    if(channel_stoi is not None):
        data = np.zeros((timesteps_new, channels), dtype=np.float32)
        for i,cl in enumerate(channel_labels):
            if(cl in channel_stoi.keys() and channel_stoi[cl]<channels):
                if(skimage_transform):
                    data[:,channel_stoi[cl]]=transform.resize(sigbufs[:,i],(timesteps_new,),order=interpolation_order).astype(np.float32)
                else:
                    data[:,channel_stoi[cl]]=zoom(sigbufs[:,i],timesteps_new/len(sigbufs),order=interpolation_order).astype(np.float32)
    else:
        if(skimage_transform):
            data=transform.resize(sigbufs,(timesteps_new,channels),order=interpolation_order).astype(np.float32)
        else:
            data=zoom(sigbufs,(timesteps_new/len(sigbufs),1),order=interpolation_order).astype(np.float32)
    return data

def get_filename_out(filename_in, target_folder=None, suffix=""):
    if target_folder is None:
        #absolute path here
        filename_out = filename_in.parent/(filename_in.stem+suffix+".npy")
        filename_out_relative = filename_out
    else:
        if("train" in filename_in.parts):
            target_folder_train = target_folder/"train"
            # relative path here
            filename_out = target_folder_train/(filename_in.stem+suffix+".npy")
            filename_out_relative = filename_out.relative_to(target_folder)

            target_folder_train.mkdir(parents=True, exist_ok=True)
        elif("eval" in filename_in.parts or "dev_test" in filename_in.parts or "valid" in filename_in.parts or "valtest" in filename_in.parts):
            target_folder_valid = target_folder/"valid"
            filename_out = target_folder_valid/(filename_in.stem+suffix+".npy")
            filename_out_relative = filename_out.relative_to(target_folder)
            target_folder_valid.mkdir(parents=True, exist_ok=True)
        else:
            filename_out = target_folder/(filename_in.stem+suffix+".npy")
            filename_out_relative = filename_out.relative_to(target_folder)
            target_folder.mkdir(parents=True, exist_ok=True)
    return filename_out, filename_out_relative

def prepare_data_ptb_xl(data_path, min_cnt=50, target_fs=100, channels=8, channel_stoi=channel_stoi_default, target_folder=None, skimage_transform=True, recreate_data=True):
    target_root_ptb_xl = Path(".") if target_folder is None else target_folder
    #print(target_root_ptb_xl)
    target_root_ptb_xl.mkdir(parents=True, exist_ok=True)

    if(recreate_data is True):
        # reading df
        ptb_xl_csv = data_path/"ptbxl_database.csv"
        df_ptb_xl=pd.read_csv(ptb_xl_csv,index_col="ecg_id")
        #print(df_ptb_xl.columns)
        df_ptb_xl.scp_codes=df_ptb_xl.scp_codes.apply(lambda x: eval(x.replace("nan","np.nan")))

        # preparing labels
        ptb_xl_label_df = pd.read_csv(data_path/"scp_statements.csv")
        ptb_xl_label_df=ptb_xl_label_df.set_index(ptb_xl_label_df.columns[0])

        ptb_xl_label_diag= ptb_xl_label_df[ptb_xl_label_df.diagnostic >0]
        ptb_xl_label_form= ptb_xl_label_df[ptb_xl_label_df.form >0]
        ptb_xl_label_rhythm= ptb_xl_label_df[ptb_xl_label_df.rhythm >0]

        diag_class_mapping={}
        diag_subclass_mapping={}
        for id,row in ptb_xl_label_diag.iterrows():
            if(isinstance(row["diagnostic_class"],str)):
                diag_class_mapping[id]=row["diagnostic_class"]
            if(isinstance(row["diagnostic_subclass"],str)):
                diag_subclass_mapping[id]=row["diagnostic_subclass"]

        df_ptb_xl["label_all"]= df_ptb_xl.scp_codes.apply(lambda x: [y for y in x.keys()])
        df_ptb_xl["label_diag"]= df_ptb_xl.scp_codes.apply(lambda x: [y for y in x.keys() if y in ptb_xl_label_diag.index])
        df_ptb_xl["label_form"]= df_ptb_xl.scp_codes.apply(lambda x: [y for y in x.keys() if y in ptb_xl_label_form.index])
        df_ptb_xl["label_rhythm"]= df_ptb_xl.scp_codes.apply(lambda x: [y for y in x.keys() if y in ptb_xl_label_rhythm.index])

        df_ptb_xl["label_diag_subclass"]= df_ptb_xl.label_diag.apply(lambda x: [diag_subclass_mapping[y] for y in x if y in diag_subclass_mapping])
        df_ptb_xl["label_diag_superclass"]= df_ptb_xl.label_diag.apply(lambda x: [diag_class_mapping[y] for y in x if y in diag_class_mapping])

        df_ptb_xl["dataset"]="ptb_xl"
        #filter (can be reapplied at any time)
        df_ptb_xl, lbl_itos_ptb_xl =filter_ptb_xl(df_ptb_xl,min_cnt=min_cnt)

        filenames = []
        for id, row in tqdm(list(df_ptb_xl.iterrows())):
            filename = data_path/row["filename_lr"] if target_fs<=100 else data_path/row["filename_hr"]
            sigbufs, header = wfdb.rdsamp(str(filename))
            data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=header['sig_name'],fs=header['fs'],target_fs=target_fs,channels=channels,skimage_transform=skimage_transform)
            assert(target_fs<=header['fs'])
            np.save(target_root_ptb_xl/(filename.stem+".npy"),data)
            filenames.append(Path(filename.stem+".npy"))
        df_ptb_xl["data"] = filenames

        #add means and std
        dataset_add_mean_col(df_ptb_xl,data_folder=target_root_ptb_xl)
        dataset_add_std_col(df_ptb_xl,data_folder=target_root_ptb_xl)
        dataset_add_length_col(df_ptb_xl,data_folder=target_root_ptb_xl)
        #dataset_add_median_col(df_ptb_xl,data_folder=target_root_ptb_xl)
        #dataset_add_iqr_col(df_ptb_xl,data_folder=target_root_ptb_xl)

        #save means and stds
        mean_ptb_xl, std_ptb_xl = dataset_get_stats(df_ptb_xl)

        #save
        save_dataset(df_ptb_xl,lbl_itos_ptb_xl,mean_ptb_xl,std_ptb_xl,target_root_ptb_xl)
    else:
        df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl = load_dataset(target_root_ptb_xl,df_mapped=False)
    return df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl

def filter_ptb_xl(df,min_cnt=10,categories=["label_all","label_diag","label_form","label_rhythm","label_diag_subclass","label_diag_superclass"]):
    #filter labels
    def select_labels(labels, min_cnt=10):
        lbl, cnt = np.unique([item for sublist in list(labels) for item in sublist], return_counts=True)
        return list(lbl[np.where(cnt>=min_cnt)[0]])
    df_ptb_xl = df.copy()
    lbl_itos_ptb_xl = {}
    for selection in categories:
        label_selected = select_labels(df_ptb_xl[selection],min_cnt=min_cnt)
        df_ptb_xl[selection+"_filtered"]=df_ptb_xl[selection].apply(lambda x:[y for y in x if y in label_selected])
        lbl_itos_ptb_xl[selection] = np.array(list(set([x for sublist in df_ptb_xl[selection+"_filtered"] for x in sublist])))
        lbl_stoi = {s:i for i,s in enumerate(lbl_itos_ptb_xl[selection])}
        df_ptb_xl[selection+"_filtered_numeric"]=df_ptb_xl[selection+"_filtered"].apply(lambda x:[lbl_stoi[y] for y in x])
    return df_ptb_xl, lbl_itos_ptb_xl



def prepare_data_cinc(data_path, datasets=["ICBEB2018","ICBEB2018_2","INCART","PTB","PTB-XL","Georgia"], target_fs=100, strat_folds=10, channels=8, channel_stoi=channel_stoi_default, target_folder=None, skimage_transform=True, recreate_data=True):
    '''unzip archives into separate folders with dataset names from above'''
    target_root = Path(".") if target_folder is None else target_folder
    target_root.mkdir(parents=True, exist_ok=True)

    if(recreate_data is True):
        dx_meta = pd.concat([pd.read_csv(data_path/"dx_mapping_scored.csv"),pd.read_csv(data_path/"dx_mapping_unscored.csv")],sort=True)
        dx_mapping_snomed_abbrev = {a:b for [a,b] in list(dx_meta.apply(lambda row: [row["SNOMED CT Code"],row["Abbreviation"]],axis=1))}

        metadata = []
        for filename in tqdm(list(data_path.glob('**/*.hea'))):
            if(not(filename.parts[-2] in datasets)):
                continue
            sigbufs, header = wfdb.rdsamp(str(filename)[:-4])
            #print(filename,sigbufs.shape,np.min(sigbufs,axis=0),np.any(np.isnan(sigbufs)))
            if(np.any(np.isnan(sigbufs))):
                print("Warning:",str(filename),"is corrupt. Skipping.")
                continue
            data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=header['sig_name'],fs=header['fs'],target_fs=target_fs,channels=channels,skimage_transform=skimage_transform)
            assert(target_fs<=header['fs'])
            np.save(target_root/(filename.stem+".npy"),data)
            labels=[]
            age=np.nan
            sex="nan"
            for l in header["comments"]:
                arrs = l.strip().split(' ')
                if l.startswith('Dx:'):
                    labels = [dx_mapping_snomed_abbrev[int(x)] for x in arrs[1].split(',')]
                elif l.startswith('Age:'):
                    try:
                        age = int(arrs[1])
                    except:
                        age= np.nan
                elif l.startswith('Sex:'):
                    sex = arrs[1].strip().lower()
                    if(sex=="m"):
                        sex="male"
                    elif(sex=="f"):
                        sex="female"

            metadata.append({"data":Path(filename.stem+".npy"),"label":labels,"sex":sex,"age":age,"dataset":"cinc_"+filename.parts[-2]})
        df =pd.DataFrame(metadata)
        lbl_itos = np.unique([item for sublist in list(df.label) for item in sublist])
        lbl_stoi = {s:i for i,s in enumerate(lbl_itos)}
        df["label"] = df["label"].apply(lambda x: [lbl_stoi[y] for y in x])

        #does not incorporate patient-level split
        df["strat_fold"]=-1
        for ds in np.unique(df["dataset"]):
            print("Creating CV folds:",ds)
            dfx = df[df.dataset==ds]
            idxs = np.array(dfx.index.values)
            lbl_itosx = np.unique([item for sublist in list(dfx.label) for item in sublist])
            stratified_ids = stratify(list(dfx["label"]), lbl_itosx, [1./strat_folds]*strat_folds)

            for i,split in enumerate(stratified_ids):
                df.loc[idxs[split],"strat_fold"]=i

        #add means and std
        dataset_add_mean_col(df,data_folder=target_root)
        dataset_add_std_col(df,data_folder=target_root)
        dataset_add_length_col(df,data_folder=target_root)

        #save means and stds
        mean, std = dataset_get_stats(df)

        #save
        save_dataset(df, lbl_itos, mean, std, target_root)
    else:
        df, lbl_itos, mean, std = load_dataset(target_root,df_mapped=False)
    return df, lbl_itos, mean, std

def prepare_data_zheng(data_path, denoised=False, target_fs=100, strat_folds=10, channels=8, channel_stoi=channel_stoi_default, target_folder=None, skimage_transform=True, recreate_data=True):
    '''prepares the Zheng et al 2020 dataset'''
    target_root = Path(".") if target_folder is None else target_folder
    target_root.mkdir(parents=True, exist_ok=True)

    if(recreate_data is True):
        #df_attributes = pd.read_excel("./AttributesDictionary.xlsx")
        #df_conditions = pd.read_excel("./ConditionNames.xlsx")
        #df_rhythm = pd.read_excel("./RhythmNames.xlsx")
        df = pd.read_excel(data_path/"Diagnostics.xlsx")
        df["id"]=df.FileName
        df["data"]=df.FileName.apply(lambda x: x+".npy")
        df["label_condition_txt"]=df.Beat.apply(lambda x: [y for y in x.split(" ") if x!="NONE"])
        df["label_rhythm_txt"]=df.Rhythm.apply(lambda x: x.split(" "))
        df["label_txt"]=df.apply(lambda row: row["label_condition_txt"]+row["label_rhythm_txt"],axis=1)
        df["sex"]=df.Gender.apply(lambda x:x.lower())
        df["age"]=df.PatientAge
        df.drop(["Gender","PatientAge","Rhythm","Beat","FileName"],inplace=True,axis=1)

        #map to numerical indices
        lbl_itos={}
        lbl_stoi={}
        lbl_itos["all"] = np.unique([item for sublist in list(df.label_txt) for item in sublist])
        lbl_stoi["all"] = {s:i for i,s in enumerate(lbl_itos["all"])}
        df["label"] = df["label_txt"].apply(lambda x: [lbl_stoi["all"][y] for y in x])
        lbl_itos["condition"] = np.unique([item for sublist in list(df.label_condition_txt) for item in sublist])
        lbl_stoi["condition"] = {s:i for i,s in enumerate(lbl_itos["condition"])}
        df["label_condition"] = df["label_condition_txt"].apply(lambda x: [lbl_stoi["condition"][y] for y in x])
        lbl_itos["rhythm"] = np.unique([item for sublist in list(df.label_rhythm_txt) for item in sublist])
        lbl_stoi["rhythm"] = {s:i for i,s in enumerate(lbl_itos["rhythm"])}
        df["label_rhythm"] = df["label_rhythm_txt"].apply(lambda x: [lbl_stoi["rhythm"][y] for y in x])
        df["dataset"]="Zheng2020"

        for id,row in tqdm(list(df.iterrows())):
            fs = 500.

            df_tmp = pd.read_csv(data_path/("ECGDataDenoised" if denoised else "ECGData")/(row["id"]+".csv"))
            channel_labels = list(df_tmp.columns)
            sigbufs = np.array(df_tmp)*0.001 #assuming data is given in muV

            data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=channel_labels,fs=fs,target_fs=target_fs,channels=channels,skimage_transform=skimage_transform)
            assert(target_fs<=fs)
            np.save(target_root/(row["id"]+".npy"),data)

        stratified_ids = stratify(list(df["label_txt"]), lbl_itos["all"], [1./strat_folds]*strat_folds)
        df["strat_fold"]=-1
        idxs = np.array(df.index.values)
        for i,split in enumerate(stratified_ids):
            df.loc[idxs[split],"strat_fold"]=i

        #add means and std
        dataset_add_mean_col(df,data_folder=target_root)
        dataset_add_std_col(df,data_folder=target_root)
        dataset_add_length_col(df,data_folder=target_root)

        #save means and stds
        mean, std = dataset_get_stats(df)

        #save
        save_dataset(df, lbl_itos, mean, std, target_root)
    else:
        df, lbl_itos, mean, std = load_dataset(target_root,df_mapped=False)
    return df, lbl_itos, mean, std

def prepare_data_ribeiro_test(data_path, denoised=False, target_fs=100, strat_folds=10, channels=8, channel_stoi=channel_stoi_default, target_folder=None, skimage_transform=True, recreate_data=True):
    '''prepares test set of Ribeiro et al Nat Comm 2020'''
    data_path = Path(data_path)
    target_root = Path(".") if target_folder is None else target_folder
    target_root.mkdir(parents=True, exist_ok=True)

    if(recreate_data is True):
        lbl_itos = ["1AVB","RBBB","LBBB","SBRAD","AFIB","STACH"]
        channel_labels = ["i","ii","iii","avr","avl","avf","v1","v2","v3","v4","v5","v6"]
        fs= 400.
        #prepare df
        df_cardiologist1 = pd.read_csv(data_path/"annotations"/"cardiologist1.csv")
        df_cardiologist2 = pd.read_csv(data_path/"annotations"/"cardiologist2.csv")
        df_gold = pd.read_csv(data_path/"annotations"/"gold_standard.csv")
        df_cardiology_residents = pd.read_csv(data_path/"annotations"/"cardiology_residents.csv")
        df_dnn = pd.read_csv(data_path/"annotations"/"dnn.csv")
        df_emergency_residents = pd.read_csv(data_path/"annotations"/"emergency_residents.csv")
        df_medical_students = pd.read_csv(data_path/"annotations"/"medical_students.csv")
        df_attributes = pd.read_csv(data_path/"attributes.csv")
        sex_map = {"M":"male", "F":"female"}
        df_attributes.sex = df_attributes.sex.apply(lambda x: sex_map[x])

        def reformat_predictions(df, colname_txt="label_txt", colname_num="label", lbl_itos=["1AVB","RBBB","LBBB","SBRAD","AFIB","STACH"]):
            lbl_stoi = {s:i for i,s in enumerate(lbl_itos)}
            df[colname_txt]=df.apply(lambda row: ("1AVB " if row["1dAVb"] else "")+("RBBB " if row["RBBB"] else "")+("LBBB " if row["LBBB"] else "")+("SBRAD " if row["SB"] else "")+("AFIB " if row["AF"] else "")+("STACH " if row["ST"] else ""),axis=1)
            df[colname_txt]=df[colname_txt].apply(lambda x: [y for y in x.strip().split(" ") if y!=""])
            df[colname_num]=df[colname_txt].apply(lambda x: [lbl_stoi[y] for y in x if y in lbl_stoi.keys()])
            df.drop(["1dAVb","RBBB","LBBB","SB","AF","ST"],axis=1,inplace=True)

        reformat_predictions(df_cardiologist1,"label_cardiologist1_txt","label_cardiologist1")
        reformat_predictions(df_cardiologist2,"label_cardiologist2_txt","label_cardiologist2")
        reformat_predictions(df_gold,"label_txt","label")
        reformat_predictions(df_cardiology_residents,"label_cardiology_residents_txt","label_cardiology_residents")
        reformat_predictions(df_emergency_residents,"label_emergency_residents_txt","label_emergency_residents")
        reformat_predictions(df_medical_students,"label_medical_students_txt","label_medical_students")
        reformat_predictions(df_dnn,"label_dnn_txt","label_dnn")

        df=df_gold.join([df_cardiologist1,df_cardiologist2,df_cardiology_residents,df_emergency_residents,df_medical_students,df_dnn,df_attributes])
        df["data"]=[Path("Ribeiro_test_"+str(i)+".npy") for i in range(len(df))]
        df["dataset"]="Ribeiro_test"
        #prepare raw data
        with h5py.File(data_path/"ecg_tracings.hdf5", "r") as f:
            sigbufs = np.array(f['tracings'])
        start_idxs=[ np.where(np.sum(np.abs(sigbufs[i]),axis=1)==0.)[0] for i in range(len(sigbufs))] #discard zeros at beginning/end
        start_idxs = [len(a)//2 for a in start_idxs]


        for id,row in tqdm(list(df.iterrows())):
            data = resample_data(sigbufs=sigbufs[id][start_idxs[id]:-start_idxs[id] or None],channel_stoi=channel_stoi,channel_labels=channel_labels,fs=fs,target_fs=target_fs,channels=channels,skimage_transform=skimage_transform)
            assert(target_fs<=fs)
            np.save(target_root/(row["data"]),data)

        stratified_ids = stratify(list(df.apply(lambda row: row["label_txt"]+[row["sex"]],axis=1)), lbl_itos+["male","female"], [1./strat_folds]*strat_folds)
        df["strat_fold"]=-1
        idxs = np.array(df.index.values)
        for i,split in enumerate(stratified_ids):
            df.loc[idxs[split],"strat_fold"]=i

        #add means and std
        dataset_add_mean_col(df,data_folder=target_root)
        dataset_add_std_col(df,data_folder=target_root)
        dataset_add_length_col(df,data_folder=target_root)

        #save means and stds
        mean, std = dataset_get_stats(df)

        #save
        save_dataset(df, lbl_itos, mean, std, target_root)
    else:
        df, lbl_itos, mean, std = load_dataset(target_root,df_mapped=False)
    return df, lbl_itos, mean, std
