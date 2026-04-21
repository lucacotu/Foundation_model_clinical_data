import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def clean_and_impute(dataset_name, df):
    match dataset_name:
        case "OrmoniTirodei":
            # Fix outlier as done in notebook
            df.loc[8143, "Documented resting \nor exertional ischemia"] = 1
            
            # Drop where Number is NaN
            df = df.dropna(subset=['Number'])

            # Date parsing
            df['Data prelievo'] = pd.to_datetime(df['Data prelievo'])
            df['Follow Up Data'] = pd.to_datetime(df['Follow Up Data'])
            
            # Fix mortality where death is known
            # violations = df_main[((df_main["Total mortality"] == 0) & (df_main["Data of death"].notna())) | ((df_main["Total mortality"] != 0) & (df_main["Data of death"].isna()))]
            df.loc[7285, "Total mortality"] = 1
            df.loc[7285, "UnKnown"] = 1

            # Calculate days
            df["Follow Up Data"] = (
                df["Follow Up Data"] - df["Data prelievo"]
            ).dt.days

            df["Data of death"] = (
                pd.to_datetime(df["Data of death"]) - df["Data prelievo"]
            ).dt.days
            df["Data of death"] = df["Data of death"].fillna(0)

            # Calculate target events
            target = ["CABG " ,"Non Fatal AMI (Follow-Up)","Ictus","PCI"]
            for t in target:
                df[t+"_event"] = df[t].notna().astype(int)
                df[t] = (
                    pd.to_datetime(df[t]) - df["Data prelievo"]
                ).dt.days
                df[t] = df[t].fillna(0)
            
            # Drop irrelevant
            df = df.drop(columns=["CardiopatiaCongenita"])
            
            # Drop all NaNs
            # We can avoid dropping all NaNs by using TabPFN's ability to handle them, 
            # but for simplicity we drop them here. 
            # We can always re-run the pipeline without dropping NaNs if we want to test TabPFN's robustness.
            #df_main = df_main.dropna()

            return df
        case "HURRAH":
            # Example preprocessing steps
            df.loc[:, "FU"] = np.floor(df["FU"]).astype(float)
            df = df.dropna(subset=["FU","STATO_AL_FU"])

            cols_max = ["FU_F_IMA", "FU_F_CBV", "FU_F_HF","FU"]

            df.loc[:, "FU"] = np.where(
                df["STATO_AL_FU"] == 1,
                df[cols_max].min(axis=1),
                df["FU"]
            )
            mask_sesso = df["SESSO"] == 1

            df.loc[mask_sesso, ["MENOPAUSA", "ANNI_MENOP"]] = -1

            mask = (df["SESSO"] == 0) & (df["MENOPAUSA"] == 0)

            df.loc[mask, "ANNI_MENOP"] = 0

            mask_meno = df["MENOPAUSA"] == 1

            mask_meno = df["MENOPAUSA"] == 1
            invalid_mask = mask_meno & (df["ANNI_MENOP"] <= 0)

            mean_anni = df[
                (df["MENOPAUSA"] == 1) & 
                (df["SESSO"] == 0) & 
                (df["ANNI_MENOP"] > 0)
            ]["ANNI_MENOP"].mean()

            df.loc[invalid_mask, "ANNI_MENOP"] = mean_anni

            df.drop(columns=["LVH","VES","SAPEVA","SESSO0"], inplace=True)

            has_any_measurement = (
                df['ALB_CREAT_MG_MMOL_3_4_34_MICRO'].notna() | 
                df['ALB_CREAT_MG_G_30_300_MICRO'].notna()
            )

            df['ALB_CAT'] = (
                (df['ALB_CREAT_MG_MMOL_3_4_34_MICRO'] > 3.4) | 
                (df['ALB_CREAT_MG_G_30_300_MICRO'] > 30)
            ).astype(float)

            df.loc[~has_any_measurement, 'ALB_CAT'] = float('nan')

            df = df.drop(columns=["ALB_CREAT_MG_MMOL_3_4_34_MICRO","ALB_CREAT_MG_G_30_300_MICRO","ALBURIA_MG_24H_30_300","ALBURIA_MG_DL"])
            
            return df



def prepare_cox_data_cv(df_main):
    """Prepares data for Cox mortality model with cross validation."""

    df_main['time'] = pd.qcut(df_main['Follow Up Data'], q=4, labels=False, duplicates='drop')

    # 2. Crea la variabile di stratificazione combinata
    #    Es: evento=1, bin=2 → stratum "1_2"
    df_main['stratum'] = df_main['Total mortality'].astype(str) + '_' + df_main['time'].astype(str)

    df_mortality_train, df_mortality_eval = train_test_split(
        df_main, test_size=0.2, random_state=42,
        stratify=df_main["stratum"]
    )

    for split in [df_mortality_train, df_mortality_eval]:
        split.drop(columns=['time', 'stratum'], inplace=True)
    '''

    binary_cols = [
        col for col in df_mortality_train.columns
        if set(df_main[col].dropna().unique()).issubset({0,1})
    ]

    #df_mortality = df_main.copy()
    cols_to_keep = binary_cols + ["Follow Up Data", "Data of death", "Data prelievo","Cause of death", "Collected by"]
    
    tmp_train = df_mortality_train[cols_to_keep].copy()
    tmp_eval = df_mortality_eval[cols_to_keep].copy()
    
    df_mortality_train = df_mortality_train.drop(columns=cols_to_keep)
    df_mortality_eval = df_mortality_eval.drop(columns=cols_to_keep)

   
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_array_train = scaler.fit_transform(df_mortality_train)
    scaled_array_eval = scaler.transform(df_mortality_eval)
    
    df_mortality_train = pd.DataFrame(scaled_array_train, columns=df_mortality_train.columns, index=df_mortality_train.index)
    df_mortality_eval = pd.DataFrame(scaled_array_eval, columns=df_mortality_eval.columns, index=df_mortality_eval.index)

    df_mortality_train = pd.concat([df_mortality_train, tmp_train], axis=1)
    df_mortality_eval = pd.concat([df_mortality_eval, tmp_eval], axis=1)
    '''
    columns_to_drop = [
        "Data of death",
        "Fatal MI or Sudden death",  
        "UnKnown", 
        "Accident",
        "Suicide",
        "Number",
        "CVD Death",
        "Data prelievo",
        "Cause of death", 
        "Collected by"]

    df_mortality_train = df_mortality_train.drop(columns=columns_to_drop)
    df_mortality_eval = df_mortality_eval.drop(columns=columns_to_drop)
    
    return df_mortality_train, df_mortality_eval

def prepare_cardiovascular_data(df_main):
    """Prepares data for cardiovascular competing risks."""
    df_cardiovascular = df_main.copy()
    df_cardiovascular["Other deaths"] = df_cardiovascular["Total mortality"] - df_cardiovascular["CVD Death"]
    df_cardiovascular["death"] = 0
    df_cardiovascular.loc[df_cardiovascular["CVD Death"] == 1, "death"] = 1
    df_cardiovascular.loc[df_cardiovascular["Other deaths"] == 1, "death"] = 2

    binary_cols = [
        col for col in df_cardiovascular.columns
        if set(df_cardiovascular[col].dropna().unique()).issubset({0,1})
    ]

    cols_to_keep = binary_cols + ["Follow Up Data", "Data of death", "Data prelievo", "death"]
    
    tmp = df_cardiovascular[cols_to_keep].copy()
    df_cardiovascular = df_cardiovascular.drop(columns=cols_to_keep)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_cardiovascular)

    df_cardiovascular = pd.DataFrame(scaled_array, columns=df_cardiovascular.columns, index=df_cardiovascular.index)
    df_cardiovascular = pd.concat([df_cardiovascular, tmp], axis=1)

    df_cardiovascular = df_cardiovascular.drop(columns=[
        "Fatal MI or Sudden death", "UnKnown", "Total mortality", 
        "Accident", "Suicide", "Number", "Data prelievo", "CVD Death", 
        "Other deaths", "Data of death"
    ])

    return df_cardiovascular

def prepare_mi_data(df_main):
    """Prepares data for MI endpoint."""
    df_MI = df_main.copy()
    df_MI["MI_event"] = df_main[["Fatal MI or Sudden death", "Non Fatal AMI (Follow-Up)_event"]].max(axis=1)
    df_MI["Other events"] = df_MI["Total mortality"] - df_MI["MI_event"]
    df_MI["events"] = 0
    df_MI.loc[df_MI["MI_event"] == 1, "events"] = 1
    df_MI.loc[df_MI["Other events"] == 1, "events"] = 2

    binary_cols = [
        col for col in df_MI.columns
        if set(df_MI[col].dropna().unique()).issubset({0,1})
    ]

    cols_to_keep = binary_cols + ["Follow Up Data", "Data of death", "Data prelievo", "events"]
    tmp = df_MI[cols_to_keep].copy()
    df_MI = df_MI.drop(columns=cols_to_keep)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_MI)

    df_MI = pd.DataFrame(scaled_array, columns=df_MI.columns, index=df_MI.index)
    df_MI = pd.concat([df_MI, tmp], axis=1)

    conditions = [
        df_main["Non Fatal AMI (Follow-Up)_event"] == 1,
        df_main["Fatal MI or Sudden death"] == 1
    ]
    choices = [
        df_main["Non Fatal AMI (Follow-Up)"],
        df_main["Data of death"]
    ]
    df_MI["MI_date"] = np.select(conditions, choices, default=df_main["Follow Up Data"])

    df_MI = df_MI.drop(columns=[
        "Fatal MI or Sudden death", "UnKnown", "Total mortality", 
        "Accident", "Suicide", "Number", "Data prelievo", "CVD Death", 
        "Other events", "Data of death", "Non Fatal AMI (Follow-Up)_event", 
        "Non Fatal AMI (Follow-Up)", "Follow Up Data", "MI_event"
    ])
    
    return df_MI

def prepare_stroke_data(df_main):
    """Prepares data for Stroke endpoint."""
    df_stroke = df_main.copy()
    df_stroke["events"] = 0
    df_stroke.loc[df_stroke["Ictus_event"] == 1, "events"] = 1
    df_stroke.loc[df_stroke["Total mortality"] == 1, "events"] = 2

    binary_cols = [
        col for col in df_stroke.columns
        if set(df_stroke[col].dropna().unique()).issubset({0,1})
    ]

    cols_to_keep = binary_cols + ["Follow Up Data", "Data of death", "Data prelievo", "events"]
    tmp = df_stroke[cols_to_keep].copy()
    df_stroke = df_stroke.drop(columns=cols_to_keep)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_stroke)

    df_stroke = pd.DataFrame(scaled_array, columns=df_stroke.columns, index=df_stroke.index)
    df_stroke = pd.concat([df_stroke, tmp], axis=1)

    df_stroke["Ictus_date"] = np.where( 
        df_stroke["Ictus_event"] == 1,
        df_stroke["Ictus"], 
        df_stroke["Follow Up Data"] 
    )

    df_stroke = df_stroke.drop(columns=[
        "Fatal MI or Sudden death", "UnKnown", "Total mortality", 
        "Accident", "Suicide", "Number", "Data prelievo", "CVD Death", 
        "Data of death", "Non Fatal AMI (Follow-Up)_event", 
        "Non Fatal AMI (Follow-Up)", "Follow Up Data", "Ictus_event", "Ictus"
    ])

    return df_stroke

def prepare_cox_data_hurrah_cv(df):
    """Prepares data for Cox mortality model."""

    df['time'] = pd.qcut(df['FU'], q=4, labels=False, duplicates='drop')

    # 2. Crea la variabile di stratificazione combinata
    #    Es: evento=1, bin=2 → stratum "1_2"
    df['stratum'] = df['STATO_AL_FU'].astype(str) + '_' + df['time'].astype(str)

    df_mortality_train, df_mortality_eval = train_test_split(
        df, test_size=0.2, random_state=42,
        stratify=df["stratum"]
    )

    for split in [df_mortality_train, df_mortality_eval]:
        split.drop(columns=['time', 'stratum'], inplace=True)


    binary_cols = [
        col for col in df_mortality_train.columns
        if set(df[col].dropna().unique()).issubset({0,1})
    ]

    cols_to_keep = binary_cols + ["FU","FU_NF_FA",]
    
    tmp_train = df_mortality_train[cols_to_keep].copy()
    tmp_eval = df_mortality_eval[cols_to_keep].copy()
    
    df_mortality_train = df_mortality_train.drop(columns=cols_to_keep)
    df_mortality_eval = df_mortality_eval.drop(columns=cols_to_keep)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_array_train = scaler.fit_transform(df_mortality_train)
    scaled_array_eval = scaler.transform(df_mortality_eval)
    
    df_mortality_train = pd.DataFrame(scaled_array_train, columns=df_mortality_train.columns, index=df_mortality_train.index)
    df_mortality_eval = pd.DataFrame(scaled_array_eval, columns=df_mortality_eval.columns, index=df_mortality_eval.index)

    df_mortality_train = pd.concat([df_mortality_train, tmp_train], axis=1)
    df_mortality_eval = pd.concat([df_mortality_eval, tmp_eval], axis=1)

    columns_to_drop = [
        "NF_IMA",
        "F_IMA",
        "NF_CBV",
        "F_CBV",
        "NF_HF",
        "F_HF",
        "RIV_COR",
        "MORTE_CV",
        "FU_NF_IMA",
        "FU_F_IMA",
        "FU_NF_CBV",
        "FU_F_CBV",
        "FU_NF_HF",
        "FU_F_HF"]

    df_mortality_train = df_mortality_train.drop(columns=columns_to_drop)
    df_mortality_eval = df_mortality_eval.drop(columns=columns_to_drop)
    
    return df_mortality_train, df_mortality_eval