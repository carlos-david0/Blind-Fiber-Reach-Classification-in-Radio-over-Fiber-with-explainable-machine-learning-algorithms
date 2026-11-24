import pandas as pd
import tsfel
import numpy as np
from pathlib import Path
from feature_engine.selection import DropCorrelatedFeatures
lenght_list = [ 20000, 40000, 60000, 80000, 100000]
for legnth in lenght_list:
    features = []
    OSNR = []
    distance = []
    direccion_base_datos = Path("../datos rotados/USPL")
    cfg = tsfel.get_features_by_domain(['statistical', 'temporal', "spectral"])
    i = 0
    for directory in sorted(direccion_base_datos.iterdir()):
        if directory.is_file():
            continue
        else:
            dir_name = directory.name
        print(dir_name)
        for subdir in sorted(directory.iterdir()):
            if subdir.is_dir():
                continue
            print(subdir.name)
            file_path = Path(f"../datos rotados/USPL/{dir_name}/{subdir.name}")
            if file_path.exists():
                data = pd.read_csv(file_path)
                data = data["rx"].to_list()
                data = [complex(point.strip("()")) for point in data]
                for i in range(0, len(data)//legnth):
                    if dir_name == "0km":
                        distance.append(0)
                    elif dir_name == "2km":
                        distance.append(1)
                    elif dir_name == "20km":
                        distance.append(2)
                    elif dir_name == "40km":
                        distance.append(3)

                    OSNR.append(float(subdir.name[subdir.name.find("osnr")+len("osnr"):subdir.name.find("dB_")].replace("p", ".")))
                    I = [point.real for point in data[i*legnth:(i+1)*legnth]]
                    Q = [point.imag for point in data[i*legnth:(i+1)*legnth]]
                    #print(tsfel.time_series_features_extractor(cfg, I).columns.append(tsfel.time_series_features_extractor(cfg, I).columns)[32])
                    features0 = tsfel.time_series_features_extractor(cfg, I, fs = 650000000).iloc[0].tolist()
                    features1 = tsfel.time_series_features_extractor(cfg, Q, fs = 650000000).iloc[0].tolist()
                    features.append(features0+features1)
    direccion_base_datos = Path("../datos rotados/DFB")
    for directory in sorted(direccion_base_datos.iterdir()):
        if directory.is_file():
            continue
        else:
            dir_name = directory.name
        print(dir_name)
        for subdir in sorted(directory.iterdir()):
            if subdir.is_dir():
                continue
            print(subdir.name)
            file_path = Path(f"../datos rotados/DFB/{dir_name}/{subdir.name}")
            if file_path.exists():
                data = pd.read_csv(file_path)
                data = data["rx"].to_list()
                data = [complex(point.strip("()")) for point in data]
                for i in range(0, len(data)//legnth):
                    if dir_name == "0km":
                        distance.append(0)
                    elif dir_name == "2km":
                        distance.append(1)
                    elif dir_name == "20km":
                        distance.append(2)
                    elif dir_name == "40km":
                        distance.append(3)

                    OSNR.append(float(subdir.name[subdir.name.find("osnr")+len("osnr"):subdir.name.find("dB_")].replace("p", ".")))
                    I = [point.real for point in data[i*legnth:(i+1)*legnth]]
                    Q = [point.imag for point in data[i*legnth:(i+1)*legnth]]
                    #print(tsfel.time_series_features_extractor(cfg, I).columns.append(tsfel.time_series_features_extractor(cfg, I).columns)[32])
                    f_I = tsfel.time_series_features_extractor(cfg, I, fs=650000000, verbose=0)
                    f_Q = tsfel.time_series_features_extractor(cfg, Q, fs=650000000, verbose=0)
                    if col_names is None:   # define col_names = None before the loops
                        col_names = [f"{c}_I" for c in f_I.columns] + [f"{c}_Q" for c in f_Q.columns]
                    features.append(f_I.iloc[0].tolist() + f_Q.iloc[0].tolist())
    df = pd.DataFrame(features, columns=col_names)
    df["OSNR"] = OSNR
    df["distance"] = distance
    df.to_csv(f"../datos tsfel/USPL/caracteristicas_raw{legnth}.csv", index=False)
