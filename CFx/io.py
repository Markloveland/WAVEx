import numpy as np


def string_convert(Dict,key,otype):
    str1 = Dict[key]
    if otype =="float":
        Dict[key]=float(str1)
    if otype =="int":
        Dict[key] = int(str1)
    if otype =="float list":
        Dict[key] = [float(x) for x in str1.split(",")]
    if otype =="int list":
        Dict[key] = [int(x) for x in str1.split(",")]
    if otype =="str list":
        Dict[key] = str1.split(",")
    return 0


def read_input_file(filename):
    #read input file into a dictionary
    Model_Parameters = {"Run Name": None,
                        "Comment": None,
                        "Weak Form": None,
                        "Theta Param": None,
                        "Mesh Type": None,
                        "Grid Address" :None,
                        "Geographic Bounds" : None,
                        "Geographic Cells": None,
                        "Bathymetry" : None,
                        "Currents" : None,
                        "Wet/Dry" : None,
                        "Spectral Bounds": None,
                        "Spectral Cells": None,
                        "Start Time": None,
                        "End Time": None,
                        "DT": None,
                        "Plot Every" : None,
                        "Output Folder": None,
                        "Boundary Type": None,
                        "Gaussian Params": None,
                        "QoI": None,
                        "Station Params": None}

    F_in = open(filename,"r")
    for line in F_in:
        splits = line.split("!")
        split2 = splits[0].strip().split('=')
        Model_Parameters[split2[0]] = split2[1]

    #convert things into proper types
    string_convert(Model_Parameters,"Theta Param","float")
    string_convert(Model_Parameters,"Geographic Bounds","float list")
    string_convert(Model_Parameters,"Geographic Cells","int list") 
    string_convert(Model_Parameters,"Spectral Bounds","float list") 
    string_convert(Model_Parameters,"Spectral Cells","int list") 
    string_convert(Model_Parameters,"Start Time","float") 
    string_convert(Model_Parameters,"End Time","float") 
    string_convert(Model_Parameters,"DT","float") 
    string_convert(Model_Parameters,"Plot Every","int") 
    string_convert(Model_Parameters,"Gaussian Params","float list") 
    string_convert(Model_Parameters,"Station Params","int list") 
    string_convert(Model_Parameters,"QoI","str list") 

    print(Model_Parameters)
    return 0
    
