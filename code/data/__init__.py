import os, sys 
sys.path.append(os.path.join(os.path.dirname(__file__)))  # data/*
from dataset import Dataset, MiniBatchReader