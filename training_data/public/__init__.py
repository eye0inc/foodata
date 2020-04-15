#All hail the BDFL
import os

def datapath():
    i = __file__.rfind("/")
    return __file__[:i+1]

def srcpath():
    i = __file__.rfind("/")
    i = __file__[:i].rfind("/")
    return __file__[:i+1]

def list():
    return os.listdir(datapath())
