import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as math
import scipy
from scipy.stats import norm

def getdata(filename):
    data = open(filename, 'r')
    data.readlines()
    for line in data:
        for i in range(len(line)):
            if i == 

    print(data)


if __name__ == '__main__':
    getdata('23-01-24_testing1.csv')