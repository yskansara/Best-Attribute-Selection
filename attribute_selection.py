import pandas as pd
from math import log

# Read stdin line by line and store it into a list of elements
lines = []
flag = True
while flag:
    try:
        line = input().split()
        lines.append(line)
    except EOFError:
        flag = False

# Use this block for custom input
# lines = []
# while True:
#     line = input()
#     if line:
#         lines.append(line)
#     else:
#         break


num_lines = int(lines[0][0])
del lines[0]
lines = [i.split(',') for item in lines for i in item]
headers = lines[0]
del lines[0]

# dataframe for each element
df = pd.DataFrame(lines, columns=headers)
col_name = []
for col in df:
    col_name.append(col)
u_list = list(df[col_name[-1]].unique())
resp_val = list(df[col_name[-1]].value_counts())

# Function to calculate entropy
def calc_entropy1(truth_list):
    entropy = 0
    for i in truth_list:
        i = i / sum(truth_list)
        if i != 0:
            entropy += -i * log(i, 2)
        else:
            entropy += 0
    return entropy

# Funstion to calculate information gain (resp_attr: Target Attribute)
def info_gain(resp_attr, attr):
    t = 0
    for i in attr:
        t += sum(i) / sum(resp_attr) * calc_entropy1(i)
    gain = calc_entropy1(resp_attr) - t
    return gain

# Calculate Gain Ratio
def calc_gain_ratio(attr):
    split_info = 0
    sum_i = 0
    for i in attr:
        sum_i = sum(i) / sum(sum(j) for j in attr)
        split_info += -sum_i * log(sum_i, 2)
    gain_ratio = attr_gain / split_info
    return gain_ratio


def calc_gstart(truth_list):
    sub = 0
    gstart = 0
    for i in truth_list:
        i = (i / sum(truth_list)) ** 2
        if i != 0:
            sub += i
        else:
            sub += 0
    gstart = 1 - sub
    return gstart

# Calculate Gini index
def gini_index(resp_attr, attr):
    gf = 0
    gsplit = 0
    gini = 0
    for i in attr:
        gf = calc_gstart(i)
        sum_i = sum(i) / sum(sum(j) for j in attr)
        gsplit += sum_i * gf
    gini = calc_gstart(resp_attr) - gsplit
    return gini


all_gain = []
all_gain_ratio = []
all_gini_index = []
for col in col_name[:-1]:
    attr_gain = []
    attr_ratio = []
    attr_gini = []
    df1 = df[[col, col_name[-1]]].copy()
    attr_ulist = list(df1[col].unique())
    temp1 = []
    for i in attr_ulist:
        temp = []
        for j in u_list:
            temp.append((df1[(df1[col] == i) & (df1[col_name[-1]] == j)][col]).count())
        temp1.append(temp)
    attr_gain = info_gain(resp_val, temp1)
    attr_ratio = calc_gain_ratio(temp1)
    attr_gini = gini_index(resp_val, temp1)
    all_gain.append(attr_gain)
    all_gain_ratio.append(attr_ratio)
    all_gini_index.append(attr_gini)

# Print the best attribute name
print(col_name[all_gain.index(max(all_gain))])
print(col_name[all_gain_ratio.index(max(all_gain_ratio))])
print(col_name[all_gini_index.index(max(all_gini_index))])p