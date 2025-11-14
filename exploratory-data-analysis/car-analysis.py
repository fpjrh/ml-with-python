import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load the data
FILE_NAME = '/Users/fpj/Development/python/ml-with-python/exploratory-data-analysis/data/automobileEDA.csv'

df = pd.read_csv(FILE_NAME, header=0)
print(df.head())
#
print(df.dtypes)
# 
dfn = df.select_dtypes(include=np.number)
#
print(dfn.corr())
#
dfbschp = df[['bore', 'stroke', 'compression-ratio', 'horsepower']]
#
print(dfbschp.corr())
#
# Engine size as potential predictor variable of price
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)
plt.show()
#
print(df[["engine-size", "price"]].corr())
#Code: 
sns.regplot(x="stroke", y="price", data=df)
plt.ylim(0,)
plt.show()
#
sns.boxplot(x="body-style", y="price", data=df)
plt.show()

# grouping
df['drive-wheels'].unique()
df_group_one = df[['drive-wheels','body-style','price']]
# grouping results
df_group_one = df_group_one.groupby(['drive-wheels','body-style'],as_index= False).mean()
print(df_group_one)
# pivot table
grouped_pivot = df_group_one.pivot(index='drive-wheels', columns='body-style').fillna(0)
print(grouped_pivot)
# heatmap
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()
#
df_group_two = df[['body-style','price']]
# grouping results
df_group_two = df_group_two.groupby(['body-style'],as_index= False).mean()
print(df_group_two)

# let's do a bit of correlation analysis
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient for wheel-base is", pearson_coef, " with a P-value of P =", p_value)  
#
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient horsepower is", pearson_coef, " with a P-value of P = ", p_value)  
#
pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient length is", pearson_coef, " with a P-value of P = ", p_value)  
#
pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient engine-size is", pearson_coef, " with a P-value of P =", p_value) 
#
pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print( "The Pearson Correlation Coefficient curb-weight is", pearson_coef, " with a P-value of P = ", p_value)  