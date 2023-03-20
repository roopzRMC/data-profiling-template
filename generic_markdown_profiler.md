###  **_Required Dependancies_**


```python
#installing Dependancies 
%pip install pandas
%pip install numpy 
%pip install glob2 
%pip install pypi-json 
%pip install matplotlib 
%pip install missingno 
%pip install klib 
%pip install plotly  
%pip install pypi-json 
```


```python
#importing Dependancies

import os
import json
import joblib
import glob
import time
import pandas as pd 
import numpy as np 
import re 
import missingno as mn
import seaborn as sns
from pandas_profiling import ProfileReport
from pandas.api.types import is_numeric_dtype
import missingno as msno
import matplotlib.pyplot as plt
%matplotlib inline

```

Is a modelling target attribute present in your data if you expect one to exist? 


How is your attribute distributed? 

* Distribution refers to the statisticsl way values are spread out or arranged in a dataset. 
* Common types include normal, skewed, bimodal, uniform, and exponential distribution. 
* Understanding the distribution is important for selecting appropriate statistical tests and identifying outliers and anomalies in the data.


```python
# Basic statistical information
statistics = df.describe(include=np.number)
for col in statistics.columns:
    std = statistics.loc['std',col]
    skew = df[col].skew()
    kurt = df[col].kurtosis()
    print(f'* **{col}**: (std={std:.4f}, skew={skew:.2f}, kurt={kurt:.2f})')
```

If available, how does your attribute change over time? 
* Analyzing attribute distribution over time can reveal trends and patterns that may be missed in overall distribution analysis


```python

```

Is there a time seasonality within the attribute? 
* Time seasonality within an attribute refers to repeating patterns over specific periods of time, such as weeks, months, or years.
* eg. Seasonal fluctuations in sales of a product or variations in website traffic over the course of a day or week.

Statistical Analysis of numerical variables in dataset


```python

```

## Data Quality 

DAMA - Data managment 

- Completness
- Consistenccy  
- Accuracy 
- Timeliness 
- Validity 

> Data Quality and Data Managment Best Practice is used to enable a governed set of outcomes against the BPM Business Process Model 


```python
# Reading Data For Assesment 
## File Format of the data is unknown, a method of importing a specific piece of data, or files within the folder structure 
# * Assuming Flatfile Format*

df = pd.read_csv(file, delimiter= '\t', na_values= 'Nan', low_memory= False)

```

### _Dataset completness_ 
#####  Thresholds can be set to remove a pct of null values within the dataset 


```python
#Caluculating missing values and resprective %
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total Missing Values', '%Missing'])
missing_data.index.name = 'Features'
missing_data

```


```python
## Method of filling missing
df = df.interpolate(method='linear', * ,
                    axis=0, 
                    limit=None, 
                    inplace=True, 
                    limit_directions=None,
                    limit_area=None, 
                    downcast=None, **kwargs)
```

#### Correlation between attributes* C orrelation is a statistical measure that describes the strength and direction of a linear relationship between two variables. I
* t ranges from -1 to +1, whhere,
    Crrelation coefficient of -1 indicates a perfect negative correlation, a
    Cefficient of +1 indicates a perfect positive correlation, and a
    Cefficient of 0 indicates no correlation.
 * There are different types of correlation tests that can be used depending on the nature of the data and the research question being asked. Some common types of correlation tests include:

    Pearson's correlation coefficient
    Spearman's rank correlation coefficient
    Kendall's tau correlation coefficient
    Point-biserial correlation coefficient
    Phi coefficient:


```python
corr = df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(10, 220, as_cmap=True)
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(7, 5))
    ax = sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        square=True,
        annot=False,
        linewidth=0.5,
        cbar_kws={"shrink": 0.7},
    )
corr
```

Are there any hidden characters in your attribute you need to deal with? 

initial data cleaning steps, i.e striping the white space, changing the case of textual data to lowercase formats, and removing special characters.

* cleaning column names 
* removing values from initial dataset 

"Transformations"

```
df.lower()
df.upper()
```
"Aggregations"

* Mode 
* Median 
* Mean



What attributes can be discarded from future analysis? 

> redundancy of columns and/or data with x correlation can be disregarded from the dataset 
an example method of this could be to: 



```python
# corr within data, dropping columns 
cols = corr.columns
redunt = []
k = 0
for ind, c in enumerate(corr[y]):
    if c<1-threshold: 
        redunt.append(cols[ind])
return redunt

```

Are the values in your data consistently cased? (categorical). 

Consistency
    : Definition

* Consitent values in the dataset helps ensure that the data is accurate and reliable. inconsistencies and/or errors in the data, can potentialy lead to inaccurate analysis and recommendations  



```python

```

### Uniquness 

* Uniqueness refers to the property of data where each observation or entity is distinct and can be identified uniquely. 
* Uniqueness can be ensured by using unique identifiers such as primary keys or unique combinations of variables to distinguish between different observations or entities.

How many unique values exists for the attribute?

###  Exploring Uniqeness in dataset 
```uniqueVals = list(np.unique(col))```





```python
# finding distinct and unique values within dataset 
unique = pd.DataFrame(columns=['Feature', 'Unique Count', 'Distinct Values'])
for i, col in enumerate(df.columns):
    unique.loc[i] = [col, len(df[col].unique()), str(df[col].unique())]
```

Are there any hidden characters in your attribute you need to deal with? 
* Special Characters
* Whitespaces 



```python
#Remove whitespaces using Lambda Function
df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x) 
df = df.apply(lambda x: x.str.strip() if x.dtype == 'int' else x) 
df = df.apply(lambda x: x.str.strip() if x.dtype == 'float64' else x) 


#Removing Regex 
#Pattern using regex 
df["col"] = df['col_spec_wrds'].str.replace('\w', '',  regex=True) # This can be done for pii, phi, information also
df["col_nums"] = col["_words_nums"].str.replace(r'\w\d'+ $) # sudo 

#Remove special characters 
#disribution of data 
# Get basic statistical information on the dataset
describe = df.describe()
quartiles = pd.concat([df.quantile(.25), df.quantile(.5), df.quantile(.75)], axis=1)
quartiles.columns = ["25%", "50%", "75%"]
```


```python
#quick example can be excluded 

import re # This is Regex : Regual Expression

text = "This is some random text with 9999 numbers and an email address example@example.com and some white spaces."

# Find all white spaces
white_spaces = re.findall('\s+', text)
print("White Spaces:", white_spaces)

# Find all numbers
numbers = re.findall('\d+', text)
print("Numbers:", numbers)

# Find all email addresses
email = re.findall('\S+@\S+', text)
print("Email Addresses:", email)


# example of this in comprehension 
nums = [1, 2, 3, 4, 5, 6]
evens = [x for x in nums if x % 2 == 0]
print(evens)

keys = ['a', 'b', 'c']
values = [1, 2, 3]
d = {k: v for k, v in zip(keys, values)}
print(d)


```


      Cell In[15], line 21
        num nums = [1, 2, 3, 4, 5, 6]
            ^
    SyntaxError: invalid syntax
    


> quick example 
```python
import re # This is Regex : Regual Expression
text = "This is some random text with 9999 numbers and an email address example@example.com and some white spaces."
# Find all white spaces
white_spaces = re.findall('\s+', text)
print("White Spaces:", white_spaces)
# Find all numbers
numbers = re.findall('\d+', text)
print("Numbers:", numbers)
# Find all email addresses
email = re.findall('\S+@\S+', text)
print("Email Addresses:", email)
```
 ### Visualization
 * visualization is the graphical representation of data and statistical analysis, and it can help to reveal patterns, trends, and relationships.
 * Based on the type of attribute (Numeric/ catagotical), visualization type is selected.
 * eg., Scatterplot, Histogram, Boxplot, Barchart, Heatmap, Pie-chart

What is the best way to visualise the values in this attribute? 


```python
## Boxplot
plt.boxplot(df)

# Set the title and axis labels
plt.title("Histogram of Example Data")
plt.ylabel("Frequency")

# View the plot
plt.show


## Histogram
# Create a histogram with 20 bins
plt.hist(df, bins=20)

# Set the title and axis labels
plt.title("Histogram of Example Data")
plt.xlabel("Value")
plt.ylabel("Frequency")

# View the plot
plt.show()
```

Do you know what each of the attributes mean? (Reference or build a data dictionary). 

_Data Literacy_: 
 * Entity
    : X Definition
 
* Elements
    : Y Definition 
* Attributes
    : z Definition


```python

```

What attributes exists that can exist as keys for data linkage? 
* are the code or id columns actually unique for the given dataset

```python
## Identifying the unique ids that can make an effective join across multiple datasets

assert df['id'].nunique() == df.shape[0]

```

Are the data types standardised in the column? 


```python

df.dtypes
df.select_dtypes(include='int64')


```

Does the attribute contain any PII? 

* Personal identfiable Information, within the dataset a method of bulk minning, and or bulk analysis on key=[ "Name", "Numbers", "Phone Numbers", "Email Address"]
* if this is found report it and hash it



```python

import hashlib
# Encode our string using UTF-8 default 
stringToHash = 'example@email.com'.encode()


```

What outliers exist in the attribute? 
* Performing univariate tests such as Box plots, IQR Bounding

```python

```

Are any of your attributes nested object structures that need further processing?
* Are there repeated ids
* Are the multi level indexes apparent in the data

```python

df.index.nlevels 
df.columns.nlevels 

```

How many missing values exists? 
* Leverage Yavins data profiling tool - missingness section

```python
msno.matrix(df)

```

Can you deduce any patterns from missing values or are they randomly missing (use a package such as missingno to help you determine)? 
* Missing completely at random - no systematic relationsjip between missing data and other values
* Missing at Random - relationship between missing data and other observed values
* Missing not at random - relationship between missing data and unobserved values

```python
msno.matrix(df)

```

Do any features have a greater than 30% missingness? 


```python

```

Are there any correlations in data missingness? 
* Using the msno missingness coreelation matrix to determine whether there relationship between features that are missing


```python

```

What options exists for performing imputation on the missing data? 
* ML approaches such as KNN vs statistic scalar imputation (not recommended)

```python

```

When joining datasets together are there duplicates? Ie is there a granularity mismatch 
* test the length of the resulting dataframe having executed a merge
* check for uniqueness of the PK

```python

```

**Do you have the information to create a data model across the provided sets?** FOR MASTER summary


```python
## List here the relevant PKs and FKS

```

How many rows of data are available? 

* Is the extract of the expected length?


```python
df.shape
```

What are the date ranges in your data, are there any gaps?

* At a minimum we are expecting 3 years of data across each dataset 
* Determine baseline start date with other datasets - take the start date as the most recent available across all datasets
* This requires team engagement


```python

```

Do you have duplicate row records in your dataset? 
* Tests can be performed on a complete row by row basis or by subset

```python
df.duplicated(subset = None, keep = ‘first’)
```

If you evaluate the head/tail of your dataset, do any observations grab your attention?
 
* Test for obvious datatype mismatches - text where you expect to see numerical values etc
* what format is the unique identifer in - are the preceeding zeros
* Have any obvious encoding errors occured in the process of converting to a pandas df

```python

```

How relevant is this dataset for each of the commercial use cases? 
* Test against the original data request and determine RAG status


```python
## No code required
```

Does your dataset have appropriate attribute headers? Do you need to add a header? 


```python

```


> Data Quality

Are the data types standardised in the column?

* Are categorical features encoded as objects or visa-versa
* Are datetime features encoded as datetime?
* Are ints encoded as floats?

```python

```






What is the distribution of values in the attribute?
* Use appropriating binning techniques and visualise with histrograms, displots

```python

sns.displot

```

What duplicates exist in your attribute, are they valid? 

* Test for multi indices

Are any of your attributes nested object structures that need further processing? 

* Check for evidence of long objects / presence of ```{}``` within df cells


Does your data contain any text-based information you need to mine? 

* Test for features encoded as objects and/or headers that contain the word 'comments'
* Earmark for NLP

Do any values look strange/unexpected for the attribute? [DUPE]

Do the attributes follow consistent naming conventions? 
* test headers 

Are the attributes named in an accurate manner that represents the values? 

> Data Linkage

What attributes exists that can exist as keys for data linkage? [DUPE]

When joining datasets together are there duplicates? Ie is there a granularity mismatch 
* On checking the shape the of the resulting dataframe having performed a merge - do the number of rows increase?

