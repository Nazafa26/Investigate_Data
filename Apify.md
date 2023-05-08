# Apify Pricing

Apify is a platform for web scraping. It lets developers and companies automate manual workflows they usually do on the web. Customers are able to use one of 200+ ready-made tools for web scraping and web automation to scrape web pages, e-commerce platforms, mobile apps, social media, and other websites.  __At least, it's what claimed on the Apify website.__

All of these tools which are called __actors__ in Apify Store fall into one of these three pricing models:

__Free__ - you can run the actor freely and you only pay for platform usage the actor generates.

__Paid__ - same as free, but in order to be able to run the actor after the trial period, you need to rent the actor from the developer and pay a flat monthly fee on the top of the platform usage the actor generates.

__Paid per result__ - you do not pay for platform usage the actor generates and only pay for the results it produces.

You can find more details about the pricing models through the following link: https://docs.apify.com/platform/actors/running/actors-in-store

### Aim

In this case, we find out which one of the the pricing models: __Paid__ , __Paid per result__ and __Free__ in this platform, generated more profit. To analyse the data and display the result, we use Multivariate regression in supervised machine learning. It is an extension of multiple regression with one dependent variable and multiple independent variables. Based on the number of independent variables, we try to predict the output.

We analyze the usage of the platform in __USD__ based on the two required features: __Compute Unit__ and __Dataset Items Count__ and compare the amount of profit generated in each of these two variables.

### Data

****The data used in this projected was provided by the Apify data team lead, during the job interview.****

### Libraries Used


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns

from sklearn.metrics import r2_score
from pylab import *
from scipy import stats
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```

### Loading the Data


```python
df = pd.read_csv('actor_runs_2023.csv')
```

### Sample Data


```python
df.head()
# x1 => run_id        x
# x2 => user_id     y
# x3 => actor_id z
# x4 => build_id
# x5 => started_at
# x6 => finished_at
# x7 => status
# x8 => usage_key_value_store_reads
# x9 => usage_key_value_store_writes
#.
#.
#.
# x17 => usage_compute_units
# x18 => dataset_items_count
# x19 => total_platform_usage_usd 
# y = a1 * x1 + a2 * x2 + a3 * x3 + 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>run_id</th>
      <th>user_id</th>
      <th>actor_id</th>
      <th>build_id</th>
      <th>started_at</th>
      <th>finished_at</th>
      <th>status</th>
      <th>usage_key_value_store_reads</th>
      <th>usage_key_value_store_writes</th>
      <th>usage_key_value_store_lists</th>
      <th>usage_request_queue_reads</th>
      <th>usage_request_queue_writes</th>
      <th>usage_dataset_reads</th>
      <th>usage_dataset_writes</th>
      <th>usage_residential_proxy_transfer_bytes</th>
      <th>usage_proxy_serps</th>
      <th>usage_compute_units</th>
      <th>dataset_items_count</th>
      <th>total_platform_usage_usd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Re6S1iNa9VakoMgaR</td>
      <td>vmNDHzHvi8srXSbk8</td>
      <td>m1NPxIieJaZd9UhP6</td>
      <td>Mi8kbH6PMC65ZmuvV</td>
      <td>2023-03-16T11:12:02.824Z</td>
      <td>2023-03-16T11:12:22.534Z</td>
      <td>SUCCEEDED</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>28</td>
      <td>0</td>
      <td>0</td>
      <td>0.001357</td>
      <td>28</td>
      <td>0.000534</td>
    </tr>
    <tr>
      <th>1</th>
      <td>c0NvWUPZEXI3jDRC7</td>
      <td>WYC4rsszWSw5iCkkW</td>
      <td>bf54TfrKoJrQZsrZm</td>
      <td>Mnsb9UmkTtab7oMFG</td>
      <td>2023-03-16T12:25:10.06Z</td>
      <td>2023-03-16T12:25:13.708Z</td>
      <td>FAILED</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.001906</td>
      <td>0</td>
      <td>0.000532</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7Ta0QDxg8gQChzKnF</td>
      <td>Z9JpDf74uHecET97E</td>
      <td>9Sk4JJhEma9vBKqrg</td>
      <td>kqMNhAaZCiyxawUns</td>
      <td>2023-03-16T19:39:03.889Z</td>
      <td>2023-03-16T19:40:08.04Z</td>
      <td>SUCCEEDED</td>
      <td>1</td>
      <td>11</td>
      <td>0</td>
      <td>14</td>
      <td>73</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0.071098</td>
      <td>0</td>
      <td>0.020086</td>
    </tr>
    <tr>
      <th>3</th>
      <td>IR4pAPViSQcu39Yws</td>
      <td>Wb79moXaBTjTo3ieC</td>
      <td>YPh5JENjSSR6vBf2E</td>
      <td>WmEnXz3WBxfEaM6Ko</td>
      <td>2023-03-16T17:17:20.111Z</td>
      <td>2023-03-16T17:17:29.662Z</td>
      <td>SUCCEEDED</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.005203</td>
      <td>0</td>
      <td>0.001461</td>
    </tr>
    <tr>
      <th>4</th>
      <td>YqBVQQN5emULPxBuL</td>
      <td>pqQB4FWGoS8nsfMoo</td>
      <td>moJRLRc85AitArpNN</td>
      <td>VezLeaxKZT0ZmuguN</td>
      <td>2023-03-16T16:47:55.406Z</td>
      <td>2023-03-16T16:48:07.312Z</td>
      <td>SUCCEEDED</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>8</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.001630</td>
      <td>1</td>
      <td>0.003703</td>
    </tr>
  </tbody>
</table>
</div>



### Plot of 'Platform Usage in USD' vs. 'CU' and 'Dataset Items'


```python
sns.relplot(x='total_platform_usage_usd', y = 'usage_compute_units', data = df )
sns.relplot(x='total_platform_usage_usd', y = 'dataset_items_count', data = df )
# this is just to show there is no correlation on one variables
```




    <seaborn.axisgrid.FacetGrid at 0x7fc96fdf9dc0>



![output_13_1](https://user-images.githubusercontent.com/112628373/236853554-756985f6-0be3-492e-92f4-dbc35ed40f2a.png)



    



![output_13_2](https://user-images.githubusercontent.com/112628373/236853587-747670ae-a5c2-48f4-971d-92c0513d3b3f.png)


    


### Observation

We can observe that there is a multivariate regression between total_platform_usage_usd and dataset_items_count and usage_compute_units. In the dataset we can see when usage_compute_units increases, total_platform_usage_usd increases as well. But, when dataset_items_count increases, the total_platform_usage-usd has a polynomial regression and not neccessarily increases.

### MultiVariate Function:

We drop some of the columns which are not nominal as the function will have an error to have a combination of float and string.


```python
# Setting the X and y
X = df.drop(columns= ['total_platform_usage_usd', 'usage_proxy_serps', 'build_id', 'run_id', 'user_id', 'started_at', 'finished_at', 'status', 'actor_id']) # X => Feature df
y = df['total_platform_usage_usd'] # tagert
X.head() # Feture
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>usage_key_value_store_reads</th>
      <th>usage_key_value_store_writes</th>
      <th>usage_key_value_store_lists</th>
      <th>usage_request_queue_reads</th>
      <th>usage_request_queue_writes</th>
      <th>usage_dataset_reads</th>
      <th>usage_dataset_writes</th>
      <th>usage_residential_proxy_transfer_bytes</th>
      <th>usage_compute_units</th>
      <th>dataset_items_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>28</td>
      <td>0</td>
      <td>0.001357</td>
      <td>28</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.001906</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>11</td>
      <td>0</td>
      <td>14</td>
      <td>73</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0.071098</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.005203</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>8</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.001630</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Testing the Dataset

To test the dataset we created four variables to train and then test the created function. 
We chose the test size equal to 20 percent. 


```python
X_train , X_test, y_train, y_test = train_test_split(X, y , test_size =.2, random_state=0)
```


```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # This is just the library not simple liner regression 
regressor.fit (X_train, y_train) # Training
```




    LinearRegression()




```python
y_pred = regressor.predict(X_test) # Test on previously not seen dataset

results = pd.DataFrame({'Actual' :y_test, 'Predicted': y_pred}) # Creating a comaprsion Table
```


```python
results.head() # only display first 5
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1348815</th>
      <td>0.003862</td>
      <td>0.001596</td>
    </tr>
    <tr>
      <th>4438721</th>
      <td>0.017274</td>
      <td>0.043365</td>
    </tr>
    <tr>
      <th>4519922</th>
      <td>0.006635</td>
      <td>0.005133</td>
    </tr>
    <tr>
      <th>4527259</th>
      <td>0.001924</td>
      <td>0.001848</td>
    </tr>
    <tr>
      <th>2429339</th>
      <td>0.007755</td>
      <td>0.012183</td>
    </tr>
  </tbody>
</table>
</div>



### R2 Score

R is the accuracy of the algoritm, which in this case is a good number because it is close to 1.


```python
from sklearn.metrics import r2_score
r2_score(y_test, y_pred) 
```




    0.9122665509540744



## Conclusion

In __Usage Compute Units__ pricing model, most consumers used the free pricing model and up to 100 dollars which can also be fixed pricing. The usage was decreasing for more expensive pricing.
In __Dataset Items Count__ pricing model, the consumers were charged even if they didn't use the CPU and memory resources consumed by the Actor.

Apify's website and pricing plans are not straightforward without evident and consumer-friendly pricing descriptions; therefore, any improvement in the process of this project is welcomed.
