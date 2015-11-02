# Data science project with Avazu Dataset

More information can be seen [here](https://www.kaggle.com/c/avazu-ctr-prediction/).

### Problem Description

The data consists in user logs saved along 11 days, in which there is an information concerning whether an user clicked in an ad or not. The aim of the project is to calculate the probability of clicking in an ad given the set of features captured.

### Features in the dataset:
* **id**: ad identifier
* **click**: 0/1 for non-click/click
* **hour**: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC
* **C1**: anonymized categorical variable
* **banner_pos**
* **site_id**
* **site_domain**
* **site_category**
* **app_id**
* **app_domain**
* **app_category**
* **device_id**
* **device_ip**
* **device_model**
* **device_type**
* **device_conn_type**
* **C14-C21**: anonymized categorical variables


### An overview on the implementation
To run the code:
cd <path_to_this_root_folder>/src
python avazu.py -t <training_file>

**Note**: The -h flag will show more flags that you can use to specify samplings or more processes that you may want to perform.