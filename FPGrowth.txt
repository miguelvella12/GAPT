from datetime import datetime
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
import psycopg2

#Starts time to check how long script takes to run
startTime = datetime.now()

#Establishes connection to database
conn = psycopg2.connect(host="localhost", port=5432, database="gapt2",
                            user="postgres", password="gapt")
cur = conn.cursor()

#Retrieves all data that is saved in dataset
s = "SELECT * FROM groceries"
cur.execute(s)
transaction_list = cur.fetchall()

cur.close()
conn.close()

#Sets column and row size in output
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',10)
dataset = pd.DataFrame(transaction_list)

#Changes need to be made to dataset to apply fp algorithm
#Converting the data frame into a list of lists
records = []
for i in range (0, 9835):
    records.append([str(dataset.values[i,j]) for j in range(0, 20)])

#"Cleaning" dataset by using TransactionEncoder and dropping the 'None' column
TE = TransactionEncoder()
array = TE.fit(records).transform(records)
transf_df = pd.DataFrame(array, columns = TE.columns_)
cleanDataset = transf_df.drop(['None'], axis = 1)

#Using association rules to mine dataset
assocRules = fpgrowth(cleanDataset, min_support=0.05, use_colnames=True)
rules = association_rules(assocRules, metric = 'lift', min_threshold = 1)
print(rules)

#Prints time taken for execution
print(datetime.now() - startTime)
