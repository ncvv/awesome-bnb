# Test
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

print("Hello World")

customer_data = pd.read_excel('CustomerDataSet.xls')
print(customer_data)

estimator = KMeans(n_clusters = 2)
labels = estimator.fit_predict(customer_data[['ItemsBought', 'ItemsReturned']])
print(labels)

plt.title("KMeans #cluster = 2")
plt.xlabel('ItemsBought')
plt.ylabel('ItemsReturned')
plt.scatter(customer_data['ItemsBought'], customer_data['ItemsReturned'], c=estimator.labels_)
plt.show()