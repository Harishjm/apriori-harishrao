from websocket._http import proxy_info

dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

product = input("Enter any product")

sub_dataset = []
for i in dataset:
    if product in i:
        sub_dataset.append(i)
#pip install mlxtend
#pip install pandas



import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(sub_dataset).transform(sub_dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)


df1 = df
df1[product] = False
df1


from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

df2 = apriori(df1, min_support=0.6)
print(df2['itemsets'])




from apyori import apriori


#frequent_itemsets = apriori(df1, min_support=0.6, use_colnames=True)
#from mlxtend.frequent_patterns import association_rules
#rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
#print(rules.head())



rules = apriori(dataset, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length=2)
results = list(rules)
print("number of rules", len(results))
for i in range(0, 20):
    result = results[i]
    supp = int(result.support * 10000) / 100
    conf = int(result.ordered_statistics[0].confidence * 100)
    hypo = ''.join([x + ' ' for x in result.ordered_statistics[0].items_base])
    conc = ''.join([x + ' ' for x in result.ordered_statistics[0].items_add])
    print("if " + str(hypo) + " is visited --> " + str(conf) + " % that " + str(conc) + " is visited [support = " + str(
        supp) + "%]")
