import pandas as pd
labels = {1:"Netral",2:"Netral",3:"Negatif",4:"Negatif",5:"Netral",6:"Negatif",7:"Negatif",
8:"Negatif",9:"Netral",10:"Negatif",11:"Netral",12:"Netral",13:"Netral",14:"Netral",
15:"Netral",16:"Netral",17:"Negatif",18:"Netral",19:"Netral",20:"Netral",21:"Netral",
22:"Negatif",23:"Negatif",24:"Negatif",25:"Negatif",26:"Negatif",27:"Netral",28:"Netral",
29:"Negatif",30:"Negatif",31:"Negatif",32:"Netral",33:"Netral",34:"Negatif",35:"Netral",
36:"Positif",37:"Netral",38:"Positif",39:"Positif",40:"Netral"}
df = pd.read_csv('data/validasi_manual.csv', encoding='utf-8-sig')
df['label_ai'] = df['id'].map(labels)
df['tinjauan_manual'] = ''
df['catatan'] = ''
df.to_csv('data/validasi_manual.csv', index=False, encoding='utf-8-sig')
print(df['label_ai'].value_counts().to_string())
print('agreement:', round((df['label_ai'] == df['label_hybrid']).mean(), 3))
print(pd.crosstab(df['label_hybrid'], df['label_ai']))
