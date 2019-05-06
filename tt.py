import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f', 'h'],
                 columns=['one', 'two', 'three']) 

df['four'] = 'bar'

df['five'] = df['one'] > 0
print(df)
df2 = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
print(df2)
print(df2['one'])
print(pd.isna(df2['one']['a']))
print(type(df2['one']['a']))
df2.fillna(np.nan,inplace=True)
print('df2',df2)
values = df2.four.unique()
print('val',values)
dt = pd.DataFrame([[np.nan, 2, np.nan, 0],
                    [3, 4, np.nan, 1],
                    [np.nan, np.nan, np.nan, 5],
                    [np.nan, 3, np.nan, 4]],
                 columns=list('ABCD'))
print(dt)

dt.fillna('ss',inplace=True)
print(dt)

print(pd.isna(dt['A'][0]))
dtlist=dt.values.tolist()

x = np.array(dtlist) 
print(np.unique(x)) 

print(dtlist)
values3=set(dtlist[0])
print(values3)
values2 = set([dt['A'][0],dt['B'][0],dt['C'][0],dt['D'][0]])
print(values2)
print(dt.A.unique())
print(type(dt.A.unique()))

print(isinstance(2,int)==2)
if 2:
   print('a')



testc=1

def g():
   global testc
   testc=testc+1

g()

print(testc)