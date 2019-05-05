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
dt=df.fillna(0)
values = set(dt['one'])
print('val',values)
df = pd.DataFrame([[np.nan, 2, np.nan, 0],
                    [3, 4, np.nan, 1],
                    [np.nan, np.nan, np.nan, 5],
                    [np.nan, 3, np.nan, 4]],
                 columns=list('ABCD'))

print(df)

# dt=df.fillna('ss')

# print(dt)

values = set(df['A'])
print(values)

