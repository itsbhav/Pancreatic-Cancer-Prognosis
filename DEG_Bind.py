import pandas as pd

# Use the read_table function to read the tsv file
df1 = pd.read_table('C:/Users/bhave/OneDrive/Desktop/Minor/GSE_16515.tsv')
df2=pd.read_table('C:/Users/bhave/OneDrive/Desktop/Minor/DEG_15471.tsv')
# Display the updated TSV format file
df3=df1.iloc[:,0]
df4=df2.iloc[:,0]
comm=list(set(df3).intersection(set(df4)))
print(len(comm))
df_1=pd.read_csv("normalized_expressions_GSE15471.csv")
df_2=pd.read_csv("normalized_expressions_GSE16515.csv")

df_1_f=df_1[df_1['ID'].isin(comm)]
df_2_f=df_2[df_2['ID'].isin(comm)]

merged_df=pd.merge(df_1_f,df_2_f,on='ID',how="outer")
print(merged_df.shape)
merged_df.drop(columns=['Gene Symbol_y','Gene Symbol_x'],inplace=True)
# print(merged_df)
print(merged_df.shape)
merged_df.to_csv('testDEG.csv')