import pandas as pd
df=pd.read_csv("normalized_expressions_GSE32676.csv")
matchi=["X1553102_a_at", "X1555136_at" , "X1555137_a_at",
 "X201417_at"   , "X202267_at"   , "X202286_s_at" ,
 "X202504_at"  ,  "X202935_s_at" , "X203476_at"   ,
 "X203510_at"  ,  "X203757_s_at" , "X204320_at"   ,
 "X204351_at"  ,  "X204602_at"  ,  "X204713_s_at" ,
 "X205941_s_at" , "X209016_s_at" , "X211719_x_at" ,
 "X212354_at"   , "X212444_at" ,   "X212464_s_at" ,
 "X216442_x_at" , "X218856_at" ,   "X219901_at"   ,
 "X226237_at"   , "X227051_at" ,   "X228923_at"   ,
 "X229479_at"   , "X230831_at" ,   "X241137_at"   ,
 "X242397_at"   ]
matc=[]
for i in matchi:
    matc.append(i.lstrip("X"))
print(matc)
filtered_df = df[df['ID'].isin(matc)]
print(df['ID'])
# Print the filtered DataFrame
print(filtered_df)

filtered_df.to_csv("test_data_new.csv")