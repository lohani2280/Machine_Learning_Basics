import pandas as pd

df = pd.read_csv('submission_xgb_500.csv')
# print(df.head(5))

df_final_status = df['final_status']
df_project_id = df['project_id']

df['final_status'] = df_project_id
df['project_id'] = df_final_status

df=df.rename(columns = {'final_status':'project_id' , 'project_id':'final_status'})

print(df.head(5))

result = []
for _,row in df.iterrows():
	if row['final_status'] > 0.4:
		result.append(1)
	else:
		result.append(0)

df.drop('final_status',1,inplace = True)
df['final_status'] = pd.Series(result)			

print(df.head(5))

df.to_csv("submit.csv", index=False)
