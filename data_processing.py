import pandas as pd
from seq_parser import AAIndex, CustomIndex


# getting the total_positive data
total_positive = pd.read_csv("total_positive.csv")
total_positive = list(total_positive.iloc[:, 2])
# print(total_positive)

total_positive_aaidx = AAIndex(total_positive)
total_positive_aaidx()
total_positive_aaidx.frame.to_csv("total_positive_aaidx.csv", index= True)
print("CSV file 'total_positive_aaidx.csv' written successfully.")



total_positive_cstmidx = CustomIndex(total_positive)
total_positive_cstmidx()
total_positive_cstmidx.frame.to_csv("total_positive_cstmidx.csv", index=True)
print("CSV file 'total_positive_cstmidx.csv' written successfully.")


# Merge both the files into single one

df1 = pd.read_csv("total_positive_aaidx.csv")
df2 = pd.read_csv("total_positive_cstmidx.csv")

df2_trimmed = df2.iloc[:, 1:]

# Merge column-wise
merged_df = pd.concat([df1, df2_trimmed], axis=1)
merged_df['Label'] = 1

# Save to new CSV
merged_df.to_csv("total_positive_merged.csv", index=False)
print("CSV files merged successfully into 'total_positive_merged.csv'")
merged_df_positive = merged_df





# total negative
# getting the total_negative data
total_negative = pd.read_csv("total_negative.csv")
total_negative = list(total_negative.iloc[:, 2])
# print(total_negative)

total_negative_aaidx = AAIndex(total_negative)
total_negative_aaidx()
total_negative_aaidx.frame.to_csv("total_negative_aaidx.csv", index= True)
print("CSV file 'total_negative_aaidx.csv' written successfully.")



total_negative_cstmidx = CustomIndex(total_negative)
total_negative_cstmidx()
total_negative_cstmidx.frame.to_csv("total_negative_cstmidx.csv", index=True)
print("CSV file 'total_negative_cstmidx.csv' written successfully.")


# Merge both the files into single one

df1 = pd.read_csv("total_negative_aaidx.csv")
df2 = pd.read_csv("total_negative_cstmidx.csv")

df2_trimmed = df2.iloc[:, 1:]

# Merge column-wise
merged_df = pd.concat([df1, df2_trimmed], axis=1)
merged_df['Label'] = 0

# Save to new CSV
merged_df.to_csv("total_negative_merged.csv", index=False)
print("CSV files merged successfully into 'total_negative_merged.csv'")


merged_df_total = pd.concat([merged_df_positive, merged_df.iloc[1:, :]], axis=0)
merged_df_total = merged_df_total.dropna()

merged_df_total.to_csv("total_merged.csv", index=False)
print("CSV files merged successfully into 'total_merged.csv'")

# # Generating fake data

# # getting the dengue data
# fake = pd.read_csv("fake.csv")
# fake = list(fake.iloc[:, 0])
# # print(dengue)

# print(fake)

# fake_aaidx = AAIndex(fake)
# fake_aaidx()
# fake_aaidx.frame.to_csv("fake_aaidx.csv", index= True)
# print("CSV file 'fake_aaidx.csv' written successfully.")



# fake_cstmidx = CustomIndex(fake)
# fake_cstmidx()
# fake_cstmidx.frame.to_csv("fake_cstmidx.csv", index=True)
# print("CSV file 'fake_cstmidx.csv' written successfully.")


# # Merge both the files into single one

# df1 = pd.read_csv("fake_aaidx.csv")
# df2 = pd.read_csv("fake_cstmidx.csv")

# df2_trimmed = df2.iloc[:, 1:]

# # Merge column-wise
# merged_df = pd.concat([df1, df2_trimmed], axis=1)

# # Save to new CSV
# merged_df.to_csv("fake_merged.csv", index=False)
# print("CSV files merged successfully into 'fake_merged.csv'")