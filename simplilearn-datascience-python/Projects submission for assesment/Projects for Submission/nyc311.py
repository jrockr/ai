import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from pylab import subplot

raw_nyc_311_data = pd.read_csv("C:\\Users\\C740129\\OneDrive - Standard Bank\\certification\\Ai Engineer\\Data Science with Pandas C-1\\Projects submission for assesment\\Projects for Submission\\Project3_NYC311\\311-service-requests-nyc\\311_Service_Requests_from_2010_to_Present.csv",delimiter=",", low_memory=False)

print(raw_nyc_311_data.head())

#	Explore data

#   Find patterns

#   Display the complaint type and city together

#	Find the top 10 complaint types 
top_compaint_type = raw_nyc_311_data.groupby(['Complaint Type']).size().reset_index(name='count')

top_ten_complaint = top_compaint_type.sort_values(by=['count'], ascending=False).head(10)

print(top_ten_complaint)

print("*********")

print(top_ten_complaint['Complaint Type'])

# Plot a bar graph of count vs. complaint types


# Display the major complaint types and their count

plt.bar(top_ten_complaint['Complaint Type'],top_ten_complaint['count'],align="center",alpha=0.5)

plt.title('Count vs Complaint')

plt.show()

