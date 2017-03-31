
from collections import defaultdict
from sklearn.preprocessing import scale
import pandas as pd
import csv
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
from hcluster import pdist, linkage, dendrogram, squareform
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
from scipy.spatial import distance
from scipy.cluster import hierarchy

from matplotlib.colors import rgb2hex, colorConverter
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette


#all columns 
master = pd.read_csv('../datasets/merged_2_noblanks_2.csv')
#all columns with objects except ID is removed
master_2 = pd.read_csv('../datasets/new.csv')

#turn objects to float for standardization
num_sex = {"FEMALE":1 ,"MALE" :0}
num_hand = {"LEFT":0,"RIGHT":1,"AMBIDEXTEROUS":2}

master['Sex_Num'] = master['Sex'].apply(num_sex.get).astype(float)
master['Handedness_Num'] = master['Handedness'].apply(num_hand.get).astype(float)

#columns that are objects like subject id, sex, etc. must be removed in order to normalize the data
remove_objects = master.select_dtypes(exclude = ['object'])
keep_objects = master_2.select_dtypes(include = ['object'])


# In[31]:

#data is standardized by column (axis = 0) and then by row (axis = 1 )
full_merge_scale = scale(remove_objects)
full_merge_scale_2 = scale(full_merge_scale, axis= 1)

#column names are removed during scale, so they are added back in
full_merge_scale_2_df= pd.DataFrame(full_merge_scale_2, columns = remove_objects.columns)

#put back the columns that are objects 
double_standardized_master = keep_objects.join(full_merge_scale_2_df, how = 'right')

#rotated dataset
rotate_rows = full_merge_scale_2_df.transpose()

#reindex master to have ID's as labels 
df = double_standardized_master.set_index('Anonymized_ID')

#distance between datapoints
row_linkage = hierarchy.linkage(distance.pdist(rotate_rows.T, 'correlation'), method='ward')

col_linkage = hierarchy.linkage(distance.pdist(df.T, 'correlation'), method='ward')

#seaborn cluster map 
cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True)

result = sns.clustermap(df, row_linkage=row_linkage, col_linkage=col_linkage, method="ward", 
                        figsize=(20, 20), cmap=cmap)

plt.setp(result.ax_heatmap.get_yticklabels(), rotation=0)

plt.savefig('correlation_map.png')



#row dendrogram plot and group extraction (cluster_classes)
den = scipy.cluster.hierarchy.dendrogram(result.dendrogram_row.linkage,
                                         labels = df.index, color_threshold = 3.5)
plt.savefig('correlation_row.png')


sns.set_palette('Set1', 10, 0.65)
palette = sns.color_palette()
set_link_color_palette(map(rgb2hex, palette))
sns.set_style('white')


class Clusters(dict):
    def _repr_html_(self):
        html = '<table style="border: 0;">'
        for c in self:
            hx = rgb2hex(colorConverter.to_rgb(c))
            html += '<tr style="border: 0;">'             '<td style="background-color: {0}; '                        'border: 0;">'             '<code style="background-color: {0};">'.format(hx)
            html += c + '</code></td>'
            html += '<td style="border: 0"><code>' 
            html += repr(self[c]) + '</code>'
            html += '</td></tr>'
        
        html += '</table>'
        
        return html


def get_cluster_classes(den, label='ivl'):
    cluster_idxs = defaultdict(list)
    for c, pi in zip(den['color_list'], den['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))
    
    cluster_classes = Clusters() 
    for c, l in cluster_idxs.items():
        i_l = [den[label][i] for i in l]
        cluster_classes[c] = i_l
    
    return cluster_classes

get_cluster_classes(den)


#column dendrogram plot and group extraction (cluster_classes)
den_2 = scipy.cluster.hierarchy.dendrogram(result.dendrogram_col.linkage,
                                           labels = rotate_rows.index, color_threshold = 2)

plt.savefig('correlation_col.png')


def get_cluster_classes(den_2, label='ivl'):
    cluster_idxs = defaultdict(list)
    for c, pi in zip(den_2['color_list'], den_2['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))
    
    cluster_classes = Clusters() 
    for c, l in cluster_idxs.items():
        i_l = [den_2[label][i] for i in l]
        cluster_classes[c] = i_l
    
    return cluster_classes


get_cluster_classes(den_2)




