import numpy as np
import pandas as pd
import libpysal as ps
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from mgwr.utils import compare_surfaces, truncate_colormap
import geopandas as gp
import matplotlib.pyplot as plt
import matplotlib as mpl


'''
#Step 1: Prepare Data

'''

#LOAD DATA
#needs to be shapefile with x & y centroid coordinates 
floodsovi = gp.read_file('shapefile/floodsvi.shp')


#Create plot of shapefile and centroids
fig, ax = plt.subplots(figsize = (10, 10))
floodsovi.plot(ax=ax, **{'edgecolor': 'black', 'facecolor': 'white'})
floodsovi.centroid.plot(ax = ax, c = 'black')
plt.savefig('floodsovi_shp')
#plt.show()

'''
#PREPARE DATASET INPUTS
'''

#uniqueID
FIPS = floodsovi['FIPS']
df_FIPS = pd.DataFrame(FIPS)

#Dependent Var: 
ar_y = floodsovi['logPAR_BL_'].values.reshape((-1,1)) #log of percent count of buildings in floodplain

#Explanatory Vars: 
ar_x = floodsovi[['EP_POV150', 'EP_UNEMP', 'EP_HBURD', 'EP_NOHSDP', 'EP_UNINSUR', 'EP_AGE65', 'EP_AGE17', 'EP_DISABL', 'EP_SNGPNT', 'EP_LIMENG', 'EP_MUNIT', 'EP_MOBILE', 'EP_CROWD', 'EP_NOVEH', 'EP_GROUPQ', 'EP_AFAM', 'EP_HISP', 'EP_ASIAN', 'EP_AIAN']].values

#Coordinates
u = floodsovi['x']
v = floodsovi['y']

a_coords = list(zip(u,v)) 

#Standardize Variables (mean of zero and variance of unity)
ar_x = (ar_x - ar_x.mean(axis = 0)) / ar_x.std(axis = 0)
ar_y = (ar_y - ar_y.mean(axis = 0)) / ar_y.std(axis = 0)


'''
#CALIBRATE GWR MODEL
'''

gwr_selector = Sel_BW(a_coords, ar_y, ar_x)
gwr_bw = gwr_selector.search()
print('gwr bandwidth:', gwr_bw)
gwr_model = GWR(a_coords, ar_y, ar_x, gwr_bw)
gwr_results = gwr_model.fit()

#Print AICc & R2
print('GWR AICc =', gwr_results.aicc)
print('GWR R2 = ', gwr_results.R2)

#Add GWR parameters to GeoDataFrame
floodsovi['gwr_intercept'] = gwr_results.params[:, 0]
floodsovi['gwr_POV150'] = gwr_results.params[:, 1] 
floodsovi['gwr_UNEMP'] = gwr_results.params[:, 2] 
floodsovi['gwr_HBURD'] = gwr_results.params[:, 3] 
floodsovi['gwr_NOHSDP'] = gwr_results.params[:, 4]
floodsovi['gwr_UNINSUR'] = gwr_results.params[:, 5] 
floodsovi['gwr_AGE65'] = gwr_results.params[:, 6] 
floodsovi['gwr_AGE17'] = gwr_results.params[:, 7] 
floodsovi['gwr_DISABL'] = gwr_results.params[:, 8] 
floodsovi['gwr_SNGPNT'] = gwr_results.params[:, 9] 
floodsovi['gwr_LIMENG'] = gwr_results.params[:, 10] 
floodsovi['gwr_MUNIT'] = gwr_results.params[:, 11] 
floodsovi['gwr_MOBILE'] = gwr_results.params[:, 12] 
floodsovi['gwr_CROWD'] = gwr_results.params[:, 13] 
floodsovi['gwr_NOVEH'] = gwr_results.params[:, 14] 
floodsovi['gwr_GROUPQ'] = gwr_results.params[:, 15] 
floodsovi['gwr_AFAM'] = gwr_results.params[:, 16] 
floodsovi['gwr_HISP'] = gwr_results.params[:, 17] 
floodsovi['gwr_ASIAN'] = gwr_results.params[:, 18] 
floodsovi['gwr_AIAN'] = gwr_results.params[:, 19] 

#Obtain t-vals filtered based on multiple testing correction
gwr_filtered_t = gwr_results.filter_tvals()

print('gwr results type:', type(gwr_results)) 
print('gwr results params type:', type(gwr_results.params)) 


#export to csv

#Create dataframes
df_gwr_results = pd.DataFrame(gwr_results.params) #dataframe for parameters
df_gwr_filtered_t  = pd.DataFrame(gwr_filtered_t)  #T-value pandas dataframe
df_gwr_filtered_t = df_gwr_filtered_t.add_prefix('T_')  #add T
df_gwr_results = pd.DataFrame(gwr_results.params)

gwr_result = pd.concat([FIPS, df_gwr_results,df_gwr_filtered_t], axis=1)
gwr_result .to_csv('gwr_results.csv')



#CALIIBRATE MGWR MODEL

mgwr_selector = Sel_BW(a_coords, ar_y, ar_x, multi = True)
mgwr_bw = mgwr_selector.search(multi_bw_min = [2])
print('MGWR bandwidths:', mgwr_bw)
mgwr_results = MGWR(a_coords, ar_y, ar_x, mgwr_selector).fit()

#Prepare MGWR results for mapping

#Add MGWR parameters to GeoDataframe
floodsovi['mgwr_intercept'] = mgwr_results.params[:, 0]
floodsovi['mgwr_POV150'] = mgwr_results.params[:, 1]  # % 150% Poverty 
floodsovi['mgwr_UNEMP'] = mgwr_results.params[:, 2] # % Unemployed
floodsovi['mgwr_HBURD'] = mgwr_results.params[:, 3]  # % Housing burdened
floodsovi['mgwr_NOHSDP'] = mgwr_results.params[:, 4] # % No High School Diploma
floodsovi['mgwr_UNINSUR'] = mgwr_results.params[:, 5]  
floodsovi['mgwr_AGE65'] = mgwr_results.params[:, 6] 
floodsovi['mgwr_AGE17'] = mgwr_results.params[:, 7] 
floodsovi['mgwr_DISABL'] = mgwr_results.params[:, 8] 
floodsovi['mgwr_SNGPNT'] = mgwr_results.params[:, 9] 
floodsovi['mgwr_LIMENG'] = mgwr_results.params[:, 10] 
floodsovi['mgwr_MUNIT'] = mgwr_results.params[:, 11] 
floodsovi['mgwr_MOBILE'] = mgwr_results.params[:, 12] 
floodsovi['mgwr_CROWD'] = mgwr_results.params[:, 13] 
floodsovi['mgwr_NOVEH'] = mgwr_results.params[:, 14] 
floodsovi['mgwr_GROUPQ'] = mgwr_results.params[:, 15] 
floodsovi['mgwr_AFAM'] = mgwr_results.params[:, 16] 
floodsovi['mgwr_HISP'] = mgwr_results.params[:, 17] 
floodsovi['mgwr_ASIAN'] = mgwr_results.params[:, 18] 
floodsovi['mgwr_AIAN'] = mgwr_results.params[:, 19] 

#Obtain t-vals filtered based on multiple testing correction
mgwr_filtered_t = mgwr_results.filter_tvals()


#Print AICc & R2
print('MGWR AICc =', mgwr_results.aicc)
print('MGWR R2 = ', mgwr_results.R2)

'''
#Export Results
'''

#Parameters and F-test to CSV

#Create dataframes
df_mgwr_results = pd.DataFrame(mgwr_results.params) #dataframe for parameters
df_mgwr_filtered_t  = pd.DataFrame(mgwr_filtered_t)  #T-value pandas dataframe
df_mgwr_filtered_t = df_mgwr_filtered_t.add_prefix('T_')  #add T
df_mgwr_results = pd.DataFrame(mgwr_results.params)

mgwr_result = pd.concat([FIPS, df_mgwr_results,df_mgwr_filtered_t], axis=1)
mgwr_result .to_csv('mgwr_results.csv')

#GWR & MGWR AICc & R2

with open('modelresults.txt', 'a') as f:
    f.write('GWR AICc' + '\n')
    f.write(str(gwr_results.aicc) + '\n')
    f.write('GWR R2' + '\n')
    f.write(str(gwr_results.R2) + '\n')
    f.write('GWR Bandwidths' + '\n')
    f.write(str(gwr_bw) + '\n')
    f.write('\n')
    f.write('MGWR AICc' + '\n')
    f.write(str(mgwr_results.aicc) + '\n')
    f.write('GWR R2' + '\n')
    f.write(str(mgwr_results.R2) + '\n')
    f.write('MGWR Bandwidths' + '\n')
    f.write(str(mgwr_bw) + '\n')
    
'''
#Compare GWR & MGWR Surfaces
'''


kwargs1 = {'edgecolor': 'black', 'alpha': .65}
kwargs2 = {'edgecolor': 'black'}

#Intercept
compare_surfaces(floodsovi, 'gwr_intercept', 'mgwr_intercept',
    gwr_filtered_t[:, 0], gwr_bw, mgwr_filtered_t[:, 0],
    mgwr_bw[0], 'Intercept', kwargs1, kwargs2,
    savefig = '0Intercept')

#EP_POV150 
compare_surfaces(floodsovi, 'gwr_POV150', 'mgwr_POV150', gwr_filtered_t[:, 1],
    gwr_bw, mgwr_filtered_t[:, 1], mgwr_bw[1],
    '% Over 150% Poverty Line', kwargs1, kwargs2, savefig = 'POV150')

#EP_UNEMP
compare_surfaces(floodsovi, 'gwr_UNEMP', 'mgwr_UNEMP', gwr_filtered_t[:, 2],
    gwr_bw, mgwr_filtered_t[:, 2], mgwr_bw[2],
    '% Unemployment', kwargs1, kwargs2, savefig = 'UNEMP')

#EP_HBURD
compare_surfaces(floodsovi, 'gwr_HBURD', 'mgwr_HBURD', gwr_filtered_t[:, 3],
    gwr_bw, mgwr_filtered_t[:, 3], mgwr_bw[3],
    '% Housing Burdened', kwargs1, kwargs2, savefig = 'HBURD')

#EP_NOHSDP
compare_surfaces(floodsovi, 'gwr_NOHSDP', 'mgwr_NOHSDP', gwr_filtered_t[:, 4],
    gwr_bw, mgwr_filtered_t[:, 4], mgwr_bw[4],
    '% No High School Diploma', kwargs1, kwargs2, savefig = 'NOHSDP')

#EP_UNINSUR
compare_surfaces(floodsovi, 'gwr_UNINSUR', 'mgwr_UNINSUR', gwr_filtered_t[:, 5],
    gwr_bw, mgwr_filtered_t[:, 5], mgwr_bw[5],
    '% Uninsured', kwargs1, kwargs2, savefig = 'UNINSUR')

#EP_AGE65
compare_surfaces(floodsovi, 'gwr_AGE65', 'mgwr_AGE65', gwr_filtered_t[:, 6],
    gwr_bw, mgwr_filtered_t[:, 6], mgwr_bw[6],
    '% Age 65', kwargs1, kwargs2, savefig = 'AGE65')

#EP_AGE17
compare_surfaces(floodsovi, 'gwr_AGE17', 'mgwr_AGE17', gwr_filtered_t[:, 7],
    gwr_bw, mgwr_filtered_t[:, 7], mgwr_bw[7],
    '% Age 17', kwargs1, kwargs2, savefig = 'AGE17')

#EP_DISABL
compare_surfaces(floodsovi, 'gwr_DISABL', 'mgwr_DISABL', gwr_filtered_t[:, 8],
    gwr_bw, mgwr_filtered_t[:, 8], mgwr_bw[8],
    '% Disability', kwargs1, kwargs2, savefig = 'DISABL')

#EP_SNGPNT
compare_surfaces(floodsovi, 'gwr_SNGPNT', 'mgwr_SNGPNT', gwr_filtered_t[:, 9],
    gwr_bw, mgwr_filtered_t[:, 9], mgwr_bw[9],
    '% Single Parent', kwargs1, kwargs2, savefig = 'SNGPNT')

#EP_LIMENG
compare_surfaces(floodsovi, 'gwr_LIMENG', 'mgwr_LIMENG', gwr_filtered_t[:, 10],
    gwr_bw, mgwr_filtered_t[:, 10], mgwr_bw[10],
    'Limited English, %', kwargs1, kwargs2, savefig = 'LIMENG')

#EP_MUNIT
compare_surfaces(floodsovi, 'gwr_MUNIT', 'mgwr_MUNIT', gwr_filtered_t[:, 11],
    gwr_bw, mgwr_filtered_t[:, 11], mgwr_bw[11],
    '% Multi-Unit Housing', kwargs1, kwargs2, savefig = 'MUNIT')

#EP_MOBILE
compare_surfaces(floodsovi, 'gwr_MOBILE', 'mgwr_MOBILE', gwr_filtered_t[:, 12],
    gwr_bw, mgwr_filtered_t[:, 12], mgwr_bw[12],
    '% Mobile Homes', kwargs1, kwargs2, savefig = 'MOBILE')

#EP_CROWD
compare_surfaces(floodsovi, 'gwr_CROWD', 'mgwr_CROWD', gwr_filtered_t[:, 13],
    gwr_bw, mgwr_filtered_t[:, 13], mgwr_bw[13],
    '% Crowded Housing', kwargs1, kwargs2, savefig = 'CROWD')

#EP_NOVEH 
compare_surfaces(floodsovi, 'gwr_NOVEH', 'mgwr_NOVEH', gwr_filtered_t[:, 14],
    gwr_bw, mgwr_filtered_t[:, 14], mgwr_bw[14],
    '% No Vehicle', kwargs1, kwargs2, savefig = 'NOVEH')

#EP_GROUPQ
compare_surfaces(floodsovi, 'gwr_GROUPQ', 'mgwr_GROUPQ', gwr_filtered_t[:, 15],
    gwr_bw, mgwr_filtered_t[:, 15], mgwr_bw[15],
    'Group Quarters, %', kwargs1, kwargs2, savefig = 'GROUPQ')

#EP_AFAM
compare_surfaces(floodsovi, 'gwr_AFAM', 'mgwr_AFAM', gwr_filtered_t[:, 16],
    gwr_bw, mgwr_filtered_t[:, 16], mgwr_bw[16],
    '% African American', kwargs1, kwargs2, savefig = 'AFAM')

#EP_HISP
compare_surfaces(floodsovi, 'gwr_HISP', 'mgwr_HISP', gwr_filtered_t[:, 17],
    gwr_bw, mgwr_filtered_t[:, 17], mgwr_bw[17],
    '% Hispanic', kwargs1, kwargs2, savefig = 'HISP')

#EP_ASIAN 
compare_surfaces(floodsovi, 'gwr_ASIAN', 'mgwr_ASIAN', gwr_filtered_t[:, 18],
    gwr_bw, mgwr_filtered_t[:, 18], mgwr_bw[18],
    '% Asian', kwargs1, kwargs2, savefig = 'ASIAN')

#EP_AIAN 
compare_surfaces(floodsovi, 'gwr_AIAN', 'mgwr_AIAN', gwr_filtered_t[:, 19],
    gwr_bw, mgwr_filtered_t[:, 19], mgwr_bw[19],
    '% American Indian/Alaskan Native', kwargs1, kwargs2, savefig = 'AIAN')
