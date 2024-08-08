# %%
import warnings
warnings.filterwarnings("ignore")

# %%
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

food = pd.read_csv("/Users/viktorijakesaite/Desktop/CAMBRIDGE POSTDOC/UPF/data files/foodleveldietarydatauk_new.csv", encoding='latin1')

# %%
def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they 
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance), len (values))

# %% [markdown]
# Removing missing values: person day missing observations

# %%
def list_wo_nan (l):
    result_0 = []
    result_1 = []
    for y in l:
        if y[1] == y[1]:
            result_0.append(y[0])
            result_1.append(y[1])
        else:
            print ('NaN')
    return result_0, result_1

# %%
food2 = pd.read_csv ("/Users/viktorijakesaite/Desktop/CAMBRIDGE POSTDOC/UPF/reduced_dataset.csv", encoding='latin1')

# %% [markdown]
# Merge food level and individual level datasets

# %%
food_final = food.merge(food2, left_on='seriali', right_on='seriali', how = 'left')

# %%
food_final.head(5)

# %% [markdown]
# Exclude alcohol and supplements

# %%
categories= {'47A', '47B','48A', '48B', '48C', '49A', '49B', '49C', '49D', '49E', '54A', '54B', '54C', '54D', '54E', '54F', '54G', '54H', '54I', '54J', '54K', '54L', '54M', '54N', '54P'}
food_noalcnovit=food_final[~food_final['SubFoodGroupCode'].isin(categories)]

# %% [markdown]
# Create vegetable and fruit purees -- based on 'FoodName' variable

# %%
veg_pureed={'BROCOLLI SPEARS FRESH BOILED PUREED','CABBAGE-WINTER KALE FRESH BOILED (PUREED)',
            'CARROT OLD RAW PUREED','CARROTS OLD, FRESH, BOILED PUREED', 'CARROTS YOUNG FRESH BOILED PUREED', 'CARROTS YOUNG FRESH RAW PUREED',  
            'CAULIFLOWER-FRESH BOILED (PUREED)', 'CELERIAC FRESH BOILED (PUREED)', 'CELERY FRESH BOILED PUREED',  'CELERY, FRESH RAW (PUREED)',
            'COURGETTE-BOILED (PUREED)','GARLIC PUREE','GUACAMOLE PURCHASED', 'GUACAMOLE REDUCED FAT PURCHASED',
            'HARICOT BEANS, NO ADDED SALT, CANNED AND DRAINED, COOKED (PUREED)','HUMMUS, NOT CANNED', 'HUMMUS/HOUMOUS, LOW/REDUCED FAT',
            'LEEKS FRESH BOILED (PUREED)', 'LENTILS SPLIT BOILED (PUREED)', 'MUSHROOMS STEWED OR GRILLED (PUREED)',
            'ONIONS BOILED PUREED', 'PEAS FROZEN BOILED (PUREED) FF PROJECT', 'PEAS SPLIT DRIED BOILED (PUREED)', 
            'PEAS-CHICK, NO ADDED SALT OR SUGAR, CANNED AND DRAINED , COOKED (PUREED)', 'PEPPERS GREEN BOILED (PUREED)', 
            'PEPPERS YELLOW FRESH BOILED (PUREED)',  'PUMPKIN, PUREED, BOILED', 'SPINACH RAW NOT BABY SPINACH (PUREED)', 'SPINACH FRESH BOILED (PUREED)', 
            'SWEDE BOILED (PUREED)', 'SWEETCORN, CANNED, DRAINED, NON ADDED SUGAR OR SALT (PUREED) FS PROJECT', 'TURNIPS-BOILED (PUREED)'}

# %%
fruit_pureed={'GOJI BERRIES / WOLFBERRIES DRIED (PUREED)','MELON PUREE, HOMEMADE, 100% FRESH RAW FRUIT NAS','STRAWBERRIES STEWED WITHOUT SUGAR (PUREED)', 
              'PURE FRUIT PUREES NAS ANY FLAVOUR READY TO EAT (ADDED VIT C 0-10MG)', 
              'PURE FRUIT PUREES NAS ANY FLAVOUR READY TO EAT (ADDED VIT C 10-20MG)',
              'PURE FRUIT PUREES NAS ANY FLAVOUR READY TO EAT (ADDED VIT C OVER 20MG)', 'GRAPEFRUIT RAW, PUREED, FLESH ONLY NO PEEL OR PIPS', 
              'KIWI PUREE, HOMEMADE, 100% FRESH RAW FRUIT NAS', 'NECTARINE PUREE, HOMEMADE, 100% FRESH RAW FRUIT NAS', 
              'PASSION FRUIT RAW FLESH & SEEDS ONLY (PUREED)', 'RASPBERRY PUREE, HOMEMADE, 100% FRESH RAW FRUIT NAS'}

# %%
food_noalcnovit.loc[:, 'veg_pureed']  = 0
food_noalcnovit.loc[:, 'fruit_pureed']  = 0
food_noalcnovit.loc[food_noalcnovit['FoodName'].isin(veg_pureed), 'veg_pureed'] = food_noalcnovit.loc[food_noalcnovit['FoodName'].isin(veg_pureed), 'TotalGrams']
food_noalcnovit.loc[food_noalcnovit['FoodName'].isin(fruit_pureed), 'fruit_pureed'] = food_noalcnovit.loc[food_noalcnovit['FoodName'].isin(fruit_pureed), 'TotalGrams']

# %% [markdown]
# Calculate vegetable variable

# %%
food_noalcnovit.loc [:, "vegetable"] = food_noalcnovit["Tomatoesg"]+ food_noalcnovit["Brassicaceaeg"]+food_noalcnovit["YellowRedGreeng"]+food_noalcnovit["Beansg"]+food_noalcnovit["OtherVegg"]

# %% [markdown]
# Calculate fruit variable

# %%
food_noalcnovit.loc [:, "Fruitjuice_per_100_g"] = food_noalcnovit["FruitJuiceg"]/food_noalcnovit["TotalGrams"]*100

# %% [markdown]
# Calculate fruit, veg, and nuts per 100g 

# %%
food_noalcnovit.loc [:,"fvn_per_100_g"] = (food_noalcnovit["Fruitg"] + food_noalcnovit["vegetable"]+ food_noalcnovit["Nutsg"]+ food_noalcnovit['TomatoPureeg']*2+ food_noalcnovit['veg_pureed']*2+ food_noalcnovit['fruit_pureed']*2+food_noalcnovit['DriedFruitg']*2)/(food_noalcnovit["TotalGrams"]+ food_noalcnovit['TomatoPureeg'] + food_noalcnovit['veg_pureed']+ food_noalcnovit['fruit_pureed']+ food_noalcnovit['DriedFruitg'])*100

# %% [markdown]
# Calculate the amount per 100g

# %%
food_noalcnovit.loc [:, "Fruitjuice_per_100_g"] = food_noalcnovit["FruitJuiceg"]/food_noalcnovit["TotalGrams"]*100

# %%
food_noalcnovit.loc [:, "Smoothies_per_100_g"] = food_noalcnovit["SmoothieFruitg"]/food_noalcnovit["TotalGrams"]*100

# %%
food_noalcnovit.loc [:, "EnergykJ_per_100_g"] = food_noalcnovit["EnergykJ"]/food_noalcnovit["TotalGrams"]*100

# %%
food_noalcnovit.loc [:, "Saturatedfattyacidsg_per_100_g"] = food_noalcnovit["Saturatedfattyacidsg"]/food_noalcnovit["TotalGrams"]*100

# %%
food_noalcnovit.loc [:, "Totalsugarsg_per_100_g"] = food_noalcnovit["Totalsugarsg"]/food_noalcnovit["TotalGrams"]*100

# %%
food_noalcnovit.loc [:, "Sodiummg_per_100_g"] = food_noalcnovit["Sodiummg"]/food_noalcnovit["TotalGrams"]*100

# %%
food_noalcnovit.loc [:, "AOACFibreg_per_100_g"] = food_noalcnovit["AOACFibreg"]/food_noalcnovit["TotalGrams"]*100

# %%
food_noalcnovit.loc [:, "Proteing_per_100_g"] = food_noalcnovit["Proteing"]/food_noalcnovit["TotalGrams"]*100

# %%
food_noalcnovit.NOVANumNEW_agreement.value_counts(normalize=True)*100

# %% [markdown]
# Create a binary variable equalling one if UPF and zero otherwise

# %%
food_noalcnovit.loc[:, 'NOVA_UPF_NUPF_boolean'] = (food_noalcnovit['NOVANumNEW_agreement'] == 'Ultra-processed food and drink products')

# %% [markdown]
# Classify HFSS foods using NDNS food database based on the NPM 2004/5 version 

# %% [markdown]
# A POINTS

# %%
energy_points = {0: [-1, 335], 1: [335, 670], 2: [670, 1005], 3: [1005, 1340], 
                 4: [1340, 1675], 5: [1675, 2010], 6: [2010, 2345], 7: [2345, 2680],
                 8: [2680, 3015], 9: [3015, 3350], 10: [3350]}
food_noalcnovit.loc [:, 'energy_a_points'] = np.array(len(food_noalcnovit) * [0])
for points in energy_points:
    interval = energy_points[points]
    if len(interval) == 2:
        food_noalcnovit.loc[(food_noalcnovit['EnergykJ_per_100_g'] > interval[0]) & (food_noalcnovit['EnergykJ_per_100_g'] <= interval[1]),'energy_a_points'] = points
    elif len(interval) == 1:
        food_noalcnovit.loc[(food_noalcnovit['EnergykJ_per_100_g'] > interval[0]), 'energy_a_points'] = points

# %%
satfat_points = {0: [-1, 1], 1: [1, 2], 2: [2, 3], 3: [3, 4], 
                 4: [4, 5], 5: [5, 6], 6: [6, 7], 7: [7, 8],
                 8: [8, 9], 9: [9, 10], 10: [10]}
food_noalcnovit.loc[:, 'satfat_a_points'] = np.array(len(food_noalcnovit) * [0])
for points in satfat_points:
    interval = satfat_points[points]
    if len(interval) == 2:
        food_noalcnovit.loc[(food_noalcnovit['Saturatedfattyacidsg_per_100_g'] > interval[0]) & (food_noalcnovit['Saturatedfattyacidsg_per_100_g'] <= interval[1]),'satfat_a_points'] = points
    elif len(interval) == 1:
        food_noalcnovit.loc[(food_noalcnovit['Saturatedfattyacidsg_per_100_g'] > interval[0]), 'satfat_a_points'] = points

# %%
tsugar_points = {0: [-1, 4.5], 1: [4.5, 9], 2: [9, 13.5], 3: [13.5, 18], 
                 4: [18, 22.5], 5: [22.5, 27], 6: [27, 31], 7: [31, 36],
                 8: [36, 40], 9: [40, 45], 10: [45]}
food_noalcnovit.loc[:, 'tsugar_a_points'] = np.array(len(food_noalcnovit) * [0])
for points in tsugar_points:
    interval = tsugar_points[points]
    if len(interval) == 2:
        food_noalcnovit.loc[(food_noalcnovit['Totalsugarsg_per_100_g'] > interval[0]) & (food_noalcnovit['Totalsugarsg_per_100_g'] <= interval[1]),'tsugar_a_points'] = points
    elif len(interval) == 1:
        food_noalcnovit.loc[(food_noalcnovit['Totalsugarsg_per_100_g'] > interval[0]), 'tsugar_a_points'] = points

# %%
sodium_points = {0: [-1, 90], 1: [90, 180], 2: [180, 270], 3: [270, 360], 
                 4: [360, 450], 5: [450, 540], 6: [540, 630], 7: [630, 720],
                 8: [720, 810], 9: [810, 900], 10: [900]}
food_noalcnovit.loc[:, 'sodium_a_points'] = np.array(len(food_noalcnovit) * [0])
for points in sodium_points:
    interval = sodium_points[points]
    if len(interval) == 2:
        food_noalcnovit.loc[(food_noalcnovit['Sodiummg_per_100_g'] > interval[0]) & (food_noalcnovit['Sodiummg_per_100_g'] <= interval[1]),'sodium_a_points'] = points
    elif len(interval) == 1:
        food_noalcnovit.loc[(food_noalcnovit['Sodiummg_per_100_g'] > interval[0]), 'sodium_a_points'] = points

# %% [markdown]
# Calculate total A points

# %%
food_noalcnovit.loc[:, 'total_a_points'] = food_noalcnovit["energy_a_points"] + food_noalcnovit["satfat_a_points"] + food_noalcnovit["tsugar_a_points"] + food_noalcnovit["sodium_a_points"] 

# %% [markdown]
# C POINTS

# %%
smoothies = {0: [-1, 40], 1: [40, 60], 2: [60, 80], 5: [80]}
food_noalcnovit.loc [:, 'smoothies'] = np.array(len(food_noalcnovit) * [0])
for points in smoothies:
    interval = smoothies[points]
    if len(interval) == 2:
        food_noalcnovit.loc[(food_noalcnovit['Smoothies_per_100_g'] > interval[0]) & (food_noalcnovit['Smoothies_per_100_g'] <= interval[1]),'smoothies'] = points
    elif len(interval) == 1:
        food_noalcnovit.loc[(food_noalcnovit['Smoothies_per_100_g'] > interval[0]), 'smoothies'] = points

# %%
fruit_juices = {0: [-1, 40], 1: [40, 60], 2: [60, 80], 5: [80]}
food_noalcnovit.loc [:, 'fruit_juice'] = np.array(len(food_noalcnovit) * [0])
for points in fruit_juices:
    interval = fruit_juices[points]
    if len(interval) == 2:
        food_noalcnovit.loc[(food_noalcnovit['Fruitjuice_per_100_g'] > interval[0]) & (food_noalcnovit['Fruitjuice_per_100_g'] <= interval[1]),'fruit_juice'] = points
    elif len(interval) == 1:
        food_noalcnovit.loc[(food_noalcnovit['Fruitjuice_per_100_g'] > interval[0]), 'fruit_juice'] = points

# %%
fvn_points = {0: [-1, 40], 1: [40, 60], 2: [60, 80], 5: [80]} 
food_noalcnovit.loc[:, 'fvn_c_points'] = np.array(len(food_noalcnovit) * [0])
for points in fvn_points:
    interval = fvn_points[points]
    if len(interval) == 2:
        food_noalcnovit.loc[(food_noalcnovit['fvn_per_100_g'] > interval[0]) & (food_noalcnovit['fvn_per_100_g'] <= interval[1]),'fvn_c_points'] = points
    elif len(interval) == 1:
        food_noalcnovit.loc[(food_noalcnovit['fvn_per_100_g'] > interval[0]), 'fvn_c_points'] = points

# %%
aoacfibre_points = {0: [-1, 0.9], 1: [0.9, 1.9], 2: [1.9, 2.8], 3: [2.8, 3.7], 4: [3.7, 4.7], 5: [4.7]} 

food_noalcnovit.loc[:, 'aoacfibre_c_points'] = np.array(len(food_noalcnovit) * [0])
for points in aoacfibre_points:
    interval = aoacfibre_points[points]
    if len(interval) == 2:
        food_noalcnovit.loc[(food_noalcnovit['AOACFibreg_per_100_g'] > interval[0]) & (food_noalcnovit['AOACFibreg_per_100_g'] <= interval[1]),'aoacfibre_c_points'] = points
    elif len(interval) == 1:
        food_noalcnovit.loc[(food_noalcnovit['AOACFibreg_per_100_g'] > interval[0]), 'aoacfibre_c_points'] = points

# %% [markdown]
# If a food or drink scores 11 or more ‘A’ points then it cannot score points for protein unless it also scores 5 points for fruit, vegetables and nuts.

# %%
protein_points = {0: [-1, 1.6], 1: [1.6, 3.2], 2: [3.2, 4.8], 3: [4.8, 6.4], 4: [6.4, 8.0], 5: [8.0]} 

food_noalcnovit.loc[:, 'protein_c_points'] = np.array(len(food_noalcnovit) * [0])
for points in protein_points:
    interval = protein_points[points]
    if len(interval) == 2:
        food_noalcnovit.loc[(food_noalcnovit['Proteing_per_100_g'] > interval[0]) & (food_noalcnovit['Proteing_per_100_g'] <= interval[1])  & (food_noalcnovit['total_a_points'] < 11),'protein_c_points'] = points
        food_noalcnovit.loc[(food_noalcnovit['Proteing_per_100_g'] > interval[0]) & (food_noalcnovit['Proteing_per_100_g'] <= interval[1])  & (food_noalcnovit['total_a_points'] >= 11) & (food_noalcnovit['fvn_c_points'] >= 5),'protein_c_points'] = points
    elif len(interval) == 1:
        food_noalcnovit.loc[(food_noalcnovit['Proteing_per_100_g'] > interval[0])  & (food_noalcnovit['total_a_points'] < 11), 'protein_c_points'] = points
        food_noalcnovit.loc[(food_noalcnovit['Proteing_per_100_g'] > interval[0])  & (food_noalcnovit['total_a_points'] >= 11) & (food_noalcnovit['fvn_c_points'] >= 5), 'protein_c_points'] = points

# %% [markdown]
# Calculate total C POINTS

# %%
food_noalcnovit.loc[:, 'total_c_points'] = food_noalcnovit["fvn_c_points"] + food_noalcnovit["aoacfibre_c_points"] + food_noalcnovit["protein_c_points"] +food_noalcnovit['fruit_juice']+ food_noalcnovit['smoothies']

# %% [markdown]
# TOTAL POINTS

# %%
food_noalcnovit.loc[:, "total_points"] = food_noalcnovit["total_a_points"]-food_noalcnovit ["total_c_points"]

# %%
food_noalcnovit.loc[:, "lesshealthy"] = food_noalcnovit["total_points"] >= 4 

# %% [markdown]
# Create a drink set - include all drink names (Remove fruit juice and smoothies since they are calculated elsewhere)

# %%
drink_set={'SEMI SKIMMED MILK','SKIMMED MILK', 'SOFT DRINKS NOT LOW CALORIE', 'SOFT DRINKS LOW CALORIE', 'COMMERCIAL TODDLERS FOODS AND DRINKS', 
       '1% Fat Milk', 'TEA COFFEE AND WATER', 'WHOLE MILK'} 

# %%
drink_set2={'ACTIMEL PROBIOTIC DRINKING YOGURT',
'ACTIMEL PROBIOTIC YOGURT DRINK 0.1% FAT',
'BENECOL PLUS HEART YOGURT DRINK', 'BENECOL YOGURT DRINKS',
'BENECOL YOGURT DRINKS, CONTAINING STANOLS',
'CHILDRENS YOGURT DRINK WITH OMEGA 3, CALCIUM AND VITAMIN D',
'CHILDRENS YOGURT DRINK WITH VITAMIN D',
'MULLER VITALITY PROBIOTIC DRINK',
'MULLER VITALITY PROBIOTIC DRINK WITH OMEGA 3',
'OPTIFIT YOGURT DRINKS, ANY FLAVOUR, FORTIFIED WITH VITS C, E, B6',
'SMOOTHIES WITH FRUIT AND DAIRY PRODUCTS, BOTTLED, PURCHASED, NOT FORTIFIED',
'YAKULT', 'YOGURT DRINK', 'YOGURT DRINK CONTAINING FRUIT PUREE',
'YOGURT DRINK FORTIFIED WITH VITAMINS B6, C AND D'}

# %%
food_noalcnovit.loc[food_noalcnovit['MainFoodGroupDesc'].isin(drink_set), 'lesshealthy'] = food_noalcnovit.loc[food_noalcnovit['MainFoodGroupDesc'].isin(drink_set), "total_points"]>= 1 

# %%
food_noalcnovit.loc[food_noalcnovit['FoodName'].isin(drink_set2), 'lesshealthy'] = food_noalcnovit.loc[food_noalcnovit['FoodName'].isin(drink_set2), "total_points"]>= 1 

# %%
np.sum(food_noalcnovit["lesshealthy"])/len(food_noalcnovit["lesshealthy"])

# %%
np.sum(food_noalcnovit["NOVA_UPF_NUPF_boolean"])/len(food_noalcnovit["NOVA_UPF_NUPF_boolean"])

# %%
food_noalcnovit_upf = food_noalcnovit[food_noalcnovit['NOVA_UPF_NUPF_boolean']]

# %% [markdown]
# Check the proportion of HFSS if added artificial sweeteners and soft drinks, low calorie

# %%
food_noalcnovit_upf.loc[:,"LH_assd"] = (food_noalcnovit_upf["lesshealthy"])| (food_noalcnovit_upf['MainFoodGroupDesc'] == "ARTIFICIAL SWEETENERS")|(food_noalcnovit_upf['MainFoodGroupDesc'] == "SOFT DRINKS LOW CALORIE")

# %% [markdown]
# Check the proportion of HFSS if added artificial sweeteners and soft drinks, low calorie

# %%
np.sum(food_noalcnovit_upf["LH_assd"] )/len(food_noalcnovit_upf.loc[:,"LH_assd"] )

# %% [markdown]
# Proportion of energy derived from HFSS+ARTIFICIAL SWEETENERS AND SOFT DRINKS, LOW CALORIE

# %%
np.sum(food_noalcnovit_upf["LH_assd"]* food_noalcnovit_upf['Energykcal'])/np.sum(food_noalcnovit_upf['Energykcal'])

# %% [markdown]
# Proportion of energy derived from HFSS among UPF sample

# %%
np.sum((food_noalcnovit_upf["lesshealthy"]* food_noalcnovit_upf['Energykcal']))/np.sum(food_noalcnovit_upf['Energykcal'])#len(food_noalcnovit[food_noalcnovit["NOVA_UPF_NUPF_boolean"]])

# %% [markdown]
# Proportion of weight derived from HFSS among UPF sample

# %%
np.sum((food_noalcnovit_upf["lesshealthy"]* food_noalcnovit_upf['TotalGrams']))/np.sum(food_noalcnovit_upf['TotalGrams'])#len(food_noalcnovit[food_noalcnovit["NOVA_UPF_NUPF_boolean"]])

# %% [markdown]
# Summarise for all foods

# %%
value_notupf_healthy = np.sum((food_noalcnovit["NOVA_UPF_NUPF_boolean"])*~food_noalcnovit["lesshealthy"])/len(food_noalcnovit)
value_upf_healthy = np.sum((food_noalcnovit["NOVA_UPF_NUPF_boolean"])*~food_noalcnovit["lesshealthy"])/len(food_noalcnovit)
value_notupf_lesshealthy = np.sum((food_noalcnovit["NOVA_UPF_NUPF_boolean"])*food_noalcnovit["lesshealthy"])/len(food_noalcnovit)
value_upf_lesshealthy = np.sum((food_noalcnovit["NOVA_UPF_NUPF_boolean"])*food_noalcnovit["lesshealthy"])/len(food_noalcnovit)
print (value_notupf_healthy)
print (value_upf_healthy)
print (value_notupf_lesshealthy)
print (value_upf_lesshealthy)

# %% [markdown]
# Create a sum of weights based on the following (df['c'] = df['a'] + df['b']). It's been advised not to use 'apply() for numeric operations' 
# https://stackoverflow.com/questions/26886653/create-new-column-based-on-values-from-other-columns-apply-a-function-of-multi

# %%
food_noalcnovit.loc[food_noalcnovit['wti_UKY1234rs'] != food_noalcnovit['wti_UKY1234rs'], 'wti_UKY1234rs'] = 0
food_noalcnovit.loc[food_noalcnovit['wti_Y56rs'] != food_noalcnovit['wti_Y56rs'], 'wti_Y56rs'] = 0
food_noalcnovit.loc[food_noalcnovit['wti_Y78rs'] != food_noalcnovit['wti_Y78rs'], 'wti_Y78rs'] = 0
food_noalcnovit.loc[food_noalcnovit['wti_Y911rs'] != food_noalcnovit['wti_Y911rs'], 'wti_Y911rs'] = 0

# %%
food_noalcnovit.loc[:, 'total_weights'] = food_noalcnovit ['wti_UKY1234rs']+ food_noalcnovit['wti_Y56rs'] + food_noalcnovit['wti_Y78rs'] + food_noalcnovit['wti_Y911rs'] 

# %% [markdown]
# SUMMARIES FOR INDIVIDUAL DAILY CONSUMPTION, WEIGHTED

# %% [markdown]
# The food diary entries are based on 3 or 4 day entry. 

# %% [markdown]
# HFSS WEIGHTED 

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4], if we want to limit to 4 days only. 

# %% [markdown]
# Grouping data by serial number and day number. Proportion of weight from HFSS 

# %%
grouped_food_noalcnovit_individual = food_noalcnovit.groupby(['seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['lesshealthy'] * x['TotalGrams'])/np.sum(x['TotalGrams'])))

# %%
grouped_food_noalcnovit_individual = grouped_food_noalcnovit_individual.groupby(level=[0]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
x = grouped_food_noalcnovit_individual
weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]])

# %% [markdown]
# UPF WEIGHTED 

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_individual = food_noalcnovit.groupby(['seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['NOVA_UPF_NUPF_boolean'] * x['TotalGrams'])/np.sum(x['TotalGrams'])))

# %%
grouped_food_noalcnovit_individual = grouped_food_noalcnovit_individual.groupby(level=[0]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
x = grouped_food_noalcnovit_individual
weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]])

# %% [markdown]
# upf anf hfss weighted by weight 

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_upf_hfss = food_noalcnovit.groupby(['seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['lesshealthy'] * x['NOVA_UPF_NUPF_boolean']* x['TotalGrams'])/np.sum(x['TotalGrams'])))

# %%
grouped_food_noalcnovit_upf_hfss = grouped_food_noalcnovit_upf_hfss.groupby(level=[0]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
        
x = grouped_food_noalcnovit_upf_hfss
weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]])

# %% [markdown]
# Proportion of weight derived from UPF only

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_upf_only = food_noalcnovit.groupby(['seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum ((~x['lesshealthy']) * x['NOVA_UPF_NUPF_boolean']* x['TotalGrams'])/np.sum(x['TotalGrams'])))

# %%
grouped_food_noalcnovit_upf_only = grouped_food_noalcnovit_upf_only.groupby(level=[0]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
x = grouped_food_noalcnovit_upf_only
weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]])

# %% [markdown]
# Proportion of weight derived from HFSS only

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_hfss_only = food_noalcnovit.groupby(['seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['lesshealthy'] * (~x['NOVA_UPF_NUPF_boolean'])* x['TotalGrams'])/np.sum(x['TotalGrams'])))

# %%
grouped_food_noalcnovit_hfss_only = grouped_food_noalcnovit_hfss_only.groupby(level=[0]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
x = grouped_food_noalcnovit_hfss_only
weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]])

# %% [markdown]
# Proportion of weight derived from neither UPF nor HFSS

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_hfss_upf_neither = food_noalcnovit.groupby(['seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum ((~x['lesshealthy']) * (~x['NOVA_UPF_NUPF_boolean'])* x['TotalGrams'])/np.sum(x['TotalGrams'])))

# %%
grouped_food_noalcnovit_hfss_upf_neither = grouped_food_noalcnovit_hfss_upf_neither.groupby(level=[0]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
x = grouped_food_noalcnovit_hfss_upf_neither
weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]])

# %% [markdown]
# UPF SAMPLE ONLY

# %% [markdown]
# Proportion of weight derived from HFSS and UPF based on UPF sample

# %%
food_noalcnovit_four_upf = food_noalcnovit[(food_noalcnovit["NOVA_UPF_NUPF_boolean"])]

# %%
grouped_food_noalcnovit_hfss_upfONLY = food_noalcnovit_four_upf.groupby(['seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['lesshealthy'] * x['NOVA_UPF_NUPF_boolean']* x['TotalGrams'])/np.sum(x['TotalGrams'])))

# %%
grouped_food_noalcnovit_hfss_upfONLY = grouped_food_noalcnovit_hfss_upfONLY.groupby(level=[0]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
x = grouped_food_noalcnovit_hfss_upfONLY
weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]])

# %% [markdown]
# INDIVIDUAL DAILY WEIGHTED BY ENERGY

# %% [markdown]
# Proportion of energy derived from HFSS 

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_individual_HFSS = food_noalcnovit.groupby(['seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['lesshealthy'] * x['Energykcal'])/np.sum(x['Energykcal'])))

# %%
grouped_food_noalcnovit_individual_HFSS = grouped_food_noalcnovit_individual_HFSS.groupby(level=[0]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
x = grouped_food_noalcnovit_individual_HFSS
weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]])

# %% [markdown]
# Proportion of energy derived from UPF 

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_individual_UPF = food_noalcnovit.groupby(['seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['NOVA_UPF_NUPF_boolean'] * x['Energykcal'])/np.sum(x['Energykcal'])))

# %%
grouped_food_noalcnovit_individual_UPF = grouped_food_noalcnovit_individual_UPF.groupby(level=[0]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
x = grouped_food_noalcnovit_individual_UPF
weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]])

# %% [markdown]
# Proportion of energy derived from HFSS and UPF

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_hfss_upf = food_noalcnovit.groupby(['seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['lesshealthy'] * x['NOVA_UPF_NUPF_boolean']* x['Energykcal'])/np.sum(x['Energykcal'])))

# %%
grouped_food_noalcnovit_hfss_upf = grouped_food_noalcnovit_hfss_upf.groupby(level=[0]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
x = grouped_food_noalcnovit_hfss_upf
weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]])

# %% [markdown]
# Proportion of energy derived from HFSS only, among UPF sample 

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_hfss_only = food_noalcnovit.groupby(['seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['lesshealthy'] * (~x['NOVA_UPF_NUPF_boolean'])* x['Energykcal'])/np.sum(x['Energykcal'])))

# %%
grouped_food_noalcnovit_hfss_only = grouped_food_noalcnovit_hfss_only.groupby(level=[0]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
x = grouped_food_noalcnovit_hfss_only
weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]])

# %% [markdown]
# Proportion of energy derived from UPF only, among UPF sample 

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_upf_only = food_noalcnovit.groupby(['seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum ((~x['lesshealthy']) * x['NOVA_UPF_NUPF_boolean']* x['Energykcal'])/np.sum(x['Energykcal'])))

# %%
grouped_food_noalcnovit_upf_only = grouped_food_noalcnovit_upf_only.groupby(level=[0]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
x = grouped_food_noalcnovit_upf_only
weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]])

# %% [markdown]
# UPF SAMPLE ONLY

# %%
food_noalcnovit_four_upf = food_noalcnovit[(food_noalcnovit["NOVA_UPF_NUPF_boolean"])]

# %%
grouped_food_noalcnovit_upfsample = food_noalcnovit_four_upf.groupby(['seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['lesshealthy'] * x['NOVA_UPF_NUPF_boolean']* x['Energykcal'])/np.sum(x['Energykcal'])))

# %%
grouped_food_noalcnovit_upfsample = grouped_food_noalcnovit_upfsample.groupby(level=[0]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
x = grouped_food_noalcnovit_upfsample
weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]])

# %% [markdown]
# Summaries for individual daily consumption

# %% [markdown]
# individual daily consumption

# %% [markdown]
# HFSS

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_individual = food_noalcnovit.groupby(['seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['lesshealthy'])/len(x['lesshealthy'])))

# %%
grouped_food_noalcnovit_individual = grouped_food_noalcnovit_individual.groupby(level=[0]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
x = grouped_food_noalcnovit_individual
weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]])

# %% [markdown]
# UPFs

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_individual_upf = food_noalcnovit.groupby(['seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['NOVA_UPF_NUPF_boolean'])/len(x['NOVA_UPF_NUPF_boolean'])))

# %%
grouped_food_noalcnovit_individual_upf = grouped_food_noalcnovit_individual_upf.groupby(level=[0]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
x = grouped_food_noalcnovit_individual_upf
weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]])

# %% [markdown]
# UPFs and HFSS

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_upf_hfss = food_noalcnovit.groupby(['seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['lesshealthy'] * x['NOVA_UPF_NUPF_boolean'])/len(x['NOVA_UPF_NUPF_boolean'])))

# %%
grouped_food_noalcnovit_upf_hfss = grouped_food_noalcnovit_upf_hfss.groupby(level=[0]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
x = grouped_food_noalcnovit_upf_hfss
weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]])

# %% [markdown]
# UPF only

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_upf_only = food_noalcnovit.groupby(['seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (~x['lesshealthy'] * x['NOVA_UPF_NUPF_boolean'])/len(x['NOVA_UPF_NUPF_boolean'])))

# %%
grouped_food_noalcnovit_upf_only = grouped_food_noalcnovit_upf_only.groupby(level=[0]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
x = grouped_food_noalcnovit_upf_only
weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]])

# %% [markdown]
# HFSS only

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_hfss_only = food_noalcnovit.groupby(['seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['lesshealthy'] * (~x['NOVA_UPF_NUPF_boolean']))/len(x['NOVA_UPF_NUPF_boolean'])))

# %%
grouped_food_noalcnovit_hfss_only = grouped_food_noalcnovit_hfss_only.groupby(level=[0]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
x = grouped_food_noalcnovit_hfss_only
weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]])

# %% [markdown]
# Neither HFSS nor UPF

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_hfss_upf_neither = food_noalcnovit.groupby(['seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (~x['lesshealthy'] * (~x['NOVA_UPF_NUPF_boolean']))/len(x['NOVA_UPF_NUPF_boolean'])))

# %%
grouped_food_noalcnovit_hfss_upf_neither = grouped_food_noalcnovit_hfss_upf_neither.groupby(level=[0]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
x = grouped_food_noalcnovit_hfss_upf_neither
weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]])

# %% [markdown]
# UPF and HFSS OVERLAP AMONG UPF PRODUCTS

# %%
food_noalcnovit_four_upf = food_noalcnovit[(food_noalcnovit["NOVA_UPF_NUPF_boolean"])]

# %%
grouped_food_noalcnovit_hfss_upf_upfsample = food_noalcnovit_four_upf.groupby(['seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['lesshealthy'] * x['NOVA_UPF_NUPF_boolean'])/len(x['NOVA_UPF_NUPF_boolean'])))

# %%
grouped_food_noalcnovit_hfss_upf_upfsample = grouped_food_noalcnovit_hfss_upf_upfsample.groupby(level=[0]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
x = grouped_food_noalcnovit_hfss_upf_upfsample
weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]])

# %% [markdown]
# Individual analyses by gender

# %% [markdown]
# HFSS

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_individual = food_noalcnovit.groupby(['Sex', 'seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['lesshealthy'])/len(x['lesshealthy'])))

# %%
grouped_food_noalcnovit_individual = grouped_food_noalcnovit_individual.groupby(level=[0, 1]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
result = grouped_food_noalcnovit_individual.groupby(level=[0]).apply (lambda x: weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]]))
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
print (result.to_frame().to_string())

# %% [markdown]
# UPF 

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_individual_upf = food_noalcnovit.groupby(['Sex', 'seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['NOVA_UPF_NUPF_boolean'])/len(x['NOVA_UPF_NUPF_boolean'])))

# %%
grouped_food_noalcnovit_individual_upf = grouped_food_noalcnovit_individual_upf.groupby(level=[0, 1]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%

        
grouped_food_noalcnovit_individual_upf.groupby(level=[0]).apply (lambda x:  weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]]))

# %% [markdown]
# UPF and HFSS

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_hfss_upf= food_noalcnovit.groupby(['Sex','seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['lesshealthy'] * x['NOVA_UPF_NUPF_boolean'])/len(x['NOVA_UPF_NUPF_boolean'])))

# %%
grouped_food_noalcnovit_hfss_upf_combined = grouped_food_noalcnovit_hfss_upf.groupby(level=[0, 1]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
grouped_food_noalcnovit_hfss_upf_combined.groupby(level=[0]).apply (lambda x: weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]]))

# %% [markdown]
# UPF only

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_upf_only= food_noalcnovit.groupby(['Sex','seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (~x['lesshealthy'] * x['NOVA_UPF_NUPF_boolean'])/len(x['NOVA_UPF_NUPF_boolean'])))

# %%
grouped_food_noalcnovit_upf_only.groupby(level=[0]).apply (lambda x: weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]]))

# %% [markdown]
# HFSS only

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_hfss_only= food_noalcnovit.groupby(['Sex','seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['lesshealthy'] * ~x['NOVA_UPF_NUPF_boolean'])/len(x['NOVA_UPF_NUPF_boolean'])))

# %%
grouped_food_noalcnovit_hfss_only.groupby(level=[0]).apply (lambda x: weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]]))

# %% [markdown]
# Neither HFSS nor UPF

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_hfss_upf_neither= food_noalcnovit.groupby(['Sex','seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (~x['lesshealthy'] * ~x['NOVA_UPF_NUPF_boolean'])/len(x['NOVA_UPF_NUPF_boolean'])))

# %%
grouped_food_noalcnovit_hfss_upf_neither.groupby(level=[0]).apply (lambda x: weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]]))

# %% [markdown]
# UPF and HFSS among UPF only

# %%
food_noalcnovit_four_upf = food_noalcnovit[(food_noalcnovit["NOVA_UPF_NUPF_boolean"])]

# %%
grouped_food_noalcnovit_upf_sample = food_noalcnovit_four_upf.groupby(['Sex', 'seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['lesshealthy'] * x['NOVA_UPF_NUPF_boolean'])/len(x['NOVA_UPF_NUPF_boolean'])))

# %%
grouped_food_noalcnovit_upf_sample.groupby(level=[0]).apply (lambda x: weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]]))

# %% [markdown]
# HFSS by total weight

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_hfss_weight = food_noalcnovit.groupby(['Sex', 'seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['lesshealthy'] * x['TotalGrams'])/np.sum(x['TotalGrams'])))

# %%
grouped_food_noalcnovit_hfss_weight = grouped_food_noalcnovit_hfss_weight.groupby(level=[0, 1]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
grouped_food_noalcnovit_hfss_weight.groupby(level=[0]).apply (lambda x: weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]]))

# %% [markdown]
# UPF by total weight 

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_upf_weight = food_noalcnovit.groupby(['Sex', 'seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['NOVA_UPF_NUPF_boolean'] * x['TotalGrams'])/np.sum(x['TotalGrams'])))

# %%
grouped_food_noalcnovit_upf_weight.groupby(level=[0]).apply (lambda x: weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]]))

# %% [markdown]
# UPF and less healthy by weight

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_upf_hfss = food_noalcnovit.groupby(['Sex', 'seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['lesshealthy'] * x['NOVA_UPF_NUPF_boolean'] * x['TotalGrams'])/np.sum(x['TotalGrams'])))

# %%
grouped_food_noalcnovit_upf_hfss.groupby(level=[0]).apply (lambda x: weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]]))

# %% [markdown]
# HFSS only

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_hfss_only = food_noalcnovit.groupby(['Sex', 'seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['lesshealthy'] * ~x['NOVA_UPF_NUPF_boolean'] * x['TotalGrams'])/np.sum(x['TotalGrams'])))

# %%
grouped_food_noalcnovit_hfss_only = grouped_food_noalcnovit_hfss_only.groupby(level=[0, 1]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
grouped_food_noalcnovit_hfss_only.groupby(level=[0]).apply (lambda x: weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]]))

# %% [markdown]
# UPF only

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_upf_only = food_noalcnovit.groupby(['Sex', 'seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (~x['lesshealthy'] * x['NOVA_UPF_NUPF_boolean'] * x['TotalGrams'])/np.sum(x['TotalGrams'])))

# %%
grouped_food_noalcnovit_upf_only = grouped_food_noalcnovit_upf_only.groupby(level=[0, 1]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
grouped_food_noalcnovit_upf_only.groupby(level=[0]).apply (lambda x: weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]]))

# %% [markdown]
# Neither HFSS nor UPF

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_neither_hfss_upf = food_noalcnovit.groupby(['Sex', 'seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (~x['lesshealthy'] * ~x['NOVA_UPF_NUPF_boolean'] * x['TotalGrams'])/np.sum(x['TotalGrams'])))

# %%
grouped_food_noalcnovit_neither_hfss_upf = grouped_food_noalcnovit_neither_hfss_upf.groupby(level=[0, 1]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
grouped_food_noalcnovit_neither_hfss_upf.groupby(level=[0]).apply (lambda x: weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]]))

# %% [markdown]
# UPF and less healthy for UPF products only

# %%
food_noalcnovit_four_upf = food_noalcnovit[(food_noalcnovit["NOVA_UPF_NUPF_boolean"])]

# %%
grouped_food_noalcnovit_upf_sample = food_noalcnovit_four_upf.groupby(['Sex', 'seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['lesshealthy'] * x['NOVA_UPF_NUPF_boolean'] * x['TotalGrams'])/np.sum(x['TotalGrams'])))

# %%
grouped_food_noalcnovit_upf_sample = grouped_food_noalcnovit_upf_sample.groupby(level=[0, 1]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
grouped_food_noalcnovit_upf_sample.groupby(level=[0]).apply (lambda x: weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]]))

# %% [markdown]
# Neither HFSS nor UPF

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_neither_upf_hfss = food_noalcnovit.groupby(['Sex', 'seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (~x['lesshealthy'] * ~x['NOVA_UPF_NUPF_boolean'] * x['TotalGrams'])/np.sum(x['TotalGrams'])))

# %%
grouped_food_noalcnovit_neither_upf_hfss = grouped_food_noalcnovit_neither_upf_hfss.groupby(level=[0, 1]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
grouped_food_noalcnovit_neither_upf_hfss.groupby(level=[0]).apply (lambda x: weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]]))

# %% [markdown]
# UPF only

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_upf_only = food_noalcnovit.groupby(['Sex', 'seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (~x['lesshealthy'] * x['NOVA_UPF_NUPF_boolean'] * x['Energykcal'])/np.sum(x['Energykcal'])))

# %%
grouped_food_noalcnovit_upf_only = grouped_food_noalcnovit_upf_only.groupby(level=[0, 1]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
grouped_food_noalcnovit_upf_only.groupby(level=[0]).apply (lambda x: weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]]))

# %% [markdown]
# HFSS only 

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_healthy_only = food_noalcnovit.groupby(['Sex', 'seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['lesshealthy'] * ~x['NOVA_UPF_NUPF_boolean'] * x['Energykcal'])/np.sum(x['Energykcal'])))

# %%
grouped_food_noalcnovit_healthy_only = grouped_food_noalcnovit_healthy_only.groupby(level=[0, 1]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
grouped_food_noalcnovit_healthy_only.groupby(level=[0]).apply (lambda x: weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]]))

# %% [markdown]
# HFSS and UPF weighted by energy for UPF goods only

# %%
food_noalcnovit_four_upf = food_noalcnovit[(food_noalcnovit["NOVA_UPF_NUPF_boolean"])]

# %%
grouped_food_noalcnovit_upf_sample = food_noalcnovit_four_upf.groupby(['Sex', 'seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['lesshealthy'] * x['NOVA_UPF_NUPF_boolean'] * x['Energykcal'])/np.sum(x['Energykcal'])))

# %%
grouped_food_noalcnovit_upf_sample = grouped_food_noalcnovit_upf_sample.groupby(level=[0, 1]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
grouped_food_noalcnovit_upf_sample.groupby(level=[0]).apply (lambda x: weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]]))

# %% [markdown]
# NEITHER HFSS NOR UPF WEIGHTED BY ENERGY

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_upf_hfss_NEITHER = food_noalcnovit.groupby(['Sex', 'seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (~x['lesshealthy'] * ~x['NOVA_UPF_NUPF_boolean'] * x['Energykcal'])/np.sum(x['Energykcal'])))

# %%
grouped_food_noalcnovit_upf_hfss_NEITHER = grouped_food_noalcnovit_upf_hfss_NEITHER.groupby(level=[0, 1]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
grouped_food_noalcnovit_upf_hfss_NEITHER.groupby(level=[0]).apply (lambda x: weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]]))

# %% [markdown]
# HFSS and UPF weighted by energy

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_upf_hfss = food_noalcnovit.groupby(['Sex', 'seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['lesshealthy'] * x['NOVA_UPF_NUPF_boolean'] * x['Energykcal'])/np.sum(x['Energykcal'])))

# %%
grouped_food_noalcnovit_upf_hfss = grouped_food_noalcnovit_upf_hfss.groupby(level=[0, 1]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
        grouped_food_noalcnovit_upf_hfss.groupby(level=[0]).apply (lambda x: weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]]))

# %% [markdown]
# UPF by gender weighted by energy

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_upf = food_noalcnovit.groupby(['Sex', 'seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['NOVA_UPF_NUPF_boolean'] * x['Energykcal'])/np.sum(x['Energykcal'])))

# %%
grouped_food_noalcnovit_upf = grouped_food_noalcnovit_upf.groupby(level=[0, 1]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
grouped_food_noalcnovit_upf.groupby(level=[0]).apply (lambda x: weighted_avg_and_std (list(list_wo_nan(x)[1]), weights = [y for y in list_wo_nan(x)[0]]))

# %% [markdown]
# HFSS by gender weighted by energy

# %%
#food_noalcnovit_four = food_noalcnovit[food_noalcnovit['DiaryDaysCompleted']==4]

# %%
grouped_food_noalcnovit_hfss = food_noalcnovit.groupby(['Sex', 'seriali', 'DayNo']).apply (lambda x: (np.min(x['total_weights']), np.sum (x['lesshealthy'] * x['Energykcal'])/np.sum(x['Energykcal'])))

# %%
grouped_food_noalcnovit_hfss

# %%
grouped_food_noalcnovit_hfss = grouped_food_noalcnovit_hfss.groupby(level=[0, 1]).apply (lambda x: (np.min([y[0] for y in x]), np.mean ([y[1] for y in x])))

# %%
grouped_food_noalcnovit_hfss 

# %%

    grouped_food_noalcnovit_hfss.groupby(level=[0]).apply (lambda x: weighted_avg_and_std (list(list_wo_nan(x)[1]), weights =  [y for y in list_wo_nan(x)[0]]))


