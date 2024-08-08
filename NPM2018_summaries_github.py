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

# %% [markdown]
# Exclude alcohol and supplements

# %%
categories= {'47A', '47B','48A', '48B', '48C', '49A', '49B', '49C', '49D', '49E', '54A', '54B', '54C', '54D', '54E','54F', '54G', '54H', '54I', '54J', '54K', '54L', '54M', '54N', '54P'}
food_noalcnovit=food[~food['SubFoodGroupCode'].isin(categories)]

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
food_noalcnovit.loc[:,"vegetable"] = food_noalcnovit["Tomatoesg"]+ food_noalcnovit["Brassicaceaeg"]+food_noalcnovit["YellowRedGreeng"]+food_noalcnovit["Beansg"]+food_noalcnovit["OtherVegg"]

# %% [markdown]
# Calculate fruit, veg, and nuts per 100g 

# %%
food_noalcnovit.loc [:,"fvn_per_100_g"] = (food_noalcnovit["Fruitg"] + food_noalcnovit["vegetable"]+ food_noalcnovit["Nutsg"]+ food_noalcnovit['TomatoPureeg']*2+ food_noalcnovit['veg_pureed']*2+ food_noalcnovit['fruit_pureed']*2+food_noalcnovit['DriedFruitg']*2)/(food_noalcnovit["TotalGrams"]+ food_noalcnovit['TomatoPureeg'] + food_noalcnovit['veg_pureed']+ food_noalcnovit['fruit_pureed']+ food_noalcnovit['DriedFruitg'])*100

# %% [markdown]
# Calculate salt

# %%
food_noalcnovit.loc[:,"salt"]= food_noalcnovit["Sodiummg"]/393.4

# %% [markdown]
# Calculate the amount per 100g

# %%
food_noalcnovit.loc [:, "Fruitjuice_per_100_g"] = food_noalcnovit["FruitJuiceg"]/food_noalcnovit["TotalGrams"]*100

# %%
food_noalcnovit.loc [:, "Smoothies_per_100_g"] = food_noalcnovit["SmoothieFruitg"]/food_noalcnovit["TotalGrams"]*100

# %%
food_noalcnovit.loc[:,"EnergykJ_per_100_g"] = food_noalcnovit["EnergykJ"]/food_noalcnovit["TotalGrams"]*100

# %%
food_noalcnovit.loc [:,"Saturatedfattyacidsg_per_100_g"] = food_noalcnovit["Saturatedfattyacidsg"]/food_noalcnovit["TotalGrams"]*100

# %%
food_noalcnovit.loc [:,"salt_per_100_g"]= food_noalcnovit["salt"]/food_noalcnovit["TotalGrams"]*100

# %%
food_noalcnovit.loc[:,"FreeSugarsg_per_100_g"] = food_noalcnovit["FreeSugarsg"]/food_noalcnovit["TotalGrams"]*100

# %%
food_noalcnovit.loc[:,"Englystfibreg_per_100_g"] = food_noalcnovit["Englystfibreg"]/food_noalcnovit["TotalGrams"]*100

# %%
food_noalcnovit.loc [:,"AOACFibreg_per_100_g"] = food_noalcnovit["AOACFibreg"]/food_noalcnovit["TotalGrams"]*100

# %%
food_noalcnovit.loc [:,"Proteing_per_100_g"] = food_noalcnovit["Proteing"]/food_noalcnovit["TotalGrams"]*100

# %%
food_noalcnovit.NOVANumNEW_agreement.value_counts(normalize=True)*100

# %%
food_noalcnovit.loc[:, 'NOVA_UPF_NUPF_boolean'] = (food_noalcnovit['NOVANumNEW_agreement'] == 'Ultra-processed food and drink products')

# %%
food_noalcnovit.NOVA_UPF_NUPF_boolean.value_counts(normalize=True)*100

# %% [markdown]
# A POINTS

# %%
energy_points18 = {0: [-1, 315], 1: [315, 630], 2: [630, 945], 3: [945, 1260], 
                 4: [1260, 1575], 5: [1575, 1890], 6: [1890, 2205], 7: [2205, 2520],
                 8: [2520, 2835], 9: [2835, 3150], 10: [3150]}
food_noalcnovit.loc [:,'energy_a_points'] = np.array(len(food_noalcnovit) * [0])
for points in energy_points18:
    interval = energy_points18[points]
    if len(interval) == 2:
        food_noalcnovit.loc[(food_noalcnovit['EnergykJ_per_100_g'] > interval[0]) & (food_noalcnovit['EnergykJ_per_100_g'] <= interval[1]),'energy_a_points'] = points
    elif len(interval) == 1:
        food_noalcnovit.loc[(food_noalcnovit['EnergykJ_per_100_g'] > interval[0]), 'energy_a_points'] = points

# %%
satfat_points18 = {0: [-1, 0.9], 1: [0.9, 1.9], 2: [1.9, 2.8], 3: [2.8, 3.7], 
                 4: [3.7, 4.7], 5: [4.7, 5.6], 6: [5.6, 6.6], 7: [6.6, 7.5],
                 8: [7.5, 8.4], 9: [8.4, 9.4], 10: [9.4]}
food_noalcnovit.loc [:,'satfat_a_points'] = np.array(len(food_noalcnovit) * [0])
for points in satfat_points18:
    interval = satfat_points18[points]
    if len(interval) == 2:
        food_noalcnovit.loc[(food_noalcnovit['Saturatedfattyacidsg_per_100_g'] > interval[0]) & (food_noalcnovit['Saturatedfattyacidsg_per_100_g'] <= interval[1]),'satfat_a_points'] = points
    elif len(interval) == 1:
        food_noalcnovit.loc[(food_noalcnovit['Saturatedfattyacidsg_per_100_g'] > interval[0]), 'satfat_a_points'] = points

# %%
freesugar_points = {0: [-1, 0.9], 1: [0.9, 1.9], 2: [1.9, 2.8], 3: [2.8, 3.7], 
                 4: [3.7, 4.6], 5: [4.6, 5.6], 6: [5.6, 6.5], 7: [6.5, 7.4],
                 8: [7.4, 8.3], 9: [8.3, 9.3], 10: [9.3]}
food_noalcnovit.loc[:,'freesugar_a_points'] = np.array(len(food_noalcnovit) * [0])
for points in freesugar_points:
    interval = freesugar_points[points]
    if len(interval) == 2:
        food_noalcnovit.loc[(food_noalcnovit['FreeSugarsg_per_100_g'] > interval[0]) & (food_noalcnovit['FreeSugarsg_per_100_g'] <= interval[1]),'freesugar_a_points'] = points
    elif len(interval) == 1:
        food_noalcnovit.loc[(food_noalcnovit['FreeSugarsg_per_100_g'] > interval[0]), 'freesugar_a_points'] = points

# %%
salt_points = {0: [-1, 0.9], 1: [0.9, 1.9], 2: [1.9, 2.8], 3: [2.8, 3.7], 
                 4: [3.7, 4.6], 5: [4.6, 5.6], 6: [5.6, 6.5], 7: [6.5, 7.4],
                 8: [7.4, 8.3], 9: [8.3, 9.3], 10: [9.3]}
food_noalcnovit.loc [:,'salt_a_points'] = np.array(len(food_noalcnovit) * [0])
for points in salt_points:
    interval = salt_points[points]
    if len(interval) == 2:
        food_noalcnovit.loc[(food_noalcnovit['salt_per_100_g'] > interval[0])  & (food_noalcnovit['salt_per_100_g'] <= interval[1]),'salt_a_points'] = points
    elif len(interval) == 1:
        food_noalcnovit.loc[(food_noalcnovit['salt_per_100_g'] > interval[0]), 'salt_a_points'] = points

# %%
food_noalcnovit.loc [:,'total_a_points'] = food_noalcnovit["energy_a_points"] + food_noalcnovit["satfat_a_points"] + food_noalcnovit["freesugar_a_points"] + food_noalcnovit["salt_a_points"] 

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

food_noalcnovit.loc [:,'fvn_c_points'] = np.array(len(food_noalcnovit) * [0])
for points in fvn_points:
    interval = fvn_points[points]
    if len(interval) == 2:
        food_noalcnovit.loc[(food_noalcnovit['fvn_per_100_g'] > interval[0])  & (food_noalcnovit['fvn_per_100_g'] <= interval[1]),'fvn_c_points'] = points
    elif len(interval) == 1:
        food_noalcnovit.loc[(food_noalcnovit['fvn_per_100_g'] > interval[0]), 'fvn_c_points'] = points

# %%
aoacfibre_points = {0: [-1, 0.7], 1: [0.7, 1.4], 2: [1.4, 2.2], 3: [2.2, 2.9], 4: [2.9, 3.6], 5: [3.6, 4.3], 6: [4.3, 5.0], 7:[5.0, 5.8],8:[5.8]} 

food_noalcnovit.loc [:,'aoacfibre_c_points'] = np.array(len(food_noalcnovit) * [0])
for points in aoacfibre_points:
    interval = aoacfibre_points[points]
    if len(interval) == 2:
        food_noalcnovit.loc[(food_noalcnovit['AOACFibreg_per_100_g'] > interval[0]) & (food_noalcnovit['AOACFibreg_per_100_g'] <= interval[1]),'aoacfibre_c_points'] = points
    elif len(interval) == 1:
        food_noalcnovit.loc[(food_noalcnovit['AOACFibreg_per_100_g'] > interval[0]), 'aoacfibre_c_points'] = points

# %%
protein_points = {0: [-1, 1.6], 1: [1.6, 3.2], 2: [3.2, 4.8], 3: [4.8, 6.4], 4: [6.4, 8.0], 5: [8.0]} 

food_noalcnovit.loc[:,'protein_c_points'] = np.array(len(food_noalcnovit) * [0])
for points in protein_points:
    interval = protein_points[points]
    if len(interval) == 2:
        food_noalcnovit.loc[(food_noalcnovit['Proteing_per_100_g'] > interval[0]) & (food_noalcnovit['Proteing_per_100_g'] <= interval[1]) & (food_noalcnovit['total_a_points'] < 11),'protein_c_points'] = points
        food_noalcnovit.loc[(food_noalcnovit['Proteing_per_100_g'] > interval[0]) & (food_noalcnovit['Proteing_per_100_g'] <= interval[1])  & (food_noalcnovit['total_a_points'] >= 11)  & (food_noalcnovit['fvn_c_points'] >= 5),'protein_c_points'] = points
    elif len(interval) == 1:
        food_noalcnovit.loc[(food_noalcnovit['Proteing_per_100_g'] > interval[0]) & (food_noalcnovit['total_a_points'] < 11), 'protein_c_points'] = points
        food_noalcnovit.loc[(food_noalcnovit['Proteing_per_100_g'] > interval[0]) & (food_noalcnovit['total_a_points'] >= 11) & (food_noalcnovit['fvn_c_points'] >= 5), 'protein_c_points'] = points

# %%
food_noalcnovit.loc [:,'total_c_points'] = food_noalcnovit["fvn_c_points"] + food_noalcnovit["aoacfibre_c_points"] + food_noalcnovit["protein_c_points"]+food_noalcnovit['fruit_juice']+ food_noalcnovit['smoothies']

# %% [markdown]
# TOTAL POINTS

# %%
food_noalcnovit.loc[:, "total_points"] = food_noalcnovit["total_a_points"]-food_noalcnovit ["total_c_points"]

# %%
food_noalcnovit.loc [:,"lesshealthy"] = food_noalcnovit["total_points"] >= 4 

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

# %% [markdown]
# Calculate the proportion of HFSS foods

# %%
np.sum(food_noalcnovit["lesshealthy"])/len(food_noalcnovit["lesshealthy"])

# %% [markdown]
# Calculate the overlap between UPF and HFSS based on UPF sample

# %%
np.sum((food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit["lesshealthy"]))/len(food_noalcnovit[food_noalcnovit["NOVA_UPF_NUPF_boolean"]])

# %% [markdown]
# Summarise HFSS (NPM 2018); UPFs; HFSS (NPM 2018) and UPFs; Neither HFSS (NPM 2018) nor UPFs
# 

# %%
value_notupf_healthy = np.sum((~food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(~food_noalcnovit["lesshealthy"]))/len(food_noalcnovit)
value_upf_healthy = np.sum((food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(~food_noalcnovit["lesshealthy"]))/len(food_noalcnovit)
value_notupf_lesshealthy = np.sum((~food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit["lesshealthy"]))/len(food_noalcnovit)
value_upf_lesshealthy = np.sum((food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit["lesshealthy"]))/len(food_noalcnovit)
print (value_notupf_healthy)
print (value_upf_healthy)
print (value_notupf_lesshealthy)
print (value_upf_lesshealthy)

# %% [markdown]
# weight by energy

# %%
np.sum((food_noalcnovit["lesshealthy"])*(food_noalcnovit['Energykcal']))/np.sum(food_noalcnovit["Energykcal"])

# %% [markdown]
# Create a UPF sample

# %%
food_noalcnovit_upf = food_noalcnovit[food_noalcnovit['NOVA_UPF_NUPF_boolean']]

# %% [markdown]
# UPF and HFSS overlap weighted by energy

# %%
np.sum((food_noalcnovit_upf["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit_upf["lesshealthy"])*(food_noalcnovit_upf['Energykcal']))/np.sum (food_noalcnovit_upf["Energykcal"])

# %% [markdown]
# Proportion of weight from UPFs

# %%
np.sum((food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit['TotalGrams']))/np.sum(food_noalcnovit["TotalGrams"])

# %% [markdown]
# Proportion of weight from HFSS

# %%
np.sum((food_noalcnovit["lesshealthy"])*(food_noalcnovit['TotalGrams']))/np.sum(food_noalcnovit["TotalGrams"])

# %% [markdown]
# Create a UPF sample

# %%
food_noalcnovit_upf = food_noalcnovit[food_noalcnovit['NOVA_UPF_NUPF_boolean']]

# %% [markdown]
# Fraction of HFSS in the UPF by weight

# %%
np.sum((food_noalcnovit_upf["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit_upf["lesshealthy"])*(food_noalcnovit_upf['TotalGrams']))/np.sum (food_noalcnovit_upf["TotalGrams"])

# %% [markdown]
# Sub-group analyses for robustness check: see Table S1 in the apendix

# %% [markdown]
# For males

# %%
food_noalcnovit_male = food_noalcnovit[food_noalcnovit['Sex']==1]

# %% [markdown]
# proportion of HFSS

# %%
np.sum(food_noalcnovit_male["lesshealthy"])/len(food_noalcnovit_male["lesshealthy"])

# %% [markdown]
# proportion of UPF

# %%
np.sum(food_noalcnovit_male["NOVA_UPF_NUPF_boolean"])/len(food_noalcnovit_male["NOVA_UPF_NUPF_boolean"])

# %% [markdown]
# overlap between UPF and HFSS among UPF sample

# %%
np.sum((food_noalcnovit_male["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit_male["lesshealthy"]))/len(food_noalcnovit_male[food_noalcnovit_male["NOVA_UPF_NUPF_boolean"]])

# %% [markdown]
# Summaries for males only

# %%
value_notupf_healthy_m = np.sum((~food_noalcnovit_male["NOVA_UPF_NUPF_boolean"])*(~food_noalcnovit_male["lesshealthy"]))/len(food_noalcnovit_male)
value_upf_healthy_m = np.sum((food_noalcnovit_male["NOVA_UPF_NUPF_boolean"])*(~food_noalcnovit_male["lesshealthy"]))/len(food_noalcnovit_male)
value_notupf_lesshealthy_m = np.sum((~food_noalcnovit_male["NOVA_UPF_NUPF_boolean"])*food_noalcnovit_male["lesshealthy"])/len(food_noalcnovit_male)
value_upf_lesshealthy_m = np.sum((food_noalcnovit_male["NOVA_UPF_NUPF_boolean"])*food_noalcnovit_male["lesshealthy"])/len(food_noalcnovit_male)
print (value_notupf_healthy_m)
print (value_upf_healthy_m)
print (value_notupf_lesshealthy_m)
print (value_upf_lesshealthy_m)

# %% [markdown]
# For females

# %%
food_noalcnovit_female = food_noalcnovit[food_noalcnovit['Sex']==2]

# %% [markdown]
# proportion of HFSS

# %%
np.sum(food_noalcnovit_female["lesshealthy"])/len(food_noalcnovit_female["lesshealthy"])

# %% [markdown]
# proportion of UPF

# %%
np.sum(food_noalcnovit_female["NOVA_UPF_NUPF_boolean"])/len(food_noalcnovit_female["NOVA_UPF_NUPF_boolean"])

# %% [markdown]
# Overlap between UPF and HFSS among UPFs

# %%
np.sum((food_noalcnovit_female["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit_female["lesshealthy"]))/len(food_noalcnovit_female[food_noalcnovit_female["NOVA_UPF_NUPF_boolean"]])

# %% [markdown]
# Summaries for females

# %%
value_notupf_healthy_f = np.sum((~food_noalcnovit_female["NOVA_UPF_NUPF_boolean"])*~food_noalcnovit_female["lesshealthy"])/len(food_noalcnovit_female)
value_upf_healthy_f = np.sum((food_noalcnovit_female["NOVA_UPF_NUPF_boolean"])*~food_noalcnovit_female["lesshealthy"])/len(food_noalcnovit_female)
value_notupf_lesshealthy_f = np.sum((~food_noalcnovit_female["NOVA_UPF_NUPF_boolean"])*food_noalcnovit_female["lesshealthy"])/len(food_noalcnovit_female)
value_upf_lesshealthy_f = np.sum((food_noalcnovit_female["NOVA_UPF_NUPF_boolean"])*food_noalcnovit_female["lesshealthy"])/len(food_noalcnovit_female)
print (value_notupf_healthy_f)
print (value_upf_healthy_f)
print (value_notupf_lesshealthy_f)
print (value_upf_lesshealthy_f)

# %% [markdown]
# Weighted by energy

# %%
np.sum((food_noalcnovit_female["lesshealthy"])*(food_noalcnovit_female['Energykcal']))/np.sum(food_noalcnovit_female["Energykcal"])

# %%
np.sum((food_noalcnovit_male["lesshealthy"])*(food_noalcnovit_male['Energykcal']))/np.sum(food_noalcnovit_male["Energykcal"])

# %% [markdown]
# Create a UPF sample

# %%
food_noalcnovit_upf_male = food_noalcnovit_male[food_noalcnovit_male['NOVA_UPF_NUPF_boolean']]

# %% [markdown]
# Fraction of HFSS within UPF sample weighted by energy for males

# %%
np.sum((food_noalcnovit_upf_male["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit_upf_male["lesshealthy"])*(food_noalcnovit_upf_male['Energykcal']))/np.sum (food_noalcnovit_upf_male["Energykcal"])

# %% [markdown]
# Fraction of HFSS weighted by energy for females

# %%
np.sum((food_noalcnovit_female["lesshealthy"])*(food_noalcnovit_female['Energykcal']))/np.sum(food_noalcnovit_female["Energykcal"])

# %%
food_noalcnovit_upf_female = food_noalcnovit_female[food_noalcnovit_female['NOVA_UPF_NUPF_boolean']]

# %% [markdown]
# Fraction of HFSS and UPF within UPF sample weighted by energy for females

# %%
np.sum((food_noalcnovit_upf_female["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit_upf_female["lesshealthy"])*(food_noalcnovit_upf_female['Energykcal']))/np.sum (food_noalcnovit_upf_female["Energykcal"])

# %% [markdown]
# Fraction of HFSS weighted by weight for males

# %%
np.sum((food_noalcnovit_male["lesshealthy"])*(food_noalcnovit_male['TotalGrams']))/np.sum(food_noalcnovit_male["TotalGrams"])

# %% [markdown]
# Fraction of HFSS and UPF within the UPF sample weighted by weight for males

# %%
np.sum((food_noalcnovit_upf_male["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit_upf_male["lesshealthy"])*(food_noalcnovit_upf_male['TotalGrams']))/np.sum (food_noalcnovit_upf_male["NOVA_UPF_NUPF_boolean"])

# %% [markdown]
# Fraction of HFSS weighted by weight for females

# %%
np.sum((food_noalcnovit_female["lesshealthy"])*(food_noalcnovit_female['TotalGrams']))/np.sum(food_noalcnovit_female["TotalGrams"])

# %% [markdown]
# Fraction of HFSS and UPF within UPF sample weighted by weight for females

# %%
np.sum((food_noalcnovit_upf_female["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit_upf_female["lesshealthy"])*(food_noalcnovit_upf_female['TotalGrams']))/np.sum (food_noalcnovit_upf_female["TotalGrams"])

# %% [markdown]
# Fraction of UPF weighted by weight for females

# %%
np.sum((food_noalcnovit_male["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit_male['TotalGrams']))/np.sum(food_noalcnovit_male["TotalGrams"])

# %% [markdown]
# Fraction of UPF and HFSS weighted by weight for males

# %%
np.sum((food_noalcnovit_male["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit_male["lesshealthy"])*(food_noalcnovit_male['TotalGrams']))/np.sum (food_noalcnovit_male["NOVA_UPF_NUPF_boolean"])

# %% [markdown]
# Fraction of UPF and HFSS weighted by weight for females

# %%
np.sum((food_noalcnovit_female["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit_female['TotalGrams']))/np.sum(food_noalcnovit_female["TotalGrams"])

# %% [markdown]
# Fraction of UPF and HFSS weighted by weight for females

# %%
np.sum((food_noalcnovit_female["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit_female["lesshealthy"])*(food_noalcnovit_female['TotalGrams']))/np.sum (food_noalcnovit_female["NOVA_UPF_NUPF_boolean"])

# %% [markdown]
# Here we quantify the proportion of food and drink items from NDNS that are: neither HFSS (NPM 2018/19) nor UPFs; UPFs only;
# HFSS only (NPM 2018/19); HFSS (NPM 2018/19) and UPFs based on the overall sample.
# For the summary graphs aggregate for food and drink 

# %%
value_notupf_healthy = np.sum((~food_noalcnovit["NOVA_UPF_NUPF_boolean"])*~food_noalcnovit["lesshealthy"])/len(food_noalcnovit)
value_upf_healthy = np.sum((food_noalcnovit["NOVA_UPF_NUPF_boolean"])*~food_noalcnovit["lesshealthy"])/len(food_noalcnovit)
value_notupf_lesshealthy = np.sum((~food_noalcnovit["NOVA_UPF_NUPF_boolean"])*food_noalcnovit["lesshealthy"])/len(food_noalcnovit)
value_upf_lesshealthy = np.sum((food_noalcnovit["NOVA_UPF_NUPF_boolean"])*food_noalcnovit["lesshealthy"])/len(food_noalcnovit)
print (value_notupf_healthy)
print (value_upf_healthy)
print (value_notupf_lesshealthy)
print (value_upf_lesshealthy)

# %% [markdown]
# Proportion of energy derived from: neither HFSS (NPM 2004/5) nor UPFs; UPFs only; HFSS only (NPM 2018/19); HFSS (NPM 2018/19)  and UPFs based on the overall sample.

# %%
value_notupf_healthy = np.sum(((~food_noalcnovit["NOVA_UPF_NUPF_boolean"]))*(~food_noalcnovit["lesshealthy"])*(food_noalcnovit['Energykcal']))/np.sum(food_noalcnovit['Energykcal'])
value_upf_healthy = np.sum(((food_noalcnovit["NOVA_UPF_NUPF_boolean"]))*(~food_noalcnovit["lesshealthy"])*(food_noalcnovit['Energykcal']))/np.sum(food_noalcnovit['Energykcal'])
value_notupf_lesshealthy = np.sum(((~food_noalcnovit["NOVA_UPF_NUPF_boolean"]))*((food_noalcnovit["lesshealthy"]))*(food_noalcnovit['Energykcal']))/np.sum(food_noalcnovit['Energykcal'])
value_upf_lesshealthy = np.sum(((food_noalcnovit["NOVA_UPF_NUPF_boolean"]))*(food_noalcnovit["lesshealthy"])*(food_noalcnovit['Energykcal']))/np.sum(food_noalcnovit['Energykcal'])
print (value_notupf_healthy)
print (value_upf_healthy)
print (value_notupf_lesshealthy)
print (value_upf_lesshealthy)

# %% [markdown]
# Proportion of weight derived from: neither HFSS (NPM 2004/5) nor UPFs; UPFs only; HFSS only (NPM 2018/19); HFSS (NPM 2018/19)  and UPFs based on the overall sample.

# %%
value_notupf_healthy = np.sum(((~food_noalcnovit["NOVA_UPF_NUPF_boolean"]))*(~food_noalcnovit["lesshealthy"])*(food_noalcnovit['TotalGrams']))/np.sum(food_noalcnovit['TotalGrams'])
value_upf_healthy = np.sum(((food_noalcnovit["NOVA_UPF_NUPF_boolean"]))*(~food_noalcnovit["lesshealthy"])*(food_noalcnovit['TotalGrams']))/np.sum(food_noalcnovit['TotalGrams'])
value_notupf_lesshealthy = np.sum(((~food_noalcnovit["NOVA_UPF_NUPF_boolean"]))*((food_noalcnovit["lesshealthy"]))*(food_noalcnovit['TotalGrams']))/np.sum(food_noalcnovit['TotalGrams'])
value_upf_lesshealthy = np.sum(((food_noalcnovit["NOVA_UPF_NUPF_boolean"]))*(food_noalcnovit["lesshealthy"])*(food_noalcnovit['TotalGrams']))/np.sum(food_noalcnovit['TotalGrams'])
print (value_notupf_healthy)
print (value_upf_healthy)
print (value_notupf_lesshealthy)
print (value_upf_lesshealthy)

# %% [markdown]
# Analyses for males

# %%
food_noalcnovit_male = food_noalcnovit[food_noalcnovit['Sex']==1]

# %% [markdown]
# The proportion of food and drink items from NDNS that are: neither HFSS (NPM 2004/5) nor UPFs; UPFs only;
# HFSS only (NPM 2018/19); HFSS (NPM 2018/19) and UPFs based on male sample.

# %%
value_notupf_healthy_m = np.sum((~food_noalcnovit_male["NOVA_UPF_NUPF_boolean"])*~food_noalcnovit_male["lesshealthy"])/len(food_noalcnovit_male)
value_upf_healthy_m = np.sum((food_noalcnovit_male["NOVA_UPF_NUPF_boolean"])*~food_noalcnovit_male["lesshealthy"])/len(food_noalcnovit_male)
value_notupf_lesshealthy_m = np.sum((~food_noalcnovit_male["NOVA_UPF_NUPF_boolean"])*food_noalcnovit_male["lesshealthy"])/len(food_noalcnovit_male)
value_upf_lesshealthy_m = np.sum((food_noalcnovit_male["NOVA_UPF_NUPF_boolean"])*food_noalcnovit_male["lesshealthy"])/len(food_noalcnovit_male)
print (value_notupf_healthy_m)
print (value_upf_healthy_m)
print (value_notupf_lesshealthy_m)
print (value_upf_lesshealthy_m)

# %% [markdown]
# The proportion of food and drink items from NDNS that are: neither HFSS (NPM 2004/5) nor UPFs; UPFs only;
# HFSS only (NPM 2018/19); HFSS (NPM 2018/19) and UPFs based on male sample, weighted by energy.

# %%
value_notupf_healthy = np.sum(((~food_noalcnovit_male["NOVA_UPF_NUPF_boolean"]))*(~food_noalcnovit_male["lesshealthy"])*(food_noalcnovit_male['Energykcal']))/np.sum(food_noalcnovit_male['Energykcal'])
value_upf_healthy = np.sum(((food_noalcnovit_male["NOVA_UPF_NUPF_boolean"]))*(~food_noalcnovit_male["lesshealthy"])*(food_noalcnovit_male['Energykcal']))/np.sum(food_noalcnovit_male['Energykcal'])
value_notupf_lesshealthy = np.sum(((~food_noalcnovit_male["NOVA_UPF_NUPF_boolean"]))*((food_noalcnovit_male["lesshealthy"]))*(food_noalcnovit_male['Energykcal']))/np.sum(food_noalcnovit_male['Energykcal'])
value_upf_lesshealthy = np.sum(((food_noalcnovit_male["NOVA_UPF_NUPF_boolean"]))*(food_noalcnovit_male["lesshealthy"])*(food_noalcnovit_male['Energykcal']))/np.sum(food_noalcnovit_male['Energykcal'])
print (value_notupf_healthy)
print (value_upf_healthy)
print (value_notupf_lesshealthy)
print (value_upf_lesshealthy)

# %% [markdown]
# The proportion of food and drink items from NDNS that are: neither HFSS (NPM 2004/5) nor UPFs; UPFs only;
# HFSS only (NPM 2018/19); HFSS (NPM 2018/19) and UPFs based on male sample, weighted by weight.

# %%
value_notupf_healthy = np.sum(((~food_noalcnovit_male["NOVA_UPF_NUPF_boolean"]))*(~food_noalcnovit_male["lesshealthy"])*(food_noalcnovit_male['TotalGrams']))/np.sum(food_noalcnovit_male['TotalGrams'])
value_upf_healthy = np.sum(((food_noalcnovit_male["NOVA_UPF_NUPF_boolean"]))*(~food_noalcnovit_male["lesshealthy"])*(food_noalcnovit_male['TotalGrams']))/np.sum(food_noalcnovit_male['TotalGrams'])
value_notupf_lesshealthy = np.sum(((~food_noalcnovit_male["NOVA_UPF_NUPF_boolean"]))*((food_noalcnovit_male["lesshealthy"]))*(food_noalcnovit_male['TotalGrams']))/np.sum(food_noalcnovit_male['TotalGrams'])
value_upf_lesshealthy = np.sum(((food_noalcnovit_male["NOVA_UPF_NUPF_boolean"]))*(food_noalcnovit_male["lesshealthy"])*(food_noalcnovit_male['TotalGrams']))/np.sum(food_noalcnovit_male['TotalGrams'])
print (value_notupf_healthy)
print (value_upf_healthy)
print (value_notupf_lesshealthy)
print (value_upf_lesshealthy)

# %% [markdown]
# FOR FEMALES

# %%
food_noalcnovit_female = food_noalcnovit[food_noalcnovit['Sex']==2]

# %% [markdown]
# The proportion of food and drink items from NDNS that are: neither HFSS (NPM 2004/5) nor UPFs; UPFs only;
# HFSS only (NPM 2018/19); HFSS (NPM 2018/19) and UPFs based on female sample.

# %%
value_notupf_healthy_f = np.sum((~food_noalcnovit_female["NOVA_UPF_NUPF_boolean"])*~food_noalcnovit_female["lesshealthy"])/len(food_noalcnovit_female)
value_upf_healthy_f = np.sum((food_noalcnovit_female["NOVA_UPF_NUPF_boolean"])*~food_noalcnovit_female["lesshealthy"])/len(food_noalcnovit_female)
value_notupf_lesshealthy_f = np.sum((~food_noalcnovit_female["NOVA_UPF_NUPF_boolean"])*food_noalcnovit_female["lesshealthy"])/len(food_noalcnovit_female)
value_upf_lesshealthy_f = np.sum((food_noalcnovit_female["NOVA_UPF_NUPF_boolean"])*food_noalcnovit_female["lesshealthy"])/len(food_noalcnovit_female)
print (value_notupf_healthy_f)
print (value_upf_healthy_f)
print (value_notupf_lesshealthy_f)
print (value_upf_lesshealthy_f)

# %% [markdown]
# The proportion of food and drink items from NDNS that are: neither HFSS (NPM 2004/5) nor UPFs; UPFs only;
# HFSS only (NPM 2018/19); HFSS (NPM 2018/19) and UPFs based on female sample, weighted by energy.

# %%
value_notupf_healthy = np.sum(((~food_noalcnovit_female["NOVA_UPF_NUPF_boolean"]))*(~food_noalcnovit_female["lesshealthy"])*(food_noalcnovit_female['Energykcal']))/np.sum(food_noalcnovit_female['Energykcal'])
value_upf_healthy = np.sum(((food_noalcnovit_female["NOVA_UPF_NUPF_boolean"]))*(~food_noalcnovit_female["lesshealthy"])*(food_noalcnovit_female['Energykcal']))/np.sum(food_noalcnovit_female['Energykcal'])
value_notupf_lesshealthy = np.sum(((~food_noalcnovit_female["NOVA_UPF_NUPF_boolean"]))*((food_noalcnovit_female["lesshealthy"]))*(food_noalcnovit_female['Energykcal']))/np.sum(food_noalcnovit_female['Energykcal'])
value_upf_lesshealthy = np.sum(((food_noalcnovit_female["NOVA_UPF_NUPF_boolean"]))*(food_noalcnovit_female["lesshealthy"])*(food_noalcnovit_female['Energykcal']))/np.sum(food_noalcnovit_female['Energykcal'])
print (value_notupf_healthy)
print (value_upf_healthy)
print (value_notupf_lesshealthy)
print (value_upf_lesshealthy)

# %% [markdown]
# The proportion of food and drink items from NDNS that are: neither HFSS (NPM 2004/5) nor UPFs; UPFs only;
# HFSS only (NPM 2018/19); HFSS (NPM 2018/19) and UPFs based on female sample, weighted by weight.

# %%
value_notupf_healthy = np.sum(((~food_noalcnovit_female["NOVA_UPF_NUPF_boolean"]))*(~food_noalcnovit_female["lesshealthy"])*(food_noalcnovit_female['TotalGrams']))/np.sum(food_noalcnovit_female['TotalGrams'])
value_upf_healthy = np.sum(((food_noalcnovit_female["NOVA_UPF_NUPF_boolean"]))*(~food_noalcnovit_female["lesshealthy"])*(food_noalcnovit_female['TotalGrams']))/np.sum(food_noalcnovit_female['TotalGrams'])
value_notupf_lesshealthy = np.sum(((~food_noalcnovit_female["NOVA_UPF_NUPF_boolean"]))*((food_noalcnovit_female["lesshealthy"]))*(food_noalcnovit_female['TotalGrams']))/np.sum(food_noalcnovit_male['TotalGrams'])
value_upf_lesshealthy = np.sum(((food_noalcnovit_female["NOVA_UPF_NUPF_boolean"]))*(food_noalcnovit_female["lesshealthy"])*(food_noalcnovit_female['TotalGrams']))/np.sum(food_noalcnovit_female['TotalGrams'])
print (value_notupf_healthy)
print (value_upf_healthy)
print (value_notupf_lesshealthy)
print (value_upf_lesshealthy)


