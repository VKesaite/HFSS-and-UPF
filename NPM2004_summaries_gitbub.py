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
categories= {'47A', '47B','48A', '48B', '48C', '49A', '49B', '49C', '49D', '49E', '54A', '54B', '54C', '54D', '54E', '54F', '54G', '54H', '54I', '54J', '54K', '54L', '54M', '54N', '54P'}
food_noalcnovit=food[~food['SubFoodGroupCode'].isin(categories)]

# %% [markdown]
# Check the food sample size of the original data

# %%
len (food)

# %% [markdown]
# Check the food sample size after removing supplements and alcohol

# %%
len (food_noalcnovit)

# %% [markdown]
# 1.1	Classify HFSS foods using the UK Nutrient Databank based on the NPM 2004/5 version. 

# %% [markdown]
# We do not have veg and fruit purees in the NDNS dataset, so we first create these two categories

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
# Calculate vegetable variable by adding up all the vegetables in the dataset

# %%
food_noalcnovit.loc [:, "vegetable"] = food_noalcnovit["Tomatoesg"]+ food_noalcnovit["Brassicaceaeg"]+food_noalcnovit["YellowRedGreeng"]+food_noalcnovit["Beansg"]+food_noalcnovit["OtherVegg"]

# %% [markdown]
# Calculate fruit, veg, and nuts per 100g. Here we also add fruit and veg purees (as suggested by the NPM documentation we multiply all purees and dried nuts by 2)

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

# %% [markdown]
# Explanation for the coding for nova are provided elsewhere: Colombet Z., O’Flaherty M., Chavez-Ugalde Y. NOVA classification of the National Diet and Nutrition Survey, waves 1 to 11 (2008/09 to 2018/19): GitHub; 2023 [Available from: https://github.com/zoecolombet/NOVA_NDNS_code.

# %% [markdown]
# Check the percent in each nova category

# %%
food_noalcnovit.NOVANumNEW_agreement.value_counts(normalize=True)*100

# %% [markdown]
# Create a binary category: Ultra-processed food and drink products equals 1, and 0 otherwise (unprocessed or minimally processed foods, processed culinary ingredients, processed foods)

# %%
food_noalcnovit.loc[:, 'NOVA_UPF_NUPF_boolean'] = (food_noalcnovit['NOVANumNEW_agreement'] == 'Ultra-processed food and drink products')

# %% [markdown]
# Check the percentage of foods that fall into UPF and non-UPF

# %%
food_noalcnovit.NOVA_UPF_NUPF_boolean.value_counts(normalize=True)*100

# %% [markdown]
# Classify HFSS foods using NDNS food database based on the NPM 2004/5 version. This is based on the description provided in the documentation on the NPM. (Available at: https://www.gov.uk/government/publications/the-nutrient-profiling-model)

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
# Calculate total A POINTS

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
# Impose some conditions on proteins here: If a food or drink scores 11 or more ‘A’ points then it cannot score points for protein unless it also scores 5 points for fruit, vegetables and nuts.

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

# %% [markdown]
# For food: less healthy (HFSS==1) is assigned when total points is 4 or greater

# %%
food_noalcnovit.loc[:, "lesshealthy"] = food_noalcnovit["total_points"] >= 4 

# %% [markdown]
# Drinks need to be done separately, and they are calculated below.

# %% [markdown]
# Create a drink set - include all drink names (Remove fruit juice and smoothies since they are calculated elsewhere)

# %%
drink_set={'SEMI SKIMMED MILK','SKIMMED MILK', 'SOFT DRINKS NOT LOW CALORIE', 'SOFT DRINKS LOW CALORIE', 'COMMERCIAL TODDLERS FOODS AND DRINKS', 
       '1% Fat Milk', 'TEA COFFEE AND WATER', 'WHOLE MILK'} 

# %% [markdown]
# We are also including drinking yoghurt into the drinks list, so creating another drinks set for this

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

# %% [markdown]
# Drinks are less healthy (or HFSS) if the total points is 1 or greater

# %%
food_noalcnovit.loc[food_noalcnovit['MainFoodGroupDesc'].isin(drink_set), 'lesshealthy'] = food_noalcnovit.loc[food_noalcnovit['MainFoodGroupDesc'].isin(drink_set), "total_points"]>= 1 

# %%
food_noalcnovit.loc[food_noalcnovit['FoodName'].isin(drink_set2), 'lesshealthy'] = food_noalcnovit.loc[food_noalcnovit['FoodName'].isin(drink_set2), "total_points"]>= 1 

# %% [markdown]
# Calculate the proportion of foods and drinks that are less healthy (or HFSS)

# %%
np.sum(food_noalcnovit["lesshealthy"])/len(food_noalcnovit["lesshealthy"])

# %% [markdown]
# Calculate the proportion of foods and drinks that are UPF

# %%
np.sum(food_noalcnovit["NOVA_UPF_NUPF_boolean"])/len(food_noalcnovit["NOVA_UPF_NUPF_boolean"])

# %% [markdown]
# Calculate the proportion of UPFs that are also HFSS

# %%
np.sum((food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit["lesshealthy"]))/np.sum (food_noalcnovit["NOVA_UPF_NUPF_boolean"])

# %% [markdown]
# Create a table with UPF sample only

# %%
food_noalcnovit_upf = food_noalcnovit[food_noalcnovit['NOVA_UPF_NUPF_boolean']]

# %% [markdown]
# We also want to check what would be the percentage increase in HFSS and UPF overlap if we include artificial sweeteners and soft drinks, low calorie. 

# %%
food_noalcnovit_upf.loc[:,"LH_assd"] = (food_noalcnovit_upf["lesshealthy"])| (food_noalcnovit_upf['MainFoodGroupDesc'] == "ARTIFICIAL SWEETENERS")|(food_noalcnovit_upf['MainFoodGroupDesc'] == "SOFT DRINKS LOW CALORIE")

# %% [markdown]
# The proportion of UPFs that would be captured if the policy added artificial sweeteners and soft drinks, low calorie

# %%
np.sum(food_noalcnovit_upf["LH_assd"] )/len(food_noalcnovit_upf.loc[:,"LH_assd"] )

# %% [markdown]
# 'weighted' by energy

# %%
np.sum(food_noalcnovit_upf["LH_assd"]* food_noalcnovit_upf['Energykcal'])/np.sum(food_noalcnovit_upf['Energykcal'])

# %% [markdown]
# 'weighted' by weight

# %%
np.sum(food_noalcnovit_upf["LH_assd"]* food_noalcnovit_upf['TotalGrams'])/np.sum(food_noalcnovit_upf['TotalGrams'])

# %% [markdown]
# SUMMARIES FOR VENN DIAGRAMS

# %% [markdown]
# Here we quantify the proportion of food and drink items from NDNS that are: neither HFSS (NPM 2004/5) nor UPFs; UPFs only;
# HFSS only (NPM 2004/5); HFSS (NPM 2004/5) and UPFs based on the overall sample.
# For the summary graphs aggregate for food and drinks

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
# The proportion of energy from HFSS foods 

# %%
np.sum((food_noalcnovit["lesshealthy"]* food_noalcnovit['Energykcal']))/np.sum(food_noalcnovit['Energykcal'])

# %% [markdown]
# The proportion of energy from HFSS foods among UPF foods

# %%
np.sum((food_noalcnovit_upf["lesshealthy"]* food_noalcnovit_upf['Energykcal']))/np.sum(food_noalcnovit_upf['Energykcal'])#len(food_noalcnovit[food_noalcnovit["NOVA_UPF_NUPF_boolean"]])

# %% [markdown]
# The proportion of energy from UPF foods 

# %%
np.sum((food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit['Energykcal']))/np.sum(food_noalcnovit["Energykcal"])

# %% [markdown]
# The proportion of energy from HFSS foods among UPF foods

# %%
np.sum((food_noalcnovit_upf["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit_upf['Energykcal']))/np.sum(food_noalcnovit_upf["Energykcal"])

# %% [markdown]
# Proportion of energy derived from: neither HFSS (NPM 2004/5) nor UPFs; UPFs only; HFSS only (NPM 2004/5); HFSS (NPM 2004/5) and UPFs based on the overall sample.

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
# Proportion of weight derived from HFSS among UPF sample

# %%
np.sum((food_noalcnovit_upf["lesshealthy"]* food_noalcnovit_upf['TotalGrams']))/np.sum(food_noalcnovit_upf['TotalGrams'])

# %% [markdown]
# Proportion of weight derived from HFSS in all sample

# %%
np.sum((food_noalcnovit["lesshealthy"]* food_noalcnovit['TotalGrams']))/np.sum(food_noalcnovit['TotalGrams'])

# %% [markdown]
# Proportion of weight derived from UPFs in all sample

# %%
np.sum((food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit['TotalGrams']))/np.sum(food_noalcnovit["TotalGrams"])

# %% [markdown]
# Proportion of weight derived from: neither HFSS (NPM 2004/5) nor UPFs; UPFs only; HFSS only (NPM 2004/5); HFSS (NPM 2004/5) and UPFs based on the overall sample.

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
# Code for TABLES 1-3 in the main text: Examples of HFSS foods only (i.e., not UPFs), UPFs only, both or neither summaries with 'MainFoodGroupDesc' variable

# %% [markdown]
# HFSS only

# %%
food_groups_values = {}
for x in food_noalcnovit['MainFoodGroupDesc'].unique(): 
    value = np.sum((food_noalcnovit["MainFoodGroupDesc"] == x)*(food_noalcnovit["lesshealthy"])*(~food_noalcnovit["NOVA_UPF_NUPF_boolean"]))/np.sum ((food_noalcnovit["lesshealthy"])*(~food_noalcnovit["NOVA_UPF_NUPF_boolean"]))
    food_groups_values[x] = value   
    
food_groups_values_sorted = {x: value for x, value in sorted(food_groups_values.items(), key=lambda item: item[1])}    
acc = 0
for x in food_groups_values_sorted:
    value = food_groups_values_sorted[x]
    print (f'x: {x}, value: {value:.5f}')
    acc += value

print (acc)

# %% [markdown]
# UPF only 

# %%
food_groups_values = {}
for x in food_noalcnovit['MainFoodGroupDesc'].unique(): 
    value = np.sum((food_noalcnovit["MainFoodGroupDesc"] == x)*(food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(~food_noalcnovit["lesshealthy"]))/np.sum ((food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(~food_noalcnovit["lesshealthy"]))
    food_groups_values[x] = value   
    
food_groups_values_sorted = {x: value for x, value in sorted(food_groups_values.items(), key=lambda item: item[1])}    
for x in food_groups_values_sorted:
    value = food_groups_values_sorted[x]
    print (f'x: {x}, value: {value:.5f}')

# %% [markdown]
# HFSS and UPF

# %%
food_groups_values = {}
for x in food_noalcnovit['MainFoodGroupDesc'].unique(): 
    value = np.sum((food_noalcnovit["MainFoodGroupDesc"] == x)*(food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit["lesshealthy"]))/np.sum ((food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit["lesshealthy"]))
    food_groups_values[x] = value   
    
food_groups_values_sorted = {x: value for x, value in sorted(food_groups_values.items(), key=lambda item: item[1])}    
for x in food_groups_values_sorted:
    value = food_groups_values_sorted[x]
    print (f'x: {x}, value: {value:.5f}')

# %% [markdown]
# NEITHER UPF NOR HFSS

# %%
food_groups_values = {}
for x in food_noalcnovit['MainFoodGroupDesc'].unique(): 
    value = np.sum((food_noalcnovit["MainFoodGroupDesc"] == x)*(~food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(~food_noalcnovit["lesshealthy"]))/np.sum ((~food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(~food_noalcnovit["lesshealthy"]))
    food_groups_values[x] = value   
    
food_groups_values_sorted = {x: value for x, value in sorted(food_groups_values.items(), key=lambda item: item[1])}    
for x in food_groups_values_sorted:
    value = food_groups_values_sorted[x]
    print (f'x: {x}, value: {value:.5f}')

# %% [markdown]
# RERUN THE ABOVE BUT ADD 'ENERGY' INTO THE EQUATION

# %% [markdown]
# HFSS ONLY

# %%
food_groups_values = {}
for x in food_noalcnovit['MainFoodGroupDesc'].unique(): 
    value = np.sum((food_noalcnovit["MainFoodGroupDesc"] == x)*(food_noalcnovit["lesshealthy"])*(~food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit['Energykcal']))/np.sum((food_noalcnovit["lesshealthy"])*(~food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit['Energykcal']))
    food_groups_values[x] = value 
    
food_groups_values_sorted = {x: value for x, value in sorted(food_groups_values.items(), key=lambda item: item[1])}    
for x in food_groups_values_sorted:
    value = food_groups_values_sorted[x]    
    print (f'x: {x}, value: {value:.5f}')

# %% [markdown]
# UPF ONLY

# %%
food_groups_values = {}
for x in food_noalcnovit['MainFoodGroupDesc'].unique(): 
    value = np.sum((food_noalcnovit["MainFoodGroupDesc"] == x)*(food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(~food_noalcnovit["lesshealthy"])*(food_noalcnovit['Energykcal']))/np.sum((~food_noalcnovit["lesshealthy"])*(food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit['Energykcal']))
    food_groups_values[x] = value 
    
food_groups_values_sorted = {x: value for x, value in sorted(food_groups_values.items(), key=lambda item: item[1])}    
for x in food_groups_values_sorted:
    value = food_groups_values_sorted[x]    
    print (f'x: {x}, value: {value:.5f}')

# %% [markdown]
# HFSS AND UPF

# %%
food_groups_values = {}
for x in food_noalcnovit['MainFoodGroupDesc'].unique(): 
    value = np.sum((food_noalcnovit["MainFoodGroupDesc"] == x)*(food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit["lesshealthy"])*(food_noalcnovit['Energykcal']))/np.sum ((food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit["lesshealthy"])*(food_noalcnovit['Energykcal']))
    food_groups_values[x] = value 
    
food_groups_values_sorted = {x: value for x, value in sorted(food_groups_values.items(), key=lambda item: item[1])}    
for x in food_groups_values_sorted:
    value = food_groups_values_sorted[x]    
    print (f'x: {x}, value: {value:.5f}')

# %% [markdown]
# NEITHER UPF NOR HFSS

# %%
food_groups_values = {}
for x in food_noalcnovit['MainFoodGroupDesc'].unique(): 
    value = np.sum((food_noalcnovit["MainFoodGroupDesc"] == x)*(~food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(~food_noalcnovit["lesshealthy"])*(food_noalcnovit['Energykcal']))/np.sum ((~food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(~food_noalcnovit["lesshealthy"])*(food_noalcnovit['Energykcal']))
    food_groups_values[x] = value 
    
food_groups_values_sorted = {x: value for x, value in sorted(food_groups_values.items(), key=lambda item: item[1])}    
for x in food_groups_values_sorted:
    value = food_groups_values_sorted[x]    
    print (f'x: {x}, value: {value:.5f}')

# %% [markdown]
# BY WEIGHT

# %% [markdown]
# HFSS ONLY

# %%
food_groups_values = {}
for x in food_noalcnovit['MainFoodGroupDesc'].unique(): 
    value = np.sum((food_noalcnovit["MainFoodGroupDesc"] == x)*(food_noalcnovit["lesshealthy"])*(~food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit['TotalGrams']))/np.sum ((food_noalcnovit["lesshealthy"])*(~food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit['TotalGrams']))
    food_groups_values[x] = value 
    
food_groups_values_sorted = {x: value for x, value in sorted(food_groups_values.items(), key=lambda item: item[1])}    
for x in food_groups_values_sorted:
    value = food_groups_values_sorted[x]    
    print (f'x: {x}, value: {value:.5f}')

# %% [markdown]
# UPF ONLY

# %%
food_groups_values = {}
for x in food_noalcnovit['MainFoodGroupDesc'].unique(): 
    value = np.sum((food_noalcnovit["MainFoodGroupDesc"] == x)*(food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(~food_noalcnovit["lesshealthy"])*(food_noalcnovit['TotalGrams']))/np.sum ((food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(~food_noalcnovit["lesshealthy"])*(food_noalcnovit['TotalGrams']))
    food_groups_values[x] = value 
    
food_groups_values_sorted = {x: value for x, value in sorted(food_groups_values.items(), key=lambda item: item[1])}    
for x in food_groups_values_sorted:
    value = food_groups_values_sorted[x]    
    print (f'x: {x}, value: {value:.5f}')

# %% [markdown]
# UPF AND HFSS

# %%
food_groups_values = {}
for x in food_noalcnovit['MainFoodGroupDesc'].unique(): 
    value = np.sum((food_noalcnovit["MainFoodGroupDesc"] == x)*(food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit["lesshealthy"])*(food_noalcnovit['TotalGrams']))/np.sum ((food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit["lesshealthy"])*(food_noalcnovit['TotalGrams']))
    food_groups_values[x] = value 
    
food_groups_values_sorted = {x: value for x, value in sorted(food_groups_values.items(), key=lambda item: item[1])}    
for x in food_groups_values_sorted:
    value = food_groups_values_sorted[x]    
    print (f'x: {x}, value: {value:.5f}')

# %% [markdown]
# NEITHER UPF NOR HFSS

# %%
food_groups_values = {}
for x in food_noalcnovit['MainFoodGroupDesc'].unique(): 
    value = np.sum((food_noalcnovit["MainFoodGroupDesc"] == x)*(~food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(~food_noalcnovit["lesshealthy"])*(food_noalcnovit['TotalGrams']))/np.sum ((~food_noalcnovit["NOVA_UPF_NUPF_boolean"])*(~food_noalcnovit["lesshealthy"])*(food_noalcnovit['TotalGrams']))
    food_groups_values[x] = value 
    
food_groups_values_sorted = {x: value for x, value in sorted(food_groups_values.items(), key=lambda item: item[1])}    
for x in food_groups_values_sorted:
    value = food_groups_values_sorted[x]    
    print (f'x: {x}, value: {value:.5f}')

# %% [markdown]
# Sub-group analyses: robustness check (SEE TABLE S1 in the appendix)

# %% [markdown]
# Analyses for males

# %%
food_noalcnovit_male = food_noalcnovit[food_noalcnovit['Sex']==1]

# %% [markdown]
# Proportion of HFSS foods 

# %%
np.sum(food_noalcnovit_male["lesshealthy"])/len(food_noalcnovit_male["lesshealthy"])

# %% [markdown]
# Proportion of UPF foods

# %%
np.sum(food_noalcnovit_male["NOVA_UPF_NUPF_boolean"])/len(food_noalcnovit_male["NOVA_UPF_NUPF_boolean"])

# %% [markdown]
# Proportion of UPFs and HFSS among UPF sample

# %%
np.sum((food_noalcnovit_male["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit_male["lesshealthy"]))/len(food_noalcnovit_male[food_noalcnovit_male["NOVA_UPF_NUPF_boolean"]])

# %% [markdown]
# The proportion of food and drink items from NDNS that are: neither HFSS (NPM 2004/5) nor UPFs; UPFs only;
# HFSS only (NPM 2004/5); HFSS (NPM 2004/5) and UPFs based on male sample.

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
# BY ENERGY

# %% [markdown]
# Proportion of energy derived from HFSS 

# %%
np.sum((food_noalcnovit_male["lesshealthy"])*(food_noalcnovit_male['Energykcal']))/np.sum(food_noalcnovit_male["Energykcal"])

# %% [markdown]
# Proportion of energy derived from UPFs 

# %%
np.sum((food_noalcnovit_male["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit_male['Energykcal']))/np.sum(food_noalcnovit_male["Energykcal"])

# %% [markdown]
# Create a UPF subsample

# %%
food_noalcnovit_upf_male = food_noalcnovit_male[food_noalcnovit_male['NOVA_UPF_NUPF_boolean']]

# %% [markdown]
# Proportion of energy derived from HFSS based on UPF sample

# %%
np.sum((food_noalcnovit_upf_male["lesshealthy"])*(food_noalcnovit_upf_male['Energykcal']))/np.sum (food_noalcnovit_upf_male["Energykcal"])

# %% [markdown]
# The proportion of food and drink items from NDNS that are: neither HFSS (NPM 2004/5) nor UPFs; UPFs only;
# HFSS only (NPM 2004/5); HFSS (NPM 2004/5) and UPFs based on the overall sample.

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
# BY WEIGHT

# %% [markdown]
# Proportion of weight from HFSS

# %%
np.sum((food_noalcnovit_male["lesshealthy"])*(food_noalcnovit_male['TotalGrams']))/np.sum(food_noalcnovit_male["TotalGrams"])

# %% [markdown]
# Proportion of weight from UPFs

# %%
np.sum((food_noalcnovit_male["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit_male['TotalGrams']))/np.sum(food_noalcnovit_male["TotalGrams"])

# %% [markdown]
# Sub-group analyses for UPF only

# %%
food_noalcnovit_upf_male = food_noalcnovit_male[food_noalcnovit_male['NOVA_UPF_NUPF_boolean']]

# %% [markdown]
# Overlap between UPFs and HFSS weighted 

# %%
np.sum((food_noalcnovit_upf_male["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit_upf_male["lesshealthy"])*(food_noalcnovit_upf_male['TotalGrams']))/np.sum (food_noalcnovit_upf_male["TotalGrams"])

# %% [markdown]
# The proportion of food and drink items from NDNS that are: neither HFSS (NPM 2004/5) nor UPFs; UPFs only;
# HFSS only (NPM 2004/5); HFSS (NPM 2004/5) and UPFs based on the overall sample.
# For the summary graphs aggregate for food and drinks

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
# For females

# %%
food_noalcnovit_female = food_noalcnovit[food_noalcnovit['Sex']==2]

# %% [markdown]
# Proportion of HFSS foods

# %%
np.sum(food_noalcnovit_female["lesshealthy"])/len(food_noalcnovit_female["lesshealthy"])

# %% [markdown]
# Proportion of UPFs

# %%
np.sum(food_noalcnovit_female["NOVA_UPF_NUPF_boolean"])/len(food_noalcnovit_female["NOVA_UPF_NUPF_boolean"])

# %% [markdown]
# Proportion of UPFs and HFSS among UPF sample

# %%
np.sum((food_noalcnovit_female["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit_female["lesshealthy"]))/len(food_noalcnovit_female[food_noalcnovit_female["NOVA_UPF_NUPF_boolean"]])

# %% [markdown]
# The proportion of food and drink items from NDNS that are: neither HFSS (NPM 2004/5) nor UPFs; UPFs only;
# HFSS only (NPM 2004/5); HFSS (NPM 2004/5) and UPFs based on the overall sample.
# For the summary graphs aggregate for food and drinks

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
# BY WEIGHT

# %% [markdown]
# Proportion of weight derived from HFSS

# %%
np.sum((food_noalcnovit_female["lesshealthy"])*(food_noalcnovit_female['TotalGrams']))/np.sum(food_noalcnovit_female["TotalGrams"])

# %% [markdown]
# Proportion of weight derived from UPFs

# %%
np.sum((food_noalcnovit_female["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit_female['TotalGrams']))/np.sum(food_noalcnovit_female["TotalGrams"])

# %% [markdown]
# Create a UPF sample only

# %%
food_noalcnovit_upf_female = food_noalcnovit_female[food_noalcnovit_female['NOVA_UPF_NUPF_boolean']]

# %% [markdown]
# Overlap between UPF anf HFSS by weight

# %%
np.sum((food_noalcnovit_upf_female["NOVA_UPF_NUPF_boolean"])*(food_noalcnovit_upf_female["lesshealthy"])*(food_noalcnovit_upf_female['TotalGrams']))/np.sum (food_noalcnovit_upf_female["TotalGrams"])

# %% [markdown]
# The proportion of food and drink items from NDNS that are: neither HFSS (NPM 2004/5) nor UPFs; UPFs only;
# HFSS only (NPM 2004/5); HFSS (NPM 2004/5) and UPFs based on the overall sample.
# For the summary graphs aggregate for food and drinks

# %%
value_notupf_healthy = np.sum(((~food_noalcnovit_female["NOVA_UPF_NUPF_boolean"]))*(~food_noalcnovit_female["lesshealthy"])*(food_noalcnovit_female['TotalGrams']))/np.sum(food_noalcnovit_female['TotalGrams'])
value_upf_healthy = np.sum(((food_noalcnovit_female["NOVA_UPF_NUPF_boolean"]))*(~food_noalcnovit_female["lesshealthy"])*(food_noalcnovit_female['TotalGrams']))/np.sum(food_noalcnovit_female['TotalGrams'])
value_notupf_lesshealthy = np.sum(((~food_noalcnovit_female["NOVA_UPF_NUPF_boolean"]))*((food_noalcnovit_female["lesshealthy"]))*(food_noalcnovit_female['TotalGrams']))/np.sum(food_noalcnovit_male['TotalGrams'])
value_upf_lesshealthy = np.sum(((food_noalcnovit_female["NOVA_UPF_NUPF_boolean"]))*(food_noalcnovit_female["lesshealthy"])*(food_noalcnovit_female['TotalGrams']))/np.sum(food_noalcnovit_female['TotalGrams'])
print (value_notupf_healthy)
print (value_upf_healthy)
print (value_notupf_lesshealthy)
print (value_upf_lesshealthy)


