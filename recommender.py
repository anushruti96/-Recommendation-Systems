import csv
import os
from datetime import date
import numpy as np
from numpy.linalg import norm
import random
import math
import statsmodels.api as sm
import pandas
import pandas as pd
import warnings
import math
import heapq
from operator import itemgetter
import json

#Ignore deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#GLOBAL VARIABLES
debug = False
KNOWLEDGE_FILTER_SIZE = 1000
COLLABORATIVE_FILTER_SIZE = 50


#Load all recipe data
with open('./StoredData/epicurious.csv', 'r') as read_obj:
    recipes = list(csv.reader(read_obj))
#remove the header
recipes.pop(0)

#Load rating data
with open('./StoredData/ratings.csv', 'r') as read_obj:
    ratings = list(csv.reader(read_obj))

#Load the occurences data
with open('./StoredData/occurences.csv', 'r') as read_obj:
    occurences = list(csv.reader(read_obj))



#====================LEARNING AGENT====================
def updatePreferences(pid, rid, rating):

    #Find the appropriate row
    rowNum = 0
    for row in ratings:
        if(row[0]==pid):
            break
        rowNum+=1

    #Loop through all of the tags in the recipe
    cellNum=1
    for cell in recipes[rid][6:]:

        #Check if this tag applies
        if(float(cell)==1.0):


            #Calculate the new average
            totalScore = float(ratings[rowNum][cellNum])  * float(occurences[rowNum][cellNum])

            #Add in the new ratings
            totalScore += float(rating)

            #Calculate the new average
            totalScore = totalScore / (float(occurences[rowNum][cellNum])+1.0)

            #Save the data
            ratings[rowNum][cellNum]=totalScore
            occurences[rowNum][cellNum] = int(occurences[rowNum][cellNum])+1

        #Increment cellNum
        cellNum+=1

    #Adjust for the recipe
    #Calculate the new average
    totalScore = float(ratings[rowNum][rid+675])  * float(occurences[rowNum][rid+675])

    #Add in the new ratings
    totalScore += float(rating)

    #Calculate the new average
    totalScore = totalScore / (float(occurences[rowNum][rid+675])+1.0)

    #Save the data
    ratings[rowNum][rid+675]=totalScore

    occurences[rowNum][rid+675] = int(occurences[rowNum][rid+675])+1

    #Save the files
    with open('./StoredData/ratings.csv', 'w') as write_obj:
        writer = csv.writer(write_obj)
        writer.writerows(ratings)

    with open('./StoredData/occurences.csv', 'w') as write_obj:
        writer = csv.writer(write_obj)
        writer.writerows(occurences)

#Method to get and save the rating of the recipe
def coldStartRate(recipe, rid, pid):

    rating = input('How would you rate (1-5) the following recipe (please only provide a number as input): ' + recipe + '\n1. Would not make \n2. Probably would not make \n3. Unsure \n4. Probably would make \n5. Would Make \n')

    #Find the tags used in this recipe
    updatePreferences(pid, rid, rating)

#====================COMMUNICATION AGENT====================
#Method to get the users features like allergen data
def extract_features(pid):

    #Store habits in a list
    habits = []
    print('Please answer the following questions about your dietary habits: ')

    habits.append('N')
    habits.append(input('Are you vegetarian? [Y/N] '))
    habits.append(input('Are you vegan? [Y/N] '))
    habits.append(input('Are you gluten free? [Y/N] '))
    habits.append(input('Are you soy free? [Y/N] '))
    habits.append(input('Are you dairy free? [Y/N] '))

    #convert habits to binary
    features = [pid]
    for restrict in habits:
        if(restrict.upper()=='Y'):
            features.append(1)
        else:
            features.append(0)

    features.append(float(input('How likely are you to repeat a recipe? On a scale from 0 (unlikely) to 1 (likely) ')))

    #Save the dietary restrictions
    with open('./StoredData/users.csv', 'a') as write_obj:
        writer = csv.writer(write_obj)
        writer.writerow(features)

#Get the user's physiological data
def physiological(pid):

    #Store the data
    bodyData = [pid]
    print()
    print('Please answer the following questions about your physiology, goals, and activity level: ')

    bodyData.append(input('Please provide your height (inches): '))
    bodyData.append(input('Please provide your weight (lbs): '))
    bodyData.append(input('Please provide your activity level [type a number 1-3]: \n1. Not very active (<4,000 steps a day) \n2. Moderately Active (<8,000 steps a day) \n3. Very Active (>8,000 steps a day)\n'))
    bodyData.append(input('Please provide your weight goal: \n1. Lose Weight \n2. Maintain Weight \n3. Gain Weight\n'))
    bodyData.append(input('Are you male or female? [male / female] '))
    bodyData.append(input('What is your age? '))
    print()
    with open('./StoredData/bodyData.csv', 'a') as write_obj:
        writer = csv.writer(write_obj)
        writer.writerow(bodyData)

#Have the user rate the latest recipe
def getRating(pid, save = False):

    #Get the rating rating history
    with open('./StoredData/'+pid+'/recipeHistory.csv', 'r') as read_obj:
        recipeHistory = list(csv.reader(read_obj))

    #Get the rating for the most recent recommendation
    rating = int(input('How would you rate your most recent recipe? It was: ' + recipeHistory[-1][0] + '\n1. Strongly Disliked \n2. Disliked \n3. Unsure \n4. Liked \n5. Strongly Liked \n'))

    #Update the preferences
    updatePreferences(pid, int(recipeHistory[-1][1]), rating)

    #Save the users rating
    if(save):
        if(os.path.exists('./StoredData/'+pid+'/ratingHistory.csv')):
            with open('./StoredData/'+pid+'/ratingHistory.csv', 'a') as write_obj:
                writer = csv.writer(write_obj)
                writer.writerow([recipeHistory[-1][0], rating])
        else:
            with open('./StoredData/'+pid+'/ratingHistory.csv', 'w') as write_obj:
                writer = csv.writer(write_obj)
                writer.writerow([recipeHistory[-1][0], rating])

#Have the user login
def login(pid = None, rate=True):

    #Get the Partipant ID
    if(pid==None):
        pid = input('Please enter your user ID: ')

    if(os.path.exists('./StoredData/'+pid)):
        print('Providing recipe for: '+ pid)

        if(rate):
            #Ask participant to rate their last recipe
            getRating(pid, True)

    else:

        #Get the features data
        extract_features(pid)

        #Get the physiological data
        physiological(pid)

        #Add a row to ratings and occurences for the user
        tempData = [pid]
        for cell in ratings[0][1:]:
            tempData.append(0)
        ratings.append(tempData)

        #Add a row to ratings and occurences for the user
        tempData = [pid]
        for cell in ratings[0][1:]:
            tempData.append(0)
        occurences.append(tempData)

        #Get the recipes
        with open('./StoredData/coldStartRecipes.csv', 'r') as read_obj:

            #get the coldStartRecipes
            coldStartRecipes = list(csv.reader(read_obj))

        #Loop through the recipes
        print()
        print('You will now be presented with 25 different recipes. These are not recommendations, nor are they unique to you.')
        print('They are the same for all users. Please rate each recipe, so that the system can learn your preferences.')
        print()
        recipe = 1
        for row in coldStartRecipes:
            print('===========Recipe '+str(recipe)+' of ' +str(len(coldStartRecipes))+'===========')
            coldStartRate(row[0], int(row[1]), pid)
            recipe+=1

        #Log todays ratings
        #Make a folder for the user's data
        os.mkdir('./StoredData/'+pid)

        #Create a folder to store the time changes
        with open('./StoredData/'+pid+'/timeChange.csv', 'w') as write_obj:
            writer = csv.writer(write_obj)
            writer.writerows([['Time Bin']+ ratings[0][1:], [0]+ratings[-1][1:]])

        with open('./StoredData/'+pid+'/startDate.txt', 'w') as f:
            f.write(str(date.today()))

        with open('./StoredData/'+pid+'/recipeHistory.csv', 'w') as f:
            pass

        with open('./StoredData/'+pid+'/lastReccomendations', 'w') as write_obj:
            pass

    #Save the data
    with open('./StoredData/ratings.csv', 'w') as write_obj:
        writer = csv.writer(write_obj)
        writer.writerows(ratings)

    with open('./StoredData/occurences.csv', 'w') as write_obj:
        writer = csv.writer(write_obj)
        writer.writerows(occurences)

    return pid

#Communicate the recipes at the end
def communicateRecipes(pid, recipeRecs):

    print('Your possible recommendations are:')
    if(debug):
        print(recipeRecs)

    for i in range(len(recipeRecs)):

        print("======Recipe ", i, "======")
        print("Recipe:", recipeRecs[i][1])

        if recipeRecs[i][5] == "collab":
            print("Match based on nutrition:", recipeRecs[i][2])
            print("Match based on users like you:", recipeRecs[i][3])

        else:
            print("Match based on nutrition: N/A")
            print("Match based on users like you: N/A")

        print("Match based on other items you like:", recipeRecs[i][4])

        if debug:
            print(recipeRecs[i][5])

        print()

    choice = int(input('Please type the recipe number of the recipe you would like to make: '))

    with open('./StoredData/'+pid+'/lastReccomendations', 'r') as read_obj:
        history = list(csv.reader(read_obj))

    if(len(history)==100):
        history = history[5:]

    for rec in recipeRecs:
        history.append(rec)

    with open('./StoredData/'+pid+'/lastReccomendations', 'w') as write_obj:
        writer = csv.writer(write_obj)
        writer.writerows(history)

    for i in range(len(recipeRecs)):

        #Check if this is the user's choice
        if(i==choice):

            #Log the recipe
            with open('./StoredData/'+pid+'/recipeHistory.csv', 'a') as write_obj:
                writer = csv.writer(write_obj)
                writer.writerow([recipeRecs[i][1], recipeRecs[i][0]])

        else:

            #Give the other recipes an average rating
            updatePreferences(pid, recipeRecs[i][0], 3)

#====================RECOMMENDER AGENT====================
#=====Knowledge Filter=====
def knowledgeBasedFilter(user_name) :
    df= pandas.read_csv("./StoredData/epicurious.csv")
    user_file=pandas.read_csv("./StoredData/users.csv")
    body_data=pandas.read_csv("./StoredData/bodyData.csv")
    filter_res=[]
    if not os.stat('./StoredData/'+user_name+'/lastReccomendations').st_size == 0:
        recent_recipes=pandas.read_csv('./StoredData/'+user_name+'/lastReccomendations',header=None)
        filter_res=recent_recipes[1].tolist()
    user_details=[]
    user_details=user_file.loc[user_file['user'] == user_name]
    list_of_allergies=[]
    for (columnName, columnData) in user_details.iteritems():
        obj=columnData.values
        if(obj==1) :
            list_of_allergies.append(columnName)
    user_details=[]
    user_details=body_data.loc[body_data['user'] == user_name]
    if ((user_details['gender'].values[0].lower())=="male"):
        BMR=(10 * user_details['weight'].values[0].astype('int')) + (6.25 * user_details['height'].values[0].astype('int')) - (5* user_details['age'].values[0].astype('int'))+5

    if ((user_details['gender'].values[0].lower())=="female"):
        BMR=(10 * user_details['weight'].values[0].astype('int')) + (6.25 * user_details['height'].values[0].astype('int')) - (5 * user_details['age'].values[0].astype('int'))-161
    if (0<=user_details['activity level'].values[0]<2) :
        calorie_intake=BMR
    elif (2<=user_details['activity level'].values[0]<4) :
        calorie_intake=1.55*BMR
    elif (4<=user_details['activity level'].values[0]<6) :
        calorie_intake=1.725*BMR
    else :
        calorie_intake=BMR
    calorie_intake_per_meal= BMR/4
    recipe_satisfy_calorie=[]
    for ind in df.index:
        if (((calorie_intake_per_meal-50)<=(df['calories'][ind].astype(int))<=(calorie_intake_per_meal+50)) and df['calories'][ind]!=None) :
            df.at[ind, 'score'] = 40
            recipe_satisfy_calorie.append((ind))
        elif (((calorie_intake_per_meal-100)<=(df['calories'][ind].astype(int))<=(calorie_intake_per_meal+100)) and df['calories'][ind]!=None) :
            df.at[ind, 'score'] = 30
            recipe_satisfy_calorie.append((ind))
        elif (((calorie_intake_per_meal-200)<=(df['calories'][ind].astype(int))<=(calorie_intake_per_meal+200)) and df['calories'][ind]!=None) :
            df.at[ind, 'score'] = 20
            recipe_satisfy_calorie.append((ind))
    total_recipe_count=len(recipe_satisfy_calorie)
    if 'vegetarian' in list_of_allergies :
        count_veg=0
        for ind in recipe_satisfy_calorie :
            if(df['vegetarian'][ind]==1) :
                a=(df.at[ind, 'score'])
                df.at[ind, 'score'] = a+60
            else :
                count_veg+=1
                df.at[ind, 'score'] =-1
        if (total_recipe_count==count_veg):
            for ind in df.index :
                if(df['vegetarian'][ind]==1) :
                    a=(df.at[ind, 'score'])
                    df.at[ind, 'score']=a+50

    if 'vegan' in list_of_allergies :
        count_vegan=0
        for ind in recipe_satisfy_calorie :
            if(df['vegan'][ind]==1) and (df['score'][ind]!=-1) :
                a=(df.at[ind, 'score'])
                df.at[ind, 'score']=a+60
                if(df.at[ind, 'score']>100):
                    df.at[ind, 'score']=100
            else :
                count_vegan=count_vegan+1
                df.at[ind, 'score']=-1
        if (total_recipe_count==count_vegan):
            for ind in df.index :
                if(df['vegan'][ind]==1) :
                    a=(df.at[ind, 'score'])
                    df.at[ind, 'score']=a+50
                    if(df.at[ind, 'score']>100):
                        df.at[ind, 'score']=100

    if 'gluten-free' in list_of_allergies :
        count_gluten=0
        for ind in recipe_satisfy_calorie :
            if(df['wheat/gluten-free'][ind]==1) and (df['score'][ind]!=-1) :
                a=(df.at[ind, 'score'])
                df.at[ind, 'score']=a+60
                if(df.at[ind, 'score']>100):
                    df.at[ind, 'score']=100
        else :
            count_gluten+=1
            df.at[ind, 'score']=-1
        if (total_recipe_count==count_gluten):
            for ind in df.index :
                if(df['wheat/gluten-free'][ind]==1) :
                    a=(df.at[ind, 'score'])
                    df.at[ind, 'score']=a+50
                    if(df.at[ind, 'score']>100):
                        df.at[ind, 'score']=100
    if 'soy-free' in list_of_allergies :
        count_soy=0
        for ind in recipe_satisfy_calorie :
            if(df['soy free'][ind]==1) and (df['score'][ind]!=-1) :
                a=(df.at[ind, 'score'])
                df.at[ind, 'score']=a+60
                if(df.at[ind, 'score']>100):
                    df.at[ind, 'score']=100
            else :
                df.at[ind, 'score']=-1
                count_soy+=1
        if (total_recipe_count==count_soy):
            for ind in df.index :
                if(df['soy free'][ind]==1) :
                    a=(df.at[ind, 'score'])
                    df.at[ind, 'score']=a+50
                    if(df.at[ind, 'score']>100):
                        df.at[ind, 'score']=100

    if 'dairy-free' in list_of_allergies :
        for ind in recipe_satisfy_calorie :
            count_dairy=0
            if(df['dairy free'][ind]==1) and (df['score'][ind]!=-1) :
                a=(df.at[ind, 'score'])
                df.at[ind, 'score']=a+60
                if(df.at[ind, 'score']>100):
                    df.at[ind, 'score']=100
            else :
                df.at[ind, 'score']=-1
                count_dairy+=1
        if (total_recipe_count==count_dairy):
            for ind in df.index :
                if(df['dairy free'][ind]==1) :
                    a=(df.at[ind, 'score'])
                    df.at[ind, 'score']=a+50
                    if(df.at[ind, 'score']>100):
                        df.at[ind, 'score']=100

    if len(list_of_allergies)==0 :
        for ind in recipe_satisfy_calorie :
            a=(df.at[ind, 'score'])
            df.at[ind, 'score']=a+80
            if(df.at[ind, 'score']>100):
                df.at[ind, 'score']=100
    for name in filter_res:
        for ind in recipe_satisfy_calorie :
            a=(df.at[ind, 'title'])
            if(name==a) :
                df.at[ind, 'score']=-1
    df1=df.sort_values(['score'],ascending=[False])
    df1.drop(df1[df1['score']== -1].index, inplace = True)
    df1=df1.nlargest(1000, ['score'])
    df1.drop(df1.columns.difference(['title','score']), 1, inplace=True)
    list_of_recipes=[df1.columns.tolist()] + df1.reset_index().values.tolist()
    del list_of_recipes[0]
    return list_of_recipes


#=====Collaborative Filter=====
def overallBias():
    allRatings = []
    for row in ratings[1:]:
        for cell in row[1:]:
            if(not float(cell)==0.0):
                allRatings.append(float(cell))
    return np.mean(allRatings)

def userBias(pid):
    allRatings = []
    count=0
    for row in ratings[1:]:
        if(row[0]==pid):
            for cell in row[1:]:
                if(not float(cell)==0.0):
                    count+=1
                    allRatings.append(float(cell))
    return np.mean(allRatings), count

def itemBias(item):


    allRatings = []
    for row in ratings[1:]:
        if(not float(row[item+1])==0.0):
            allRatings.append(float(row[item+1]))

    if(len(allRatings)==0):
        return 0
    return np.mean(allRatings)

def binBias(data, rowNum):
    allRatings = []
    for cell in data[rowNum][1:]:
        if(not float(cell)==0.0):
            allRatings.append(float(cell))
    return np.mean(allRatings)

#Gets the cossine similarity of all users
def cosineSimilarity(pid):

    #Get the current users row
    for row in ratings:
        if(row[0]==pid):
            currentRow = row[1:]
            break

    #Get cosine scores for each user
    cosines = []
    for row in ratings:
        if('User' not in row[0] and pid not in row[0]):
            tempRow = row[1:]
            #rint(row[0])
            #Run through each list and remove all recipes that neither user has rated
            #This is being done just to lower similarity
            currentScores = []
            tempScores = []
            for i in range(len(currentRow)):
                if(not (float(currentRow[i]) ==0.0) and not(float(tempRow[i])==0.0)):
                    currentScores.append(float(currentRow[i]))
                    tempScores.append(float(tempRow[i]))

            #Calculate the cosine similarity
            # print(currentScores)
            # print(tempScores)
            cosines.append(math.cos(np.dot(currentScores, tempScores)/(norm(currentScores)*norm(tempScores))))

    return cosines

#Calculate b_{u,j} for the collaborative filter
def timeBias(rid, pid, universalBias, uBias, iBias):

    #Get the users data
    with open('./StoredData/'+pid+'/timeChange.csv', 'r') as read_obj:
        binnedData = list(csv.reader(read_obj))

    #Start by adding the biases we already know together
    bias = universalBias+uBias+iBias

    #====Bias from change over time====
    #Get times the user has rated this item
    x = []
    y = []
    counter = 0
    for row in binnedData[1:]:
        if(not float(row[675+int(float(rid))])==0.0):
            x.append(int(row[0]))
            y.append(float(row(row[675+int(rid)])))
            lastBin = counter
        counter+=1

    if(len(x)<2):
        return 0

    x = sm.add_constant(x)
    model = sm.OLS(y, x)

    #Get the slope
    slope = model.fit().params[1]
    intercept = model.fit().params[0]

    bias+=slope

    #Bias user has for the item at this time
    predicted = intercept+slope*lastBin
    bias += predicted

    #Bin-based bias (average of all ratings from this bin)
    bias += binBias(binnedData, lastBin)

    #Return the bias
    return bias

#Method to run the collaborative filter
#==Inputs==
#Recipes = list of recipes we can choose from
#pid = current user
#==Outputs==
#New list of recipes, size determined by COLLABORATIVE_FILTER_SIZE
def collaborativeFilter(recipes, pid):

    #Get the cossine similarity for each set of users
    cosines = cosineSimilarity(pid)

    #Randomize the possible recipes
    #This is being done so ties do not always return the same results
    random.shuffle(recipes)

    #Get the constant bias terms
    universalBias = overallBias()
    uBias, numOfRatings = userBias(pid)

    #Store all of the recipes and findings in a dataframe
    df = {'Recipe': [], 'RID': [], 'Prediction': [], 'Knowledge Scores': []}

    #Loop through all of the recipes
    for recipe in recipes:

        if(debug):
            print(recipe)

        iBias = itemBias(recipe[0])

        df['Recipe'].append(recipe[1])
        df['RID'].append(recipe[0])
        df['Knowledge Scores'].append(recipe[2])

        #Calculate the total starting bias
        bias = universalBias+uBias+iBias#+

        #Calculate the predicted score based on the users
        userInfluence = 0

        index = 0
        for row in ratings[1:]:

            #Check if the user has rated this item
             if(not float(row[recipe[0]+675])==0.0 and not row[0]==pid):

                 #Find time-bias
                 tBias = timeBias(row[1], row[0], universalBias, userBias(row[0]), iBias)

                 #Calculate the user influence
                 if(debug):
                     print(cosines)
                     print(index)
                     print(row[recipe[0]+675])
                 userInfluence += (float(row[recipe[0]+675]) - tBias) * cosines[index]
                 #Increment the user index
                 index+=1

        #Get the predicted value
        predicted = bias + (numOfRatings**-0.5) * userInfluence

        #Save the predicted value
        df['Prediction'].append(predicted)

    #Convert to dataframe
    df = pd.DataFrame(df)
    df = df.sort_values('Prediction', axis=0, ascending=False)
    df = df.values.tolist()

    bestScore = df[0][2]
    # bestScore = 0
    # for element in df:
    #     bestScore = max(bestScore, )

    #Take the top COLLABORATIVE_FILTER_SIZE recommendations and return them
    recommendations = []
    for i in range(COLLABORATIVE_FILTER_SIZE):
        recommendations.append([df[i][1], df[i][0], df[i][3], int((df[i][2]/bestScore)*100)])

    return recommendations


###########################
#=====Content Filter=====
###########################


# final recipes for recommendation
CONTENT_FILTER_SIZE = 5
SIMILAR_ITEM_COUNT = 3

# load all the required files

# print("bodyData -> user information. LOADING ... ")
# with open("./StoredData/bodyData.csv", 'r') as read_obj:
#     bodyData = list(csv.reader(read_obj))

# print("epicurious -> recipe macronutrients and ingredients. LOADING ... ")
# with open("./StoredData/epicurious.csv", 'r') as read_obj:
#     epicurious = list(csv.reader(read_obj))

# print("occurences -> user interaction. LOADING ... ")
# with open("./StoredData/occurences.csv", 'r') as read_obj:
#     occurences = list(csv.reader(read_obj))

# print("quickRef -> ids of recipes and ingridients. LOADING ... ")
# with open("./StoredData/quickRef.csv", 'r') as read_obj:
#     quickRef = list(csv.reader(read_obj))

# print("recipesAndIds -> recipe name and index. LOADING ... ")
with open("./StoredData/recipesAndIds.csv", 'r') as read_obj:

    recipesAndIds = list(csv.reader(read_obj))

    idsAndRecipes = {
        i[1]: i[0]
    for i in recipesAndIds}

    recipesAndIds = {
        i[0]: i[1]
    for i in recipesAndIds}


def top_n_dict_items(d, n):
    return dict(heapq.nlargest(n, d.items(), key=itemgetter(1)))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def final_results(recipes, recs, bias, user_id):

    # recipes = {tag: score}
    # recs = {tag: {
    #     id: score
    # }}

    # {id: score}

    with open(f"./StoredData/{user_id}/lastReccomendations", 'r') as read_obj:
        user_history = list(csv.reader(read_obj))
        user_history = set([int(i[0]) for i in user_history])

    #REMOVE
    # print("user_history", user_history)
    # print("recipes", recipes)
    # print("recs", recs)
    # print("\n\n Good \n\n")

    ans = {}

    for i in recipes:
        if i in user_history or i in ans:
            continue
        ans[i] = recipes[i] * bias

    for i in recs:
        for j in recs[i]:
            if j in user_history or j in ans:
                continue
            ans[j] = ans.get(j, 0) + (recs[i][j] * (1 - bias))

    # recs = zip(ids, softmax(scores))

    return top_n_dict_items(ans, CONTENT_FILTER_SIZE)


def get_previously_rated(user_id):

    # print("ratings -> user's ratings for every recipe in epicurious. LOADING ... ")
    with open("./StoredData/ratings.csv", 'r') as read_obj:
        list(csv.reader(read_obj))

    recipes = {}

    user_row = None

    for i in range(1, len(ratings)):

        if ratings[i][0] == user_id:
            user_row = ratings[i]
            break

    if user_row is None:
        print("No user row found")
        return {}

    for i in range(675, len(user_row)):

        rec = user_row[i]

        if float(rec) > 0:
            k = recipesAndIds[ratings[0][i]]
            recipes[k] = rec

    return recipes


def get_item_similar_item(recipes):

    # print("item_item co-relational matrix. LOADING ... ")
    # item_item = dd.read_csv("./StoredData/item_item.csv")
    # item_item = pd.read_csv("./StoredData/item_item.csv", chunksize = 1000000)
    # item_item = pd.concat(item_item)
    # # item_item = pd.read_csv("./StoredData/item_item.csv", engine='c')
    # # item_item = genfromtxt("./StoredData/item_item.csv", delimiter=',', dtype = int)
    with open("./StoredData/item_item.json", 'r') as read_obj:
        item_item = json.load(read_obj)

    # print(recipes)

    new_recipes = {}

    for i in recipes:

        # recipe_id = recipesAndIds[i]

        items = item_item[str(i)]

        new_recipes[i] = {
            k: v
        for k, v in items}

    return new_recipes


def weigh_intersection(content_dict, collaborative_list):

    recipes = {}

    for i in collaborative_list:

        if str(i[0]) not in content_dict:
            continue

        recipes[i[0]] = (0.33 * i[2]) + (0.33 * i[3]) + (0.33 * (float(content_dict[str(i[0])]) * 20))

    return recipes
    # return top_n_dict_items(recipes, CONTENT_FILTER_SIZE)


def weigh_current(collaborative_list):

    recipes = {}
    for i in collaborative_list:
        recipes[i[0]] = (0.5 * i[2]) + (0.5 * i[3])

    return recipes
    # return top_n_dict_items(recipes, CONTENT_FILTER_SIZE)


def contentFilter(recipes, user_id):

    # print("users -> user_id (row), name, and dietary restrictions for each user. LOADING ... ")
    with open("./StoredData/users.csv", 'r') as read_obj:
        users = list(csv.reader(read_obj))

    # recipes = [4 x N] {index, recipe_name, knowledge_score, collaborative_score}

    recipe_ids = {recipes[i][0]: i for i in range(len(recipes))}

    user = None
    for i in users:
        if i[0] == user_id:
            user = i
            break

    if user is None:
        print("No user found")
        return

    bias = float(user[-1])

    prev_intrcs = get_previously_rated(user_id)

    #REMOVE
    # print("\n prev_intrcs:", len(prev_intrcs), '\n')

    intersected = weigh_intersection(prev_intrcs, recipes)

    #REMOVE
    # print("\n intersected:", len(intersected), '\n')

    if len(intersected.keys()) >= 5:
        ixi_recs = get_item_similar_item(intersected)
        #REMOVE
        # print("\n wooow ixi_recs:", len(ixi_recs), '\n')
        final = final_results(intersected, ixi_recs, bias, user_id)

    else:
        new_recipes = weigh_current(recipes)
        #REMOVE
        # print("\n wooow new_recipes:", len(new_recipes), '\n')
        ixi_recs = get_item_similar_item(new_recipes)
        # ixi_recs = get_item_similar_item(prev_intrcs)

        final = final_results(new_recipes, ixi_recs, bias, user_id)

    # # gather information for top 5
    # final = [{
    #     "name": i[0], # use quick ref here
    #     "id": recipesAndIds[i[0]],
    #     "calories": epicurious[i[0]],
    #     "score": i[1]
    # } for i in final]

    ans = []

    for i in final:

        if i in recipe_ids:
            row = recipes[recipe_ids[i]]
            row.extend([final[i], "collab"])

        else:
            row = [i, idsAndRecipes[str(i)], None, None, final[i], "similar"]

        ans.append(row)

    return ans


###########################
#=====Content Filter End=====
###########################

def createRecommendation(pid):

    #Knowledge Filter
    reducedRecommendations = knowledgeBasedFilter(pid)

    # print(len(reducedRecommendations))

    #Collaborateive Filter
    reducedRecommendations = collaborativeFilter(reducedRecommendations, pid)

    if(debug):
        print(reducedRecommendations)

    # print(len(reducedRecommendations))

    #Content Filter
    reducedRecommendations = contentFilter(reducedRecommendations, pid)

    # print(len(reducedRecommendations))

    if(debug):
        print(reducedRecommendations)

    #Communicate
    communicateRecipes(pid, reducedRecommendations)
    for spam in range(100):
        print()

    login(pid)

#Main Method: run the entire recommender
def recommender():

    #Log the user in
    pid = login(rate=False)
    for repeat in range(10):

        print('Recipe '+ str(repeat))

        attempt = 0

        if(debug):
            print(attempt)

        while(attempt<5):
            try:
                createRecommendation(pid)
                break
            except:
                attempt+=1
                if(debug):
                    print('Error Encountered. Trying again')
        if(attempt==5):
            print('Error Occurred Generating Recipes. Please contact group 1')
            break

    print()
    print("Please fill out the qualtrics survey for the group 1 recommender at this time: https://virginia.az1.qualtrics.com/jfe/form/SV_9oi2R7WEZb6tF9c")

#Run the recommender
recommender()
