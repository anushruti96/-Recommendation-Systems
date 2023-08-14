import csv

#This file is to get the list of recipes and their id
with open('./epicurious.csv', 'r') as read_obj:
    epicurious = list(csv.reader(read_obj))

epicurious.pop(0)

#Loop through all recipes
index = 0
recipes = []
for row in epicurious:
    recipes.append([row[0], index])
    index+=1

#Write the recipelist
with open('recipesAndIds.csv', 'w') as write_obj:
    writer = csv.writer(write_obj)
    writer.writerows(recipes)
