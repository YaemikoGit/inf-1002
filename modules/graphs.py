
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from math import exp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from dython.nominal import associations
from dython.nominal import identify_nominal_columns


#Extra functions to be used for plotting
#initialize an empty dictionary to store the percentage of diagnosed cases for different categories.
diagnosed_percentage={}
#function to calculate and append a dictionary of diagnosed percentage of population based on give categories
def percentage_of_diagnosed(x_label,cat,diagnosed_response,not_diagnosed_response):
  #clear previous dictionary of a different category
  diagnosed_percentage.clear()
  #initilize header text variable
  header_text=''
  #determine header text for given categories
  if cat=='age group':
    header_text='Age Group'
  elif cat=='gender':
    header_text='Gender'
  elif cat=='family background':
    header_text='Family History'
  elif cat=='location':
    header_text='Countries'
  elif cat=='personal history':
    header_text='Personal History'
  #print header for result
  print('%-25s%45s'%(header_text,"Diagnosed with Mental Health Disorder(%)"))
  print('='*70)
  #for each of the labels, populate the dictionary, and print it with its corresponding percentage of diagnosed participants,
  for i in range(len(x_label)):
    diagnosed_percentage[x_label[i]]=(diagnosed_response[i]/(diagnosed_response[i]+not_diagnosed_response[i])*100)
    print('%-25s%45.2f'%(x_label[i], diagnosed_percentage[x_label[i]]))
  #call the print_max() function with the popuplated dictionary and the given category
  print_max(diagnosed_percentage,cat)

#a function that takes in a dic1 and prints the max value in the list to show the group that has the highest number of population diagnosed with mental health
#dic1 is the dictionary that consists of the percentage of population being diagnosed with mental health disorder
#group is the category of metric (age group, gender, country, etc)
def print_max(dic1, cat):
  # find the key with key=dic1.get with the highest percentage
  max_item=max(dic1,key=dic1.get)
  #initialize text variables
  text=''
  cat_text=''

  #determine text to be displayed based on given category
  if cat=='age group':
    text="Age group '%s' is most susceptible towards mental health disorder with %2.2f%s of the participants diagnosed with mental health disorder."%(max_item,dic1[max_item],"%")
  elif cat=='gender':
    text="'%s' are most susceptible towards mental health disorder with %2.2f%s  of the participants diagnosed with mental health disorder"%(max_item,dic1[max_item],"%")
  elif cat=='family background':
    if max_item=='Yes':
      cat_text='Individuals with family history of mental health disorder'
    elif max_item=='No':
      cat_text='Individuals without family history of mental health disorder'
    elif max_item=='Unknown':
      cat_text='Individuals with unknown family history of mental health disorder'
    text='%s are most susceptible towards mental health disorder with %2.2f%s of the participants diagnosed with mental health disorder'%(cat_text,dic1[max_item],"%")
  elif cat=='location':
    text='Individuals in %s are most susceptible towards mental health disorder with %2.2f%s of the participants diagnosed with mental health disorder'%(max_item,dic1[max_item],"%")
  elif cat=='personal history':
    if max_item=='Yes':
      cat_text='Individuals with personal history of mental health disorder'
    elif max_item=='No':
      cat_text='Individuals without personal history of mental health disorder'
    elif max_item=='Maybe':
      cat_text='Individuals with an unsure personal history of mental health disorder'
    text='%s are most susceptible towards mental health disorder with %2.2f%s of the participants diagnosed with mental health disorder'%(cat_text,dic1[max_item],"%")


  #print the text as output
  #print(text)



# function to plot bar chart
#function to plot bar chart for the diagnosed and not diagnosed responses of the given category
def plot_barchart(x_label,diagnosed_response,not_diagnosed_response,cat):
  #determine the title text and the label text
  if (cat=='age group'):
    title_str='Age Group'
    x_text='Age group'
  elif (cat=='gender'):
    title_str='Gender'
    x_text='Gender'
  elif cat=='family background':
    title_str='Family History'
    x_text='Family History of Mental Health Illness'
  elif cat=='location':
    title_str='Countries'
    x_text='Countries'
  elif cat=='personal history':
    title_str='Personal History of Mental Health Disorder'
    x_text='Personal History of Mental Health Disorder'

  #create an array of x positions for the bars to be plotted
  x = np.arange(len(x_label))
  #detemine the width of each bar
  width=0.4
  # the first bar for participants who have been diagnosed with mental health disorder categorized based on the given category
  bar1=plt.bar(x - width/2, diagnosed_response, width, label='Yes',color='#7eb54e')
  # the second for participants who have not been diagnosed with mental health disorder categorized based on the given category
  bar2=plt.bar(x + width/2, not_diagnosed_response, width, label='No',color='#D21F3C')
  # add labels and title
  plt.xlabel(x_text)
  plt.ylabel('Number of Responses')
  plt.title('Number of Respondents with Diagnosed Mental Health Disorder \nof Different %s'%title_str)
  # add label to the bars
  plt.bar_label(bar1, labels=diagnosed_response, label_type='edge', fontsize=10, color='black')
  plt.bar_label(bar2, labels=not_diagnosed_response, label_type='edge', fontsize=10, color='black')
  # set x-axis labels for each bar
  plt.xticks(x, x_label)

  # add a legend
  plt.legend()
  plt.ion()
  # show the plot
  plt.ion()
  plt.show()


# function to plot pie chart
#function to plot multiple(same amount as the number of labels determined) pie charts for the diagnosed and not diagnosed responses of the given category
def plot_piechart(x_label,diagnosed_response,not_diagnosed_response,cat):
  #define label and its corresponding color of the pie chart
  labels = ['Yes','No']
  colors=['#7eb54e','#D21F3C']
  #number of columns for the piechart
  num_columns = 2
  #calculate the number of rows for piechart
  num_rows =(len(x_label)+1)//2

  # subplots with the specified number of rows and columns
  fig, axs = plt.subplots(num_rows, num_columns, figsize=(12, 12))
  if num_rows<2:
    #making axs a list such that it is iterable
    axs=[axs]
  # plotting one piechart for each of the group(labels) specified
  for i in range(len(x_label)):
      #determine which row and column to show the plotted graph
      row = i // num_columns
      col = i % num_columns
      ax = axs[row][col]
      #plotting the pie chart with the diagnosed and not diagnosed responses
      ax.pie([diagnosed_response[i],not_diagnosed_response[i]], labels=labels, autopct='%1.2f%%', colors=colors,startangle=90)
      #determine the title text of each pie chart depending on the given category
      if cat=='age group':
        title_text=('Percentage of Respondents of Age Group %s With\n Diagnosed Mental Health Disorder'%x_label[i])
      elif cat=='gender':
        title_text='Percentage of %s Respondents with Diagnosed \nMental Health disorder'%x_label[i]
      elif cat=='family background':
        if x_label[i]=='Yes':
          title_text= 'Percentage of Respondents with Family History that\n are Diagnosed with Mental Health Disorder'
        elif x_label[i]=='No':
          title_text= 'Percentage of Respondents without Family History that\n are Diagnosed with Mental Health Disorder'
        else:
          title_text= 'Percentage of Respondents with Unknown Family History that\n are Diagnosed with Mental Health Disorder'
      elif cat=='location':
          title_text='Percentage of Respondents in %s that are\n Diagnosed with Mental Health Disorder'%x_label[i]
      elif cat=='personal history':
        if x_label[i]=='Yes':
          title_text= 'Percentage of Respondents with Personal History of Mental\n Health Disorder that are Diagnosed with Mental Health Disorder'
        elif x_label[i]=='No':
          title_text= 'Percentage of Respondents without Personal History of Mental\n Health Disorder that are Diagnosed with Mental Health Disorder'
        else:
          title_text= 'Percentage of Respondents with an Unsure Personal History of\n Mental Health Disorder that are Diagnosed with Mental Health Disorder'
      #setting the title for each pie chart
      ax.set_title(title_text)

  #if number of graph is not even
  if(len(x_label)%2!=0):
    #removing the last empty plot
    fig.delaxes(axs[num_rows-1][1])

  plt.tight_layout()
  plt.ion()
  plt.show()




def remove_insignificant_labels(data,labels,dataframe,total):
    #for appending labels with sample size that are large enough for plotting graph
    new_label=list(labels[:])
    #for printing purposes
    removed_label=[]
    #for print formatting purposes
    removed_count=0
    for i in labels:
      #if responses are less than 10%, remove it from the new label list and add it to the removed_label list for printing purposes
      if data[dataframe==i].count()[0]/total <0.1:
        new_label.remove(i)
        removed_label.append(i)
        removed_count+=1

    #determine text to display based on the amount of labels removed
    # if removed_count==1:
    #   print("'"+removed_label[0]+"'",end='')
    #   print(' has been removed due to its small sample size')
    #
    # elif removed_count>1:
    #   for j in removed_label:
    #     print("'"+j+"'", end=', ')
    #   print(' have been removed due to their small sample size')


    return new_label






#graph plotting functions
# Bar graph on “How does mental health affect work performance?”
def workPer(data):


    # Respondents of those that feel mental health affects their productivity at work
    data['Do you believe your productivity is ever affected by a mental health issue?'].unique()


    ax = sns.countplot(x=data['Do you believe your productivity is ever affected by a mental health issue?'],
                       order=['Yes', 'Not applicable to me', 'No', 'Unsure'],
                       palette='Blues')
    ax.bar_label(ax.containers[0])
    plt.title('Mental health condition interfere with productivity?', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.ion()
    plt.show()


# Work time affeceted
def workAffected(data):
    mylabels = []
    string_search = 'If yes, what percentage of your work time (time performing primary or secondary job functions) is affected by a mental health issue?'
    num_arr = []
    for x in range(len(data[string_search].value_counts())):
        num_arr.append(data[string_search].value_counts()[x])
        mylabels.append(data[string_search].value_counts().index[x])

    data_arr = []

    # Combine data extracted from mylabels and num_arr into a new array
    for i in range(0, len(mylabels)):
        data_arr.append([mylabels[i], num_arr[i]])

    wrkProd_percentage_df = pd.DataFrame(data_arr, columns=['Percentage', 'Count'])

    # Creating autocpt arguments
    def func(pct, allvalues):
        absolute = int(pct / 100.*np.sum(allvalues))
        return "{:.1f}%\n{:d}".format(pct, absolute)

    #print(num_arr)
    fig, ax = plt.subplots(figsize =(10, 7))
    wedges, texts,autotexts = ax.pie(num_arr,
                                      labels = mylabels,
                                      autopct = lambda pct: func(pct, num_arr),
                                      shadow = False,
                                      startangle = 90,
                                      textprops = dict(color ="black"))

    # Legend
    ax.legend(wedges, mylabels,
              title ="Percentage of Time affected",
              loc ="center left",
              bbox_to_anchor =(1, 0, 0.5, 1))

    plt.setp(texts, size = 12, weight ="bold")
    ax.set_title("Work Time affected by mental health")

    plt.ion()

    # show plot
    plt.show()


# What are the common mental health combination in the dataset?
def conditions(data):
    # Amount of people diagnosed with mental health
    yes_condition = \
    data["Have you been diagnosed with a mental health condition by a medical professional?"].value_counts()[1]
    data["Have you been diagnosed with a mental health condition by a medical professional?"].value_counts()

    # Make a list containing the no. of such "conditions", and the type of "conditions"
    numbers = []
    conditions = []
    for x in range(len(data['If so, what condition(s) were you diagnosed with?'].value_counts())):
        if data['If so, what condition(s) were you diagnosed with?'].value_counts()[x] >= 10:
            numbers.append(data['If so, what condition(s) were you diagnosed with?'].value_counts()[x])
            conditions.append(data['If so, what condition(s) were you diagnosed with?'].value_counts().index[x])

    # Remove brackets (formatting)
    for x in range(len(conditions)):
        for condition_no in conditions[x].split("|"):
            if conditions[x].find("("):
                start = conditions[x].find("(")
                end = conditions[x].find(")")
                take_out = conditions[x][start - 1: end + 1]
                to_replace = conditions[x].replace(take_out, "")
                conditions[x] = to_replace

    others = yes_condition
    # For the "Other" values which were put aside
    for number in numbers:
        others -= number

    numbers.append(others)
    conditions.append("Others")

    colors = ['yellowgreen', 'red', 'gold', 'lightskyblue', 'lime', 'lightcoral', 'blue', 'pink', 'darkgreen', 'yellow',
              'grey', 'violet', 'magenta', 'cyan']
    #print(numbers)

    # Percentage of numbers
    percent = [(x / yes_condition) * 100 for x in numbers]

    # Formatting labels to show in Legend
    labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(conditions, percent)]

    patches, texts = plt.pie(numbers, colors=colors, startangle=90, radius=1.2)
    plt.legend(patches, labels, loc='center', bbox_to_anchor=(0.5, 0.5), fontsize=8)

    plt.ion()
    plt.show()



# Examine Correlation between Employer taking mental health serious vs productivity
def heatmap(data):
    yAxis = "Do you feel that your employer takes mental health as seriously as physical health?"
    xAxis = "Do you believe your productivity is ever affected by a mental health issue?"

    cat_yAxis = data[yAxis].astype('category').cat.codes
    # d_cat_yAxis = dict(enumerate(cat_yAxis.cat.categories))

    cat_xAxis = data[xAxis].astype('category').cat.codes
    # d_cat_xAxis = dict(enumerate(cat_xAxis.cat.categories))

    col_listy = cat_yAxis.values.tolist()
    col_listx = cat_xAxis.values.tolist()

    for i in range(0, len(col_listy)):
        if col_listy[i] == -1:
            col_listy[i] = 0

    for i in range(0, len(col_listx)):
        if col_listx[i] == -1:
            col_listx[i] = 1

    # print(col_listy)
    data = {'Employer taking mental health seriously': col_listy,
            'Productivity affected by mental health': col_listx
            }

    my_df = pd.DataFrame(data)

    corr_matrix = my_df.corr()

    employer = 'Employer taking mental health seriously'
    prod = 'Productivity affected by mental health'

    # adds the title
    plt.title('Examine Correlation between Employer taking mental health serious vs productivity')

    new_df = my_df
    new_df.apply(lambda x: x.factorize()[0]).corr()
    sns.heatmap(pd.crosstab(new_df[employer], new_df[prod]))
    plt.ion()
    plt.show()


# Overall correlation
def correlationHeat(data):
    # https://medium.com/@knoldus/how-to-find-correlation-value-of-categorical-variables-23de7e7a9e26
    # https://blog.knoldus.com/how-to-find-correlation-value-of-categorical-variables/

    categorical_features = identify_nominal_columns(data)
    # Make a copy to prevent changes to original
    columns_to_copy = ['What is your gender?',
                       'Have you been diagnosed with a mental health condition by a medical professional?',
                       'Do you have a family history of mental illness?', 'What is your age?',
                       'What country do you work in?', 'Have you had a mental health disorder in the past?',
                       ]
    blr = data[columns_to_copy].copy(deep=True)

    # Keeping countries UK and USA
    blr = blr.drop(blr[(blr['What country do you work in?'] != 'United Kingdom') & (
                blr['What country do you work in?'] != 'United States of America')].index)
    plt.ion()
    # Plotting the correlation heatmap using dython library
    associations(blr, figsize=(10, 10))


# What groups are more prone to mental health issues?
# a) Age
def ageGroup(data, type):

    # # define age group
    age_group_1 = '18-22'
    age_group_2 = '23-27'
    age_group_3 = '28-32'
    age_group_4 = '33-37'
    age_group_5 = '38-42'
    age_group_6 = '43-47'
    age_group_7 = '48-52'
    age_group_8 = '53-57'
    age_group_9 = '58-60'

    data['What is your age?'] = pd.cut(data['What is your age?'], bins=[18, 22, 27, 32, 37, 42, 47, 52, 57, 60], labels=[age_group_1, age_group_2, age_group_3, age_group_4, age_group_5, age_group_6,age_group_7,age_group_8,age_group_9])

    # variable to store age_group
    age_group = data['What is your age?']
    # variable to store response from participants on whether they had been diagnosed with a mental health disorder
    is_diagnosed = data['Have you been diagnosed with a mental health condition by a medical professional?']

    # seggreatating dataframes into age group and response to get number of response in each respective group
    age_group1_yes = data[(age_group == age_group_1) & (is_diagnosed == 'Yes')].count()[0]
    age_group1_no = data[(age_group == age_group_1) & (is_diagnosed == 'No')].count()[0]
    age_group2_yes = data[(age_group == age_group_2) & (is_diagnosed == 'Yes')].count()[0]
    age_group2_no = data[(age_group == age_group_2) & (is_diagnosed == 'No')].count()[0]
    age_group3_yes = data[(age_group == age_group_3) & (is_diagnosed == 'Yes')].count()[0]
    age_group3_no = data[(age_group == age_group_3) & (is_diagnosed == 'No')].count()[0]
    age_group4_yes = data[(age_group == age_group_4) & (is_diagnosed == 'Yes')].count()[0]
    age_group4_no = data[(age_group == age_group_4) & (is_diagnosed == 'No')].count()[0]
    age_group5_yes = data[(age_group == age_group_5) & (is_diagnosed == 'Yes')].count()[0]
    age_group5_no = data[(age_group == age_group_5) & (is_diagnosed == 'No')].count()[0]
    age_group6_yes = data[(age_group == age_group_6) & (is_diagnosed == 'Yes')].count()[0]
    age_group6_no = data[(age_group == age_group_6) & (is_diagnosed == 'No')].count()[0]
    age_group7_yes = data[(age_group == age_group_7) & (is_diagnosed == 'Yes')].count()[0]
    age_group7_no = data[(age_group == age_group_7) & (is_diagnosed == 'No')].count()[0]
    age_group8_yes = data[(age_group == age_group_8) & (is_diagnosed == 'Yes')].count()[0]
    age_group8_no = data[(age_group == age_group_8) & (is_diagnosed == 'No')].count()[0]
    age_group9_yes = data[(age_group == age_group_9) & (is_diagnosed == 'Yes')].count()[0]
    age_group9_no = data[(age_group == age_group_9) & (is_diagnosed == 'No')].count()[0]

    # seperating the defined groups above into diagnosed and not diagnosed list of responses
    diagnosed_age_group = [age_group1_yes, age_group2_yes, age_group3_yes, age_group4_yes, age_group5_yes,
                           age_group6_yes, age_group7_yes, age_group8_yes, age_group9_yes]
    not_diagnosed_age_group = [age_group1_no, age_group2_no, age_group3_no, age_group4_no, age_group5_no, age_group6_no,
                               age_group7_no, age_group8_no, age_group9_no]

    age_group_list = [age_group_1, age_group_2, age_group_3, age_group_4, age_group_5, age_group_6, age_group_7,
                       age_group_8, age_group_9]

    # #Select which graph to be output based on dropdown
    if type == 'Bar graph':
        plot_barchart(age_group_list, diagnosed_age_group, not_diagnosed_age_group, 'age group')
    else:
        plot_piechart(age_group_list, diagnosed_age_group, not_diagnosed_age_group, 'age group')



# b) gender
def gender(data, type):

    data['What is your gender?'].unique()

    gender_df = data['What is your gender?']

    is_diagnosed = data['Have you been diagnosed with a mental health condition by a medical professional?']
    # classfying responses according to gender
    # storing number of responses of different groups of diagnosed and not diagnosed participants of different gender
    diagnosed_male = data[(gender_df == "Male") & (is_diagnosed == 'Yes')].count()[0]
    not_diagnosed_male = data[(gender_df == 'Male') & (is_diagnosed == 'No')].count()[0]
    diagnosed_female = data[(gender_df == 'Female') & (is_diagnosed == 'Yes')].count()[0]
    not_diagnosed_female = data[(gender_df == 'Female') & (is_diagnosed == 'No')].count()[0]
    diagnosed_trans = data[(gender_df == "Other") & (is_diagnosed == 'Yes')].count()[0]
    not_diagnosed_trans = data[(gender_df == "Other") & (is_diagnosed == 'No')].count()[0]

    # the list is stored in a pattern of male, female, other, this follows df.unique of this particular df
    diagnosed_gender_group = [diagnosed_male, diagnosed_female, diagnosed_trans]
    not_diagnosed_gender_group = [not_diagnosed_male, not_diagnosed_female, not_diagnosed_trans]

    # labels for x axis
    gender_label = ['Male', 'Female', 'Other']


    # Select which graph to be output based on dropdown
    if type == 'Bar graph':
        plot_barchart(gender_label,diagnosed_gender_group,not_diagnosed_gender_group,'gender')
    else:
        plot_piechart(gender_label,diagnosed_gender_group,not_diagnosed_gender_group,'gender')



# c) Family history of mental illness
def famHistory(data, type):

    data['Do you have a family history of mental illness?'].unique()
    is_diagnosed = data['Have you been diagnosed with a mental health condition by a medical professional?']

    have_familyhistory = data['Do you have a family history of mental illness?']
    # storing number of responses of different groups of diagnosed and not diagnosed participants with different family background
    diagnosed_with_no_fh = data[(have_familyhistory == "No") & (is_diagnosed == 'Yes')].count()[0]
    not_diagnosed_with_no_fh = data[(have_familyhistory == 'No') & (is_diagnosed == 'No')].count()[0]
    diagnosed_with_fh = data[(have_familyhistory == 'Yes') & (is_diagnosed == 'Yes')].count()[0]
    not_diagnosed_with_fh = data[(have_familyhistory == 'Yes') & (is_diagnosed == 'No')].count()[0]
    diagnosed_with_unknown_fh = data[(have_familyhistory == "I don't know") & (is_diagnosed == 'Yes')].count()[0]
    not_diagnosed_with_unknown_fh = data[(have_familyhistory == "I don't know") & (is_diagnosed == 'No')].count()[0]

    # the list is stored in a pattern of no family history, with family history and unknown family history, this follows df.unique of this particular df
    diagnosed_group = [diagnosed_with_no_fh, diagnosed_with_fh, diagnosed_with_unknown_fh]
    not_diagnosed_group = [not_diagnosed_with_no_fh, not_diagnosed_with_fh, not_diagnosed_with_unknown_fh]

    # storing labels for graph plotting and analysis purposes
    family_history_label = data['Do you have a family history of mental illness?'].unique()

    if type == 'Bar graph':
        plot_barchart(family_history_label, diagnosed_group, not_diagnosed_group, 'family background')
    else:
        plot_piechart(family_history_label, diagnosed_group, not_diagnosed_group, 'family background')


# d) location
def location(data, type):

    # storing and checking of unique values in the column
    location_labels = data['What country do you work in?'].unique()
    is_diagnosed = data['Have you been diagnosed with a mental health condition by a medical professional?']

    location_df = data["What country do you work in?"]

    # remove groups with small sample size
    location_label = remove_insignificant_labels(data, location_labels, location_df, len(data))

    # classify responses into groups based on location and whether they have been diagnosed with mental health disorder
    diagnosed_uk = data[(location_df == location_label[0]) & (is_diagnosed == 'Yes')].count()[0]
    not_diagnosed_uk = data[(location_df == location_label[0]) & (is_diagnosed == 'No')].count()[0]
    diagnosed_us = data[(location_df == location_label[1]) & (is_diagnosed == 'Yes')].count()[0]
    not_diagnosed_us = data[(location_df == location_label[1]) & (is_diagnosed == 'No')].count()[0]

    # stored in a patttern that starts with uk, followed by us. This pattern follows the location label
    diagnosed_location = [diagnosed_uk, diagnosed_us]
    not_diagnosed_location = [not_diagnosed_uk, not_diagnosed_us]

    if type == 'Bar graph':
        plot_barchart(location_label,diagnosed_location,not_diagnosed_location,'location')
    else:
        plot_piechart(location_label,diagnosed_location,not_diagnosed_location,'location')



# e) Individual
def indivi(data, type):
    data['Have you had a mental health disorder in the past?'].unique()
    is_diagnosed = data['Have you been diagnosed with a mental health condition by a medical professional?']

    have_mental_health_history = data['Have you had a mental health disorder in the past?']
    # classifying responses based on personal history of mental illness
    # storing number of responses of different groups of diagnosed and not diagnosed participants with different personal history
    diagnosed_with_no_mhh = data[(have_mental_health_history == "No") & (is_diagnosed == 'Yes')].count()[0]
    not_diagnosed_with_no_mhh = data[(have_mental_health_history == 'No') & (is_diagnosed == 'No')].count()[0]
    diagnosed_with_mhh = data[(have_mental_health_history == 'Yes') & (is_diagnosed == 'Yes')].count()[0]
    not_diagnosed_with_mhh = data[(have_mental_health_history == 'Yes') & (is_diagnosed == 'No')].count()[0]
    diagnosed_with_unknown_mhh = data[(have_mental_health_history == "Maybe") & (is_diagnosed == 'Yes')].count()[0]
    not_diagnosed_with_unknown_mhh = data[(have_mental_health_history == "Maybe") & (is_diagnosed == 'No')].count()[0]

    # the list is stored in a pattern of with mental health disorder history, with mental unknown health disorder history, and with no mental health disorder historythis follows df.unique of this particular df
    diagnosed_group = [diagnosed_with_mhh, diagnosed_with_unknown_mhh, diagnosed_with_no_mhh]
    not_diagnosed_group = [not_diagnosed_with_mhh, not_diagnosed_with_unknown_mhh, not_diagnosed_with_no_mhh]

    # storing labels for graph plotting and analysis purposes
    mental_health_history_label = data['Have you had a mental health disorder in the past?'].unique()

    if type == 'Bar graph':
        # plotting bar chart for different groups of personal history of mental health disorder
        plot_barchart(mental_health_history_label, diagnosed_group, not_diagnosed_group, 'personal history')
    else:
        # plot pie chart for respective groups of different perosnal history of mental health disorder
        plot_piechart(mental_health_history_label, diagnosed_group, not_diagnosed_group, 'personal history')









# Are current efforts effective and enough?
def effort(data, type):

    if type == 'Mental health benefits in healthcare coverage':

        # plot number of people who receives/do not receive mental health benefits
        data["Does your employer provide mental health benefits as part of healthcare coverage?"].replace(
            'Not eligible for coverage / N/A', 'No', inplace=True)
        effort1 = sns.countplot(data, x="Does your employer provide mental health benefits as part of healthcare coverage?", palette='Blues')
        effort1.bar_label(effort1.containers[0], fontsize=10)
        effort1.set(title='Provides Mental Health Benefits in Healthcare Coverage', xlabel='')
        plt.ion()
        plt.show()

    elif type == 'Discussed or Conducted mental health events':

        # plot number of people whose companies host mental health events
        effort2 = sns.countplot(data, x="Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?")
        effort2.bar_label(effort2.containers[0], fontsize=10)
        effort2.set(title='Discussed or Conducted Mental Health Events', xlabel='')
        plt.ion()
        plt.show()

    else:
        # plot number of people whose companies provide extra mental health resources
        effort3 = sns.countplot(data, x="Does your employer offer resources to learn more about mental health concerns and options for seeking help?", palette='Blues')
        effort3.bar_label(effort3.containers[0], fontsize=10)
        effort3.set(title='Provides Extra Mental Health Resources', xlabel='')
        plt.ion()
        plt.show()




def conseq(data, type):

    if type == 'Employees with No Mental Health Support':
        # filter columns whereby employees do not receive any mental health support (mental health coverage benefits, campaigns, extra resources)
        filter_nosupport = ((data['Does your employer provide mental health benefits as part of healthcare coverage?'] == 'No') &
                            (data['Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?'] == 'No') &
                            (data['Does your employer offer resources to learn more about mental health concerns and options for seeking help?'] == 'No'))

        # Use the boolean mask to filter the DataFrame
        df_nosupport = pd.DataFrame(data=data[filter_nosupport])

        responses = []
        countlist = []

        negconsequence = df_nosupport['Do you think that discussing a mental health disorder with your employer would have negative consequences?']

        for i in range(len(negconsequence.value_counts())):
            countlist.append(negconsequence.value_counts()[i])
            responses.append(negconsequence.value_counts().index[i])
        # rename responses labels
        responses[0] = 'May have Negative Consequences'
        responses[1] = 'No Negative Consequences'
        responses[2] = 'Have Negative Consequences'

        # plot pie chart
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])

        ax.axis('equal')
        ax.pie(countlist, autopct='%1.2f%%', colors=['#c2be4a', '#4ac250', '#c2564a'])
        # plt.legend()
        ax.legend(countlist, labels=responses, bbox_to_anchor=(0.5, 0.5), loc='center right', fontsize=8)

        # pie chart title
        fig.suptitle('Employees with All Mental Health Support: \n Consequence upon discussing Mental Health Disorders with Employers')

        plt.ion()
        plt.show()

    else:
        # filter columns whereby employees receive all mental health support (mental health coverage benefits, campaigns, extra resources)
        filter_allsupport = ((data['Does your employer provide mental health benefits as part of healthcare coverage?'] == 'Yes') &
                             (data[ 'Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?'] == 'Yes') &
                             (data['Does your employer offer resources to learn more about mental health concerns and options for seeking help?'] == 'Yes'))

        # Use the boolean mask to filter the DataFrame
        df_allsupport = pd.DataFrame(data=data[filter_allsupport])
        allsupport_responses = []
        allsupport_countlist = []

        allnegconsequence = df_allsupport[
            'Do you think that discussing a mental health disorder with your employer would have negative consequences?']

        for i in range(len(allnegconsequence.value_counts())):
            allsupport_countlist.append(allnegconsequence.value_counts()[i])
            allsupport_responses.append(allnegconsequence.value_counts().index[i])

        # rename responses labels
        allsupport_responses[0] = 'No Negative Consequences'
        allsupport_responses[1] = 'May have Negative Consequences'
        allsupport_responses[2] = 'Have Negative Consequences'

        # plot pie chart
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('equal')

        ax.pie(allsupport_countlist, autopct='%1.2f%%', colors=['#c2be4a', '#4ac250', '#c2564a'])
        # plt.legend()
        ax.legend(allsupport_countlist, labels=allsupport_responses, bbox_to_anchor=(0.5, 0.5), loc='center right', fontsize=8)

        # displaying the title
        fig.suptitle("Employees with All Mental Health Support: \n Consequence upon discussing Mental Health Disorders with Employers")

        plt.ion()
        plt.show()




def likeInf(data):
    # filter those who have experienced
    pd.set_option('display.max_columns', None)
    filter_yesexperienced = (data['Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?'] == 'Yes, I experienced')
    df_yesexperienced = pd.DataFrame(data=data[filter_yesexperienced])

    # filter those who have observed
    filter_yesobserved = (data['Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?'] == 'Yes, I observed')
    df_yesobserved = pd.DataFrame(data=data[filter_yesobserved])

    # get count of responses for those who have experienced
    yesexperienced_responses = []
    yesexperienced_countlist = []

    influence_yesexperienced = df_yesexperienced['Have your observations of how another individual who discussed a mental health disorder made you less likely to reveal a mental health issue yourself in your current workplace?']

    for i in range(len(influence_yesexperienced.value_counts())):
        yesexperienced_countlist.append(influence_yesexperienced.value_counts()[i])
        yesexperienced_responses.append(influence_yesexperienced.value_counts().index[i])

    # get count of responses for those who have observed
    yesobserved_responses = []
    yesobserved_countlist = []

    influence_yesobserved = df_yesobserved[
        'Have your observations of how another individual who discussed a mental health disorder made you less likely to reveal a mental health issue yourself in your current workplace?']

    for i in range(len(influence_yesobserved.value_counts())):
        yesobserved_countlist.append(influence_yesobserved.value_counts()[i])
        yesobserved_responses.append(influence_yesobserved.value_counts().index[i])

    # filter those who might have experienced or observed
    filter_maybeexperienced = (data['Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?'] == 'Maybe/Not sure')
    df_maybeexperienced = pd.DataFrame(data=data[filter_maybeexperienced])


    # get count of responses for those who might have experienced or observed
    maybeexperienced_responses = []
    maybeexperienced_countlist = []

    influence_maybeexperienced = df_maybeexperienced['Have your observations of how another individual who discussed a mental health disorder made you less likely to reveal a mental health issue yourself in your current workplace?']

    for i in range(len(influence_maybeexperienced.value_counts())):
        maybeexperienced_countlist.append(influence_maybeexperienced.value_counts()[i])
        maybeexperienced_responses.append(influence_maybeexperienced.value_counts().index[i])

    # new dataframe for barchart

    # responses rename
    # Unlikely to Discuss (response:Yes)
    # Likely to Discuss (response:No)
    # Might Discuss (response:Maybe)
    influence_df = pd.DataFrame({'Have Observed': [125, 61, 55],
                                 'Have Experienced': [63, 32, 41],
                                 'Might have \nObserved or Experienced': [52, 80, 136]},
                                index=['Unlikely to Discuss', 'Might Discuss', 'Likely to Discuss'])

    # stacked bar chart: influence of actions based on experiences
    influ = influence_df.plot(kind='bar', stacked=True, color=['pink', 'yellow', 'skyblue'])

    # labels for x & y axis
    plt.xlabel('\n Likeliness to Discuss')
    plt.ylabel('Number of Employees')

    plt.xticks(rotation=0)

    influ.bar_label(influ.containers[2], fontsize=10);
    for c in influ.containers:
        # Optional: if the segment is small or 0, customize the labels
        labels = [v.get_height() if v.get_height() > 0 else '' for v in c]
        # remove the labels parameter if it's not needed for customized labels
        influ.bar_label(c, labels=labels, label_type='center')

    # # title of stacked bar chart
    plt.title('Likeliness to Discuss Mental Health, \nInfluenced by Observations/Experiences')
    plt.ion()
    plt.show()





# Prediction of whether certain factors determine mental illness
def binaryLog(data):
    # formatting the categorical values:
    def format(blr, text, newList, oldList):
        for no_list in range(len(new_familyHistory)):
            blr.loc[0:, text].replace(oldList[no_list], newList[no_list], inplace=True)

    # Change yes no to 1 0
    data.loc[0:, "Have you been diagnosed with a mental health condition by a medical professional?"].replace("Yes", 1,
                                                                                                             inplace=True)
    data.loc[0:, "Have you been diagnosed with a mental health condition by a medical professional?"].replace("No", 0,
                                                                                                             inplace=True)

    # Link to reason why must drop 1 dummy
    # https://medium.com/nerd-for-tech/what-is-dummy-variable-trap-how-it-can-be-handled-using-python-78ec17246331#:~:text=So%20if%20there%20are%20n,variable%20has%20to%20be%20dropped.
    # Dummy variable trap explained
    # https://www.algosome.com/articles/dummy-variable-trap-regression.html#:~:text=The%20solution%20to%20the%20dummy,remaining%20categories%20represent%20the%20change

    # genders
    dummy_genders = pd.get_dummies(data["What is your gender?"]).drop(["Other"], axis=1)

    # family history
    new_familyHistory = ["Idk_familyHistory", "No_familyHistory", "Yes_familyHistory"]
    old_familyHistory = ["I don't know", "No", "Yes"]
    format(data, "Do you have a family history of mental illness?", new_familyHistory, old_familyHistory)
    dummy_familyHistory = pd.get_dummies(data["Do you have a family history of mental illness?"]).drop(
        [new_familyHistory[0]], axis=1)

    # past mental health
    new_past = ["Maybe_Past", "No_Past", "Yes_Past"]
    old_past = ["Maybe", "No", "Yes"]
    format(data, "Have you had a mental health disorder in the past?", new_past, old_past)
    dummy_past = pd.get_dummies(data["Have you had a mental health disorder in the past?"]).drop([new_past[0]], axis=1)

    # countries
    dummy_countres = pd.get_dummies(data["What country do you work in?"]).drop("United Kingdom", axis=1)

    # Input x variables, y variable
    X = pd.concat([dummy_familyHistory, dummy_genders, dummy_past, dummy_countres], axis=1)
    # X = pd.concat([dummy_familyHistory , dummy_genders, dummy_countres], axis=1)
    y = data['Have you been diagnosed with a mental health condition by a medical professional?']

    # Divide the data to training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1234)

    # https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
    lr_model = LogisticRegression()

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    lr_model.fit(X_train, y_train)

    y_pred_sk = lr_model.predict(X_test)

    # Shows the accuracy by showing True positive/negativein respect of false positive/negative
    confusionMatrix = metrics.confusion_matrix(y_test, y_pred_sk)
    plt.figure(figsize=(5, 5))
    sns.heatmap(confusionMatrix, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r');
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(lr_model.score(X_test, y_test))
    plt.title(all_sample_title, size=10)
    plt.title('Prediction of whether certain factors determine mental illness')
    plt.ion()
    plt.show()





# classified them tgt (ISSUE WITH SPLIT MODEL)
def classfied(data):

    # Make a copy to prevent changes to original
    columns_to_copy = ['What is your gender?',
                       'Have you been diagnosed with a mental health condition by a medical professional?',
                       'Do you have a family history of mental illness?', 'What is your age?',
                       'What country do you work in?', 'Have you had a mental health disorder in the past?',
                       'What is your age?']
    blr = data[columns_to_copy].copy(deep=True)

    # Keeping countries UK and USA
    blr = blr.drop(blr[(blr['What country do you work in?'] != 'United Kingdom') & (blr['What country do you work in?'] != 'United States of America')].index)

    #print(blr['What country do you work in?'])

    # formatting the categorical values:
    def format(blr, text, newList, oldList):
        for no_list in range(len(new_familyHistory)):
            blr.loc[0:, text].replace(oldList[no_list], newList[no_list], inplace=True)

    # Change yes no to 1 0
    blr.loc[0:, "Have you been diagnosed with a mental health condition by a medical professional?"].replace("Yes", 1, inplace=True)
    blr.loc[0:, "Have you been diagnosed with a mental health condition by a medical professional?"].replace("No", 0, inplace= True)

    # genders
    dummy_genders = pd.get_dummies(blr["What is your gender?"]).drop(["Other"], axis=1)

    # family history
    new_familyHistory = ["Idk_familyHistory", "No_familyHistory", "Yes_familyHistory"]
    old_familyHistory = ["I don't know", "No", "Yes"]
    format(data, "Do you have a family history of mental illness?", new_familyHistory, old_familyHistory)
    dummy_familyHistory = pd.get_dummies(data["Do you have a family history of mental illness?"]).drop([new_familyHistory[0]], axis=1)

    # past mental health
    new_past = ["Maybe_Past", "No_Past", "Yes_Past"]
    old_past = ["Maybe", "No", "Yes"]
    format(blr, "Have you had a mental health disorder in the past?", new_past, old_past)
    dummy_past = pd.get_dummies(blr["Have you had a mental health disorder in the past?"]).drop([new_past[0]], axis=1)

    # countries
    dummy_country = pd.get_dummies(blr["What country do you work in?"]).drop("United Kingdom",  axis=1)


    # Input x variables, y variable x now has 1034, y has 1433
    X = pd.concat([dummy_familyHistory, dummy_genders, dummy_past, dummy_country,blr['Have you been diagnosed with a mental health condition by a medical professional?']], axis=1)
    X=X.dropna()
    y = X['Have you been diagnosed with a mental health condition by a medical professional?']
    X= X.drop("Have you been diagnosed with a mental health condition by a medical professional?",axis=1)




    # Divide the data to training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1234)

    lr_model = LogisticRegression()

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    lr_model.fit(X_train, y_train)
    y_pred_sk = lr_model.predict(X_test)

    #print(classification_report(y_test, y_pred_sk))

    importance = lr_model.coef_.flatten()
    #print(importance)
    # Shows the impact of each coefficient
    plt.rcParams["figure.figsize"] = (10, 10)
    plt.barh(X.columns, importance)
    plt.title("Feature Importance")
    plt.xlabel("score")
    plt.ion()
    plt.show()