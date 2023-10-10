
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from math import exp
from sklearn.linear_model import LogisticRegression


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
  if cat == 'age group':
    header_text = 'Age Group'
  elif cat == 'gender':
    header_text = 'Gender'
  elif cat == 'family background':
    header_text = 'Family History'
  elif cat == 'location':
    header_text = 'Countries'

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
    text="Age group %s is most susceptible towards mental health disorder with %2.2f of the participants diagnosed with mental health disorder."%(max_item,dic1[max_item])
  elif cat=='gender':
    text='%s are most susceptible towards mental health disorder with %2.2f of the participants diagnosed with mental health disorder'%(max_item,dic1[max_item])
  elif cat=='family background':
    if max_item=='Yes':
      cat_text='Individuals with family history of mental health disorder'
    elif max_item=='No':
      cat_text='Individuals without family history of mental health disorder'
    elif max_item=='Unknown':
      cat_text='Individuals with unknown family history of mental health disorder'
    text='%s are most susceptible towards mental health disorder with %2.2f of the participants diagnosed with mental health disorder'%(cat_text,dic1[max_item])
  elif cat=='location':
    text='Individuals in %s are most susceptible towards mental health disorder with %2.2f of the participants diagnosed with mental health disorder'%(max_item,dic1[max_item])

  #print the text as output
  print(text)


# function to plot bar chart
#function to plot bar chart for the diagnosed and not diagnosed responses of the given category
def plot_barchart(x_label,diagnosed_response,not_diagnosed_response,cat):
  #determine the title text and the label text
  if (cat == 'age group'):
    title_str = 'age group'
    x_text = 'Age group'
  elif (cat == 'gender'):
    title_str = 'gender'
    x_text = 'Gender'
  elif cat == 'family background':
    title_str = 'family background'
    x_text = 'Family Background'
  elif cat == 'location':
    title_str = 'countries'
    x_text = 'Countries'

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
  plt.ylabel('Number of responses')
  plt.title('Number of Participant with diagnosed mental health disorder of different %s'%title_str)
  # add label to the bars
  plt.bar_label(bar1, labels=diagnosed_response, label_type='edge', fontsize=10, color='black')
  plt.bar_label(bar2, labels=not_diagnosed_response, label_type='edge', fontsize=10, color='black')
  # set x-axis labels for each bar
  plt.xticks(x, x_label)

  # add a legend
  plt.legend()

  # show the plot
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
      ax.pie([diagnosed_response[i],not_diagnosed_response[i]], labels=labels, autopct='%1.1f%%', colors=colors,startangle=90)
      #determine the title text of each pie chart depending on the given category
      if cat=='age group':
        title_text=('Number of Participant of age %s with diagnosed \nmental health disorder'%x_label[i])
      elif cat=='gender':
        title_text='Number of %s participant with diagnosed \n mental health disorder'%x_label[i]
      elif cat=='family background':
        if x_label[i]=='Yes':
          title_text= 'Number of participant with family history that are diagnosed \n with mental health disorder'
        elif x_label[i]=='No':
          title_text= 'Number of participant without family history that are diagnosed \n with mental health disorder'
        else:
          title_text= 'Number of participant with unknown family history that are diagnosed \n with mental health disorder'
      elif cat=='location':
          title_text='Number of participant in %s that are diagnosed \n with mental health disorder'%x_label[i]
      #setting the title for each pie chart
      ax.set_title(title_text)

  #if number of graph is not even
  if(len(x_label)%2!=0):
    #removing the last empty plot
    fig.delaxes(axs[num_rows-1][1])

  plt.tight_layout()

  plt.show()



#remove label if responses of that particular label is less than 10% of the total
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
    if removed_count==1:
      print("'"+removed_label[0]+"'",end='')
      print(' has been removed due to its small sample size')

    elif removed_count>1:
      for j in removed_label:
        print("'"+j+"'", end=', ')
      print(' have been removed due to their small sample size')

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
    plt.legend(patches, labels, loc='center right', bbox_to_anchor=(0.1, 1.), fontsize=8)
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

    plt.show()




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
    age_group_7 = '47+'

    data['What is your age?'] = pd.cut(data['What is your age?'], bins=[18, 22, 27, 32, 37, 42, 47, 50],
                             labels=[age_group_1, age_group_2, age_group_3, age_group_4, age_group_5, age_group_6,
                                     age_group_7])

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

    # seperating the defined groups above into diagnosed and not diagnosed list of responses
    diagnosed_age_group = [age_group1_yes, age_group2_yes, age_group3_yes, age_group4_yes, age_group5_yes,
                           age_group6_yes, age_group7_yes]
    not_diagnosed_age_group = [age_group1_no, age_group2_no, age_group3_no, age_group4_no, age_group5_no, age_group6_no,
                               age_group7_no]

    age_group_list = [age_group_1, age_group_2, age_group_3, age_group_4, age_group_5, age_group_6, age_group_7]


    # #Select which graph to be output based on dropdown
    if type == 'Bar graph':
        plot_barchart(age_group_list, diagnosed_age_group, not_diagnosed_age_group, 'age group')
    else:
        plot_piechart(age_group_list, diagnosed_age_group, not_diagnosed_age_group, 'age group')
    # if type == 'Bar graph':
    #     plot_barchart(age_group_list, diagnosed_age_group, not_diagnosed_age_group, 'age group')
    # elif type == 'Pie chart':
    #     plot_piechart(age_group_list, diagnosed_age_group, not_diagnosed_age_group, 'age group')
    # else:
    #     # call percentage_of_diagnosed() function to display percentage of diagnosed participants according to their age group
    #     percentage_of_diagnosed(age_group_list, 'age group', diagnosed_age_group, not_diagnosed_age_group)



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

    # the list is stored in a pattern of male, female, transgender, this follows df.unique of this particular df
    diagnosed_gender_group = [diagnosed_male, diagnosed_female, diagnosed_trans]
    not_diagnosed_gender_group = [not_diagnosed_male, not_diagnosed_female, not_diagnosed_trans]

    # labels for x axis
    gender_label = ['Male', 'Female', 'Other']


    # Select which graph to be output based on dropdown
    if type == 'Bar graph':
        plot_barchart(gender_label,diagnosed_gender_group,not_diagnosed_gender_group,'gender')
    else:

        # recall the function to remove groups with sample sizes that are too small
        gender_label = remove_insignificant_labels(data, gender_label, gender_df, len(gender_df))

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

    plot_barchart(location_label, diagnosed_location, not_diagnosed_location, 'location')

    if type == 'Bar graph':
        plot_barchart(location_label,diagnosed_location,not_diagnosed_location,'location')
    else:
        plot_piechart(location_label,diagnosed_location,not_diagnosed_location,'location')





# Are current efforts effective and enough?
def effort(data, type):

    if type == 'Mental health benefits in healthcare coverage':

        data["Does your employer provide mental health benefits as part of healthcare coverage?"].replace(
            'Not eligible for coverage / N/A', 'No', inplace=True)
        effort1 = sns.countplot(data,x="Does your employer provide mental health benefits as part of healthcare coverage?", palette='Blues')
        effort1.set(title='Provides Mental Health Benefits in Healthcare Coverage', xlabel='')
        plt.show()

    elif type == 'Discussed or Conducted mental health events':

        effort2 = sns.countplot(data,x="Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?",
                                order=['No', 'Yes', 'I don\'t know'], palette='Blues')
        effort2.set(title='Discussed or Conducted Mental Health Events', xlabel='')
        plt.show()

    else:

        effort3 = sns.countplot(data,x="Does your employer offer resources to learn more about mental health concerns and options for seeking help?",
                                palette='Blues')
        effort3.set(title='Provides Extra Mental Health Resources', xlabel='')
        plt.show()




# Prediction of whether certain factors determine mental illness
def binaryLog(data):
    # Make a copy to prevent changes to original
    blr = data.copy(deep=True)

    # Change yes no to 1 0
    blr.loc[0:, "Have you been diagnosed with a mental health condition by a medical professional?"].replace("Yes", 1,
                                                                                                             inplace=True)
    blr.loc[0:, "Have you been diagnosed with a mental health condition by a medical professional?"].replace("No", 0,
                                                                                                             inplace=True)

    # Link to reason why must drop 1 dummy
    # Dummy variable trap explained
    # https://www.algosome.com/articles/dummy-variable-trap-regression.html#:~:text=The%20solution%20to%20the%20dummy,remaining%20categories%20represent%20the%20change

    # 3 genders, Male Female Transgender --> 0 and 1s
    dummy_genders = pd.get_dummies(blr["What is your gender?"]).drop(["Female"], axis=1)

    # 3 options, Yes No "I Dont know" --> 0 and 1's
    dummy_familyHistory = pd.get_dummies(blr["Do you have a family history of mental illness?"]).drop(["I don't know"],
                                                                                                      axis=1)

    dummy_past = pd.get_dummies(blr["Have you had a mental health disorder in the past?"]).drop(["Maybe"], axis=1)

    # Input x variables, y variable
    X = pd.concat([blr[["What is your age?"]], dummy_familyHistory, dummy_genders, dummy_past], axis=1)
    y = blr['Have you been diagnosed with a mental health condition by a medical professional?']

    # Divide the data to training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # Create an instance and fit the model
    lr_model = LogisticRegression()

    y_train = y_train.astype('int')
    lr_model.fit(X_train, y_train)

    # Making predictions
    y_pred_sk = lr_model.predict(X_test)

    # Accuracy
    y_test = y_test.astype('int')

    # Shows the accuracy by showing True positive/negativein respect of false positive/negative
    confusionMatrix = metrics.confusion_matrix(y_test, y_pred_sk)
    plt.figure(figsize=(5, 5))
    sns.heatmap(confusionMatrix, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(lr_model.score(X_test, y_test))
    plt.title(all_sample_title, size=10)

    plt.show()
