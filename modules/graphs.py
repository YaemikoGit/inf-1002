
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px


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
    plt.legend(fontsize=12)

    plt.show()


# TO BE EDITTED - Graph should be output to popup window like other egs
# def workAffected(data):
#     mylabels = []
#     string_search = 'If yes, what percentage of your work time (time performing primary or secondary job functions) is affected by a mental health issue?'
#     num_arr = []
#     for x in range(len(data[string_search].value_counts())):
#         num_arr.append(data[string_search].value_counts()[x])
#         mylabels.append(data[string_search].value_counts().index[x])
#
#     data_arr = []
#
#     # Combine data extracted from mylabels and num_arr into a new array
#     for i in range(0, len(mylabels)):
#         data_arr.append([mylabels[i], num_arr[i]])
#
#     wrkProd_percentage_df = pd.DataFrame(data_arr, columns=['Percentage', 'Count'])
#
#     plt = px.pie(wrkProd_percentage_df, names='Percentage', values='Count', hole=0.5, color='Percentage',
#                  template='seaborn')
#     plt.update_layout(title_text='Work time affected by mental health issue', title_x=0.5)
#     plt.update_traces(textinfo='percent + value')
#     plt.update_layout(legend=dict(
#         orientation="h",
#         yanchor="bottom",
#         y=-0.1,
#         xanchor="center",
#         x=0.5
#     ))
#     plt.show()


#
def conditions(data):
    yes_condition = data["Have you been diagnosed with a mental health condition by a medical professional?"].value_counts()[1]

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

    fig, ax = plt.subplots()
    ax.pie(numbers[:-1], labels=conditions[:-1], autopct='%1.0f%%')
    plt.show()


# Examine Correlation between Employer taking mental health serious vs productivity
def heatmap(data):
    yAxis = "Do you feel that your employer takes mental health as seriously as physical health?"
    xAxis = "Do you believe your productivity is ever affected by a mental health issue?"

    cat_yAxis = data[yAxis].astype('category').cat.codes
    # d_cat_yAxis = dict(enumerate(cat_yAxis.cat.categories))

    cat_xAxis = data[xAxis].astype('category').cat.codes
    # d_cat_xAxis = dict(enumerate(cat_xAxis.cat.categories))

    # print(d_cat_yAxis)
    # print(d_cat_xAxis)

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
    """
    # plot the data
    plt.scatter(my_df[employer], my_df[prod])

    # fits the best fitting line to the data
    plt.plot(np.unique(my_df[employer]),
             np.poly1d(np.polyfit(my_df[employer], my_df[prod], 1))
             (np.unique(my_df[employer])), color='red')

    # Labelling axes
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    """

    new_df = my_df
    new_df.apply(lambda x: x.factorize()[0]).corr()
    sns.heatmap(pd.crosstab(new_df[employer], new_df[prod]))

    plt.show()




# function to plot bar chart
def plot_barchart(x_label, diagnosed_response, not_diagnosed_response, cat):
        if (cat == 'age group'):
            title_str = 'age group'
            x_text = 'Age group'
        elif (cat == 'gender'):
            title_str = 'gender'
            x_text = 'Gender'
        else:
            title_str = 'family background'
            x_text = 'Family Background'
        # Create an array of x positions for the bars
        x = np.arange(len(x_label))
        width = 0.4
        # The first bar for each age group for the participants who have been diagnosed with mental health disorder
        bar1 = plt.bar(x - width / 2, diagnosed_response, width, label='Yes', color='#7eb54e')
        # The second for each age group for the the participant who have not been diagnosed with mental health disorder
        bar2 = plt.bar(x + width / 2, not_diagnosed_response, width, label='No', color='#D21F3C')
        # Add labels and a title
        plt.xlabel(x_text)
        plt.ylabel('Number of responses')
        plt.title('Number of Participant with diagnosed mental health disorder of different %s' % title_str)
        # Add label
        plt.bar_label(bar1, labels=diagnosed_response, label_type='edge', fontsize=10, color='black')
        plt.bar_label(bar2, labels=not_diagnosed_response, label_type='edge', fontsize=10, color='black')
        # Set x-axis labels
        plt.xticks(x, x_label)

        # Add a legend
        plt.legend()

        # Show the plot
        plt.show()


# function to plot pie chart
def plot_piechart(x_label,diagnosed_response,not_diagnosed_response,cat):
  labels = ['Yes','No']
  colors=['#7eb54e','#D21F3C']
  #number of columns for the piechart
  num_columns = 2
  #number of rows for piechart
  num_rows =(len(x_label)+1)//2

  # subplots with the specified number of rows and columns
  fig, axs = plt.subplots(num_rows, num_columns, figsize=(12, 12))

  # Plotting one piechart for each of the age group specified
  for i in range(len(x_label)):
      row = i // num_columns
      col = i % num_columns
      #defining which row and column to display the chart
      ax = axs[row][col]
      ax.pie([diagnosed_response[i],not_diagnosed_response[i]], labels=labels, autopct='%1.1f%%', colors=colors,startangle=90)
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

      ax.set_title(title_text)

  #if number of graph is not even
  if(len(x_label)%2!=0):
    #removing the last empty plot
    fig.delaxes(axs[num_rows-1][1])

  plt.tight_layout()

  plt.show()



# What age groups are more prone to mental health issues?
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

    yes_responses = [age_group1_yes, age_group2_yes, age_group3_yes, age_group4_yes, age_group5_yes, age_group6_yes, age_group7_yes]
    no_responses = [age_group1_no, age_group2_no, age_group3_no, age_group4_no, age_group5_no, age_group6_no, age_group7_no]

    age_group_list = [age_group_1, age_group_2, age_group_3, age_group_4, age_group_5, age_group_6, age_group_7]


    #Select which graph to be output based on dropdown
    if type == 'Bar graph':
        plot_barchart(age_group_list, yes_responses, no_responses, 'age group')
    else:
        plot_piechart(age_group_list, yes_responses, no_responses, 'age group')



# Which gender are more prone to mental health issues?
def gender(data, type):

    data['What is your gender?'].unique()

    gender_df = data['What is your gender?']

    print('ok')

    is_diagnosed = data['Have you been diagnosed with a mental health condition by a medical professional?']
    # classfying responses according to gender
    # storing number of responses of different groups of diagnosed and not diagnosed participants of different gender
    diagnosed_male = data[(gender_df == "Male") & (is_diagnosed == 'Yes')].count()[0]
    not_diagnosed_male = data[(gender_df == 'Male') & (is_diagnosed == 'No')].count()[0]
    diagnosed_female = data[(gender_df == 'Female') & (is_diagnosed == 'Yes')].count()[0]
    not_diagnosed_female = data[(gender_df == 'Female') & (is_diagnosed == 'No')].count()[0]
    diagnosed_trans = data[(gender_df == "Transgender") & (is_diagnosed == 'Yes')].count()[0]
    not_diagnosed_trans = data[(gender_df == "Transgender") & (is_diagnosed == 'No')].count()[0]

    # the list is stored in a pattern of male, female, transgender, this follows df.unique of this particular df
    diagnosed_gender_group = [diagnosed_male, diagnosed_female, diagnosed_trans]
    not_diagnosed_gender_group = [not_diagnosed_male, not_diagnosed_female, not_diagnosed_trans]

    df_gender = ['Male', 'Female', 'Transgender']

    # Select which graph to be output based on dropdown
    if type == 'Bar graph':
        plot_barchart(df_gender,diagnosed_gender_group,not_diagnosed_gender_group,'gender')
    else:
        plot_piechart(df_gender,diagnosed_gender_group,not_diagnosed_gender_group,'gender')

