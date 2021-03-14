---
layout: post
title: Practical Data Analysis with Pandas
---


![_config.yml]({{ site.baseurl }}/images/config.png)

The easiest way to make your first post is to edit this one. Go into /_posts/ and update the Hello World markdown file. For more instructions head over to the [Jekyll Now repository](https://github.com/barryclark/jekyll-now) on GitHub.


In my last post, I mentioned working with data in Pandas library. One of Python’s most important libraries is pandas. With Pandas, you can clean the data and make it suitable for analysis and make artificial intelligence analysis such as modeling or machine learning. In this lesson, I will show you how to preprocess data to make it suitable for analysis and analyze it in a practical way.

Before starting the topic, our Medium page includes posts on data science, artificial intelligence, machine learning, and deep learning. Please don’t forget to follow us on Medium 🌱 to see these posts and the latest posts.
Let’s get started.
First of all, let’s import the panda with the pd abbreviation.

Let’s import the data set.

If you want, you can download this data set from here. Let’s take a look first rows of this data set.

The data set is about the vehicles stopped by the police in San Diago, California, on the road. This data set includes data such as the time to stop the vehicle, the age of the driver, the reason for the stop, and the status of the arrest.
Let’s print the last three rows on the screen.

Let’s want to see the structure of the data set.

Let’s see the data type of the columns in the data set.

The isnull() method is used to see the missing data in the data set. This command returns boolean, or logical value. False indicates that there is no missing data, and True indicates that there is missing data. Let’s see the total number of missing data in each row.

There are three important data structures in pandas. This tabular data set is in a DataFrame data structure. Each column of the DataFrame is in the Series data structure. Index objects are other data structures. Let’s want to reach a column in the DataFrame.

Let’s see the column indexes.

We can choose the columns we want in the data set. For example, let’s choose the column date.

Two square brackets are used to select more than one column.

We can change the names of the columns. The rename method is used for this. The inplace = True option modifies the data set.

We can select the row or column we want with loc or iloc. Selection is made by entering the row and column names in the loc command. In the iloc command, it is done by entering row and column indexes. The first parameter row of these commands refers to the second parameter column. For example, let’s choose the first row.

Let’s select the value of the first row first column.

We can choose more than one column.

We may want to slice the rows.

We can slice both rows and columns.

We enter the name in loc. Since the row indexes are rows here, we can enter numbers directly. Let’s enter the name for the column.

Let’s want to slice the columns.

The dropna() method is used to delete all columns with missing data. First, let’s see the structure of the data set again.

Let’s delete the values whose rows are missing data.

how = “any” option means to delete any missing data.

A Simple Analysis
Let’s examine whether men or women were stopped for more violations. Now let’s look at the reasons and numbers of stopped vehicles. Let’s take a reason for the stop column for this.

Thus, the reasons for stopping the vehicles were written on the screen. Vehicles were stopped due to the most speed violations. Let’s see the numbers of men and women stopped for speeding. For this, we filter according to the moving violation value. Then we select the subject_sex column and use the value_counts function to find the total number.

So we filtered according to the Moving Violation variable, then we chose the gender column. If we want to see the percentage of men and women by gender, the option normalized = True is used.

Let’s just want to see the reasons for stopping women as a percentage. First, we filter the gender column by men.

77 percent of the women were stopped due to speed violations. If we want to see the reasons for stopping men and women together, we must first divide the data set into groups according to gender. The groupby() method is used to group the data set by gender.

Let’s see the data in a tabular form.

Now let’s look at the effect of gender on the situation in arrest. Let’s find the number of arrests first.

Let’s see the percentage of the number of arrests.

Let’s see the percentage of arrests by gender.

Men were arrested twice as many as women. Let’s see the percentages of arrests by race and gender.

If you notice, black men and women were arrested more than others. Let’s investigate which year the number of stops was less. First, let’s find the first 4 years value of the date column and find the numbers of the years in the data set.

Let’s combine the date and time variables and bring them to the datetime structure to make historical operations more comfortable. First, let’s combine date and time variables.

Now let’s convert this date variable to datetime structure in pandas and add this variable to the data set.

Let’s look at the structure of the variables in the data set.

We can get what we want from the date and time of this variable. Let’s get to the months for example.

Let’s examine the change in the arrest situation during the day. First, let’s print the arrest_made data on the screen.

Let’s convert the object to boolean data structure.

Let’s see the number of arrests.

Let’s look at the percentage of arrests.

About 1 percent were arrested. Now let’s see the average of arrests during the day.

Let’s plot this data. To see the graph inline, it is necessary to use the % Matplotlib inline command.

Let’s plot the graph.

Let’s look at the work of the clocks in the data set.

Let’s sort the data.

Let’s plot the graph now.

We can say that there are more stopping at ten o’clock in the morning.
That’s it. I hope you enjoy this post. You can access the notebook I used for this post on our GitHub page. 🚩

