import streamlit as st
from streamlit_elements import media
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns
from pylab import *
import plotly.express as px
import plotly.graph_objects as go                               

#from pyecharts.charts import Pie
#import streamlit_echarts as ste

plt.style.use('seaborn')
df = pd.read_csv('AppleStore.csv',index_col=0)

st.set_page_config(page_title="Mobile App Analysis", layout="wide")



# 设置布局
sns.set_style("darkgrid")
row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (0.1, 2, 0.2, 1, 0.1)
)

row0_1.title("Static analysis of mobile apps ")
with row0_2:
    st.write("")

row0_2.subheader(
    "A Streamlit web app by Team 42 - Xinyu Hu and Linjie Zhou"
)


row1_spacer1, row1_1, row1_spacer2 = st.columns((0.1, 3.2, 0.1))

with row1_1:
    st.markdown(
        "Hey there! Welcome to our ios app analysis. This app scrapes from [RAMANATHAN](https://www.kaggle.com/datasets/ramamet4/app-store-apple-data-set-10k-apps). The background of analysis this dataset is as follows: The ever-changing mobile landscape is a challenging space to navigate. The percentage of mobile over desktop is only increasing.  Mobile app analytics is a great way to understand the existing strategy to drive growth and retention of future user."
        )



line1_spacer1, line1_1, line1_spacer2 = st.columns((0.1, 3.2, 0.1))
with line1_1:
    st.header("Exploratory Data Analysis")
    # add a slider
    # A VIEW OF filtered raw data

    price_filter = st.sidebar.slider('Mobile App Price', 0 , 5, 2)
    m = df[df.price <= price_filter]

    st.write('Before we start analysis, lets have a quick look at the data filtered by price :blush:')
    st.write(m)

    df.isnull().sum()
    df.groupby('prime_genre').sum()
    genres = df['prime_genre'].unique()

form = st.sidebar.form("country_form")
app_filter = form.text_input('App Name (enter ALL to reset)', 'ALL')
form.form_submit_button("Apply")
# filter by name
if app_filter!='ALL':
    a = df[df.track_name == app_filter]
    st.write("")
    st.subheader('The detailed information of the searched app:')
    st.write("")
    st.write(a)

line2_spacer1, line2_1, line2_spacer2 = st.columns((0.1, 3.2, 0.1))
with line2_1:
    st.header('Part 1: Price effects of Apps')
    st.subheader('Distribution of free and paid apps')

    # show the pie chart
    fig, ax = plt.subplots(figsize = (5,5))
    freeapps = df[df.price == 0.0]
    paidapps = df[df.price != 0.0]
    data = np.array([56.4,43.6])
    pie_labels = np.array(['Free','Paid'])
    # 绘制饼图
    plt.pie(data,radius=0.6,labels=pie_labels,autopct='%3.1f%%')
    st.pyplot(fig)
    st.markdown('Conclusion')
    st.markdown('Free apps accounts for 54.6%, but paid apps accounts for 45.4%. They are almost the same.')



    # Return the numbers of free app in each genres
    def genreFree(gen):
        return len(df[(df['price'] == 0.0) & (df['prime_genre']== gen)])
    # Return the numbers of paid app in each genres
    def genrePaid(gen):
        return len(df[(df['price'] != 0.0) & (df['prime_genre']== gen)])
    def genreFreeRating(gen):
        a = df[(df['price'] == 0.0) & (df['prime_genre']== gen)]
        b = a.user_rating.mean()
        return b
    def genrePaidRating(gen):
        a = df[(df['price'] != 0.0) & (df['prime_genre']== gen)]
        b = a.user_rating.mean()
        return b

    def genreRating(gen):
        a = df[df.prime_genre==gen]
        b = a.user_rating.mean()
        return b
    def pricestotal (gen):
        dff = df[df.prime_genre==gen]
    
        a = dff.price.sum()
        return a
    def pricesmean (gen):
        dff = df[df.prime_genre==gen]

        a = dff.price.mean()
        return a


    # Make list of each genre , its free app, paid app and total app . then merge it into one dataframe
    genre_list = []
    genreFree_list = []
    genrePaid_list = []
    genreTotal_list = []
    genreFree_rating_list = []
    genrePaid_rating_list =[]
    genre_rating_list = []

    pricestotal_list = []
    pricesmean_list = []

    # append all details in respective list
    for gen in genres:  
        free_gen = genreFree(gen)
        paid_gen = genrePaid(gen)
        totalapp_gen = free_gen + paid_gen

        genreFree_rating = genreFreeRating(gen)
        genrePaid_rating = genrePaidRating(gen)
        genre_rating = genreRating(gen)

        prices_total = pricestotal(gen)
        prices_mean = pricesmean(gen)

        genre_list.append(gen)
        genreFree_list.append(free_gen)
        genrePaid_list.append(paid_gen)
        genreTotal_list.append(totalapp_gen)
        genreFree_rating_list.append(genreFree_rating)
        genrePaid_rating_list.append(genrePaid_rating)
        genre_rating_list.append(genre_rating)

        pricestotal_list.append(prices_total)
        pricesmean_list.append(prices_mean)

    # Let's make a dataframe of it


    genre_df = pd.DataFrame({
        "genre_name" : genre_list,
        "genre_freeApp" : genreFree_list,
        "genre_paidApp" : genrePaid_list,
        "genre_totalApp" : genreTotal_list,
        "genre_free_App_rating": genreFree_rating_list,
        "genre_paid_App_rating": genrePaid_rating_list,
        "genre_rating":genre_rating_list,
        "pricestotal":pricestotal_list,
        "pricesmean" :pricesmean_list
    },columns=['genre_name','genre_freeApp','genre_paidApp','genre_totalApp','genre_free_App_rating','genre_paid_App_rating','pricestotal','pricesmean','genre_rating'])

    #sorting into descending order
    app_amounts = genre_df.sort_values('genre_totalApp', ascending=False)
    appfree_ratings = genre_df.sort_values('genre_free_App_rating',ascending = False, ignore_index=True)
    apppaid_ratings = genre_df.sort_values('genre_paid_App_rating',ascending = False, ignore_index= True)
    apppaid_ratings = genre_df.sort_values('genre_paid_App_rating',ascending = False, ignore_index= True)



    # remove duplicate genre 
    app_amounts.drop_duplicates('genre_name',inplace=True)
    appfree_ratings.drop_duplicates('genre_name',inplace=True)
    apppaid_ratings.drop_duplicates('genre_name',inplace=True)
    appfree_ratings = appfree_ratings[['genre_name','genre_free_App_rating']]





    x = appfree_ratings['genre_name']
    y1 = appfree_ratings['genre_free_App_rating']
    y2 = apppaid_ratings['genre_paid_App_rating']




    st.subheader('Q1: Are paid apps better than free apps?')
    st.markdown('(use user rating to evaluate)')
    fig, ax = plt.subplots()
    plt.plot(x, y1, label= 'Users rating of free apps',marker = '*', color = 'coral') #coral
    plt.plot(x, y2, label= 'Users rating of paid apps',marker = '.', color = 'violet')
    plt.xticks(range(23), x,rotation=80,color = 'darkblue')
    plt.xlabel('Genre',color = 'darkblue')
    plt.ylabel("Users rating",color = 'darkblue')
    plt.legend()
    st.pyplot(fig)
    st.markdown('Conclusion: According to the diagram,average price rating of paid apps are higher than free apps.')


    # bubble 折线图2
    # Q2:How does the price distribution get affected by category ?

    st.write('\n')

    st.subheader('Q2: How does the price distribution get affected by category ?')
    st.write('\n')
    st.write('\n')

    a = genre_df.sort_values(by = 'genre_rating', ascending = True)
    fig = px.scatter(
        a,
        x="genre_name",
        y="genre_rating",
        hover_data=["genre_name","pricesmean"],   # 列表形式
        color="pricesmean",
        size="genre_totalApp",
        size_max=60,

        )

    st.plotly_chart(fig)
    st.markdown('The size of each bubble shows the quantity of each category. Coll=or at the right side shows the average price.')
    st.markdown('Conclusion: The quantity of games app is the most and the price is not too high. Thus the rating of games is relatively high.')
    # 补充表单


st.write("")
line3_spacer1, line3_1, line3_spacer2 = st.columns((0.1, 3.2, 0.1))

with line3_1:
    st.header('Part 2: Find the inside law of rating')
    

## top 前20的popular app:() rating, 条形图
# 前20 top rating 的 app 是一个关于rating的综述

# Now let's look at the top 20 rating apps
popular_apps = df.sort_values(['user_rating','rating_count_tot'], ascending=False)
popular_apps.head() 

st.write("")
line4_spacer1, line4_1, line4_spacer2 = st.columns((0.1, 3.2, 0.1))
with line4_1:
    st.subheader('Now let\'s look at the top 20 popular apps showing in bar plot.')
    st.markdown('   Remark:(popularity = total count of rating/ user rating)')
    st.write('\n')
    # 这个纵坐标显示什么??
    fig = plt.figure(figsize = (20, 8))                               
    plt.bar(popular_apps['track_name'][0:20], (popular_apps['rating_count_tot']/popular_apps['user_rating'])[0:20]) 
    plt.xticks(rotation=45,ha='right')  
    st.pyplot(fig)
    st.write('\n')
    st.write('\n')
    st.markdown('Conclusion: All of the top 20 are games, the popularity of games is definately wide because it is a kind of recreation.')


st.write("")
line5_spacer1, line5_1, line5_spacer2 = st.columns((0.1, 3.2, 0.1))
with line5_1:
    st.subheader('Let\'s take a look at the highest rating app —— head soccer.')
    st.markdown('You may use VPN to reach this.')
    st.write('\n')
    st.video('https://www.youtube.com/watch?v=ca_sbxTKsxY')
    st.write('It seems like playing this game can release people\'s pressure :smirk:')


st.write("")
line5_spacer1, line5_1, line5_spacer2 = st.columns((0.1, 3.2, 0.1))
with line5_1:
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.subheader('To be specific, choose genres on the sidebar and have a quick look of high rating apps of different genres.')
    st.write('\n')
    # All higher rating applications 
    ratingapp = popular_apps[(popular_apps['user_rating'] == 4.0) | (popular_apps['user_rating'] == 5.0) | (popular_apps['user_rating']==4.5)]
    ratingapp.head(5)


def dountChart(gen,title):  
    # Create a circle for the center of the plot
    circle=plt.Circle( (0,0), 0.7, color='white')
    
    # just keep on user rating as name not overlapping while pie chart plotting
    plt.pie(ratingapp['user_rating'][ratingapp['prime_genre']==gen][0:10], labels= ratingapp['track_name'][ratingapp['prime_genre']==gen][0:10])
    p=plt.gcf() #gcf = get current figure
    p.gca().add_artist(circle)
    plt.title(title , fontname="arial black")
    gens = ['Games','Shopping','Social Networking','Music','Food & Drink', 'Photo & Video','Sports','Finance']

# 函数
gens = ['Games','Shopping','Social Networking','Music','Food & Drink', 'Photo & Video','Sports','Finance']


st.write('\n')

add_selectbox = st.sidebar.radio(
        "Genres",
        ("Games", "Music", "Shopping","Photo & Video")
    )
if add_selectbox=="Games":
    fig = plt.figure(figsize=(25,30))
    plt.subplot(421)
    dountChart(gens[0],'Top Higher rating '+gens[0]+' apps') 
    st.write('\n') 
    st.pyplot(fig)
elif add_selectbox=="Music": 
    fig = plt.figure(figsize=(30,45))
    plt.subplot(421)
    dountChart(gens[3],'Top Higher rating '+gens[3]+' apps')  
    st.write('\n')
    st.pyplot(fig)
elif add_selectbox == "Shopping":
    fig = plt.figure(figsize=(25,30))
    plt.subplot(421)
    dountChart(gens[1],'Top Higher rating '+gens[1]+' apps')  
    st.write('\n')
    st.pyplot(fig)

elif add_selectbox == "Photo & Video":    
    fig = plt.figure(figsize=(25,30))
    plt.subplot(421)
    dountChart(gens[5],'Top Higher rating '+gens[5]+' apps')  
    st.write('\n')
    st.pyplot(fig)

st.write("")
line6_spacer1, line6_1, line6_spacer2 = st.columns((0.1, 3.2, 0.1))
with line6_1:
    st.write("")
    st.subheader('General view of rating distribution')
row2_space1, row2_1, row2_space2, row2_2, row2_space3 = st.columns(
(0.1, 1, 0.1, 1, 0.1)
)   


# rating change:
#  lmplot对所选择的数据集做出了一条最佳的拟合直线
with row2_1:
    fig, ax = plt.subplots(figsize=(10,5))
    free_apps = df[(df.price==0.00)]
    paid_apps  = df[(df.price>0)]
    sns.set_style('white')
    sns.violinplot(x=paid_apps['user_rating'],color='#79FF79')
    plt.xlim(0,5)
    plt.xlabel('Rating (0 to 5 stars)')
    _ = plt.title('Distribution of Paid Apps Ratings')
    st.pyplot(fig)

with row2_2:
    fig,ax = plt.subplots(figsize=(10,5))
    sns.set_style('white')
    sns.violinplot(x=free_apps['user_rating'],color='#66B3FF')
    plt.xlim(0,5)
    plt.xlabel('Rating (0 to 5 stars)')
    _ = plt.title('Distribution of free Apps Ratings')
    st.pyplot(fig)


st.write("")
row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns(
(0.1, 1, 0.1, 1, 0.1)
)
with row3_1:
    fig,ax = plt.subplots(figsize=(10,5))
    bins = (0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5)
    plt.style.use('seaborn-white')
    plt.hist(paid_apps['user_rating'],alpha=.8,bins=bins,color='#79FF79')
    plt.xticks((0,1,2,3,4,5))
    plt.title('Paid Apps - User Ratings ')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    _ = plt.xlim(right=5.5)
    st.pyplot(fig)

with row3_2:
    fig,ax = plt.subplots(figsize=(10,5))
    bins = (0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5)
    plt.style.use('seaborn-white')
    plt.hist(free_apps['user_rating'],alpha=.8,bins=bins,color='#66B3FF')
    plt.xticks((0,1,2,3,4,5))
    plt.title('Free Apps - User Ratings ')
    plt.xlabel('Rating')
    plt.ylabel('Frequency') 
    _ = plt.xlim(right=5.5)
    st.pyplot(fig)
st.markdown('Conclusion: The number of rating of apps around 4.5 is the largest.')


st.write("")
line7_spacer1, line7_1, line7_spacer2 = st.columns((0.1, 3.2, 0.1))
with line7_1:
    st.write("")
    st.subheader('Show the top 5 rating paid pps and las 5 rating paid apps')
row4_space1, row4_1, row4_space2, row4_2, row4_space3 = st.columns(
    (0.1, 1, 0.1, 1, 0.1)
)

with row4_1:
    fig,ax = plt.subplots(figsize=(10,5))
    st.write('\n')
    st.write('\n')
    st.write('\n')
    Top_Apps = paid_apps.sort_values('price', ascending=False)

    sns.barplot(x=Top_Apps.user_rating.head(),y=Top_Apps.track_name.head())
    plt.title('TOP 5 Paid-APPs With User-Rating')
    plt.ylabel('APP Name')
    st.pyplot(fig)


with row4_2:
    fig,ax = plt.subplots(figsize=(10,5))
    st.write('\n')
    st.write('\n')
    st.write('\n')
    Low_Apps = paid_apps.sort_values('price', ascending=True)
    sns.barplot(x=Low_Apps.user_rating.head(),y=Low_Apps.track_name.head())
    plt.title('Lower 5 Paid-APPs With User-Rating')
    plt.ylabel('APP Name')
    st.pyplot(fig)


st.write('\n')
st.write('\n')
st.write('\n')
st.write("")
line8_spacer1, line8_1, line8_spacer2 = st.columns((0.1, 3.2, 0.1))
with line8_1:
    st.subheader('Q3: Which category has the highest rating?')
    st.markdown('Bar plot more directly, set ci = 0')

# Which category has the most highgest rating?
# Does mean user rating depend on the category?

st.write("")
row5_space1, row5_1, row5_space2, row5_2, row5_space3 = st.columns(
(0.1, 1, 0.1, 1, 0.1)
)
with row5_1:
    st.write('\n')
    st.write('\n')
    fig,ax = plt.subplots(figsize=(10,5))
    sns.barplot(y =paid_apps.prime_genre,x =paid_apps['user_rating'],ci = 0)
    plt.title('Paid APPs')
    plt.ylabel('Categories')
    plt.xlabel('Average of user rating')
    st.pyplot(fig)

    st.write('Shopping & Catalogs have the highest rating among paid apps')
with row5_2:
    fig,ax = plt.subplots(figsize=(10,5))
    st.write('\n')
    st.write('\n')
    st.write('\n')
    sns.barplot(y =free_apps.prime_genre,x =free_apps['user_rating'],ci = 0)
    plt.title('Free APPs')
    plt.ylabel('Categories')
    plt.xlabel('Average of user rating')
    st.pyplot(fig)

    st.write('Productivity & Music have the lowesr rating among paid apps')
    st.write('We see that books in paid apps have high mean rating, however very less in free apps. The same in Catalogs.')


st.write('\n')
st.write('\n')
st.write('\n')
st.write("")
line9_spacer1, line9_1, line9_spacer2 = st.columns((0.1, 3.2, 0.1))
with line9_1:
    st.subheader('Q4: What about the total number of user rating(meaningthe feed back) of paid and free apps?')
fig,ax=plt.subplots(1, 2, figsize=(10, 5))
st.write('\n')

g=sns.barplot(x=paid_apps.rating_count_tot,y=paid_apps.prime_genre,ci=0,ax=ax[0])
g.set_xticks([20000,40000,60000])
g.set_ylabel('Categories')
g.set_title('Paid Apps')
f=sns.barplot(x=free_apps.rating_count_tot,y=free_apps.prime_genre,ci=0,ax=ax[1])
f.set_xticks([20000,40000,60000])
f.set_ylabel('Categories')
f.set_title('Free Apps')
f.set_xlabel('Average of total number of user rating')
g.set_xlabel('Average of total number of user rating')
plt.tight_layout()
st.pyplot(fig)

st.write('\n')
st.write('\n')
st.write('Illustrate:Users are less inclined to give feedback or rating to paid apps. However they do in free apps!Side reflection:Free apps reach a much larger audience.Users are more likely to use free apps.(since the average ratings between them are same!)')



    































##结论： 
 # Users don’t give feedback or rating to paid apps. However they do in free 


 ## Does the current version always have more rating than total overall rating?

st.write('\n')
st.write('\n')
st.write('\n')
sns.lmplot(x='user_rating_ver',y='user_rating',data= df)

## 由拟合线性回归图像可以看出，
## Correlation between current version user rating and total overall user rating is 明显正相关的关系， which leads to a strong positive correlation between them as the above plot shows.

## Newer versions of most of the apps have better rating than the median rating. 
## Developers always try to publish a better app that worth a better rating always. 


 
@st.cache
def save_csv():
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    
    return df.to_csv().encode('utf-8')
 
csv = save_csv()
 
st.sidebar.write('Download resources:')
st.sidebar.download_button(
     label="Download iOS apps data",
     data=csv,
     file_name='AppleStore.csv',
     mime='text/csv',
 )

