# Word2Vec Application Tutorial

The tutorial was created as a final project for IDCE 30274 Programming in GIS taught by professor Shadrock Robests at Clark University.

In this tutorial, we will learn how to perform basic operations on word vectors. Word vectors represent words as multidimensional continuous floating point numbers where semantically similar words are mapped to proximate points in geometric space. In simpler terms, a word vector is a row of real-valued numbers where each point captures a dimension of the word's meaning and where semantically similar words have similar vectors. While there are many Natural Language Processing (NLP) libraries in Python, such as NLTK, gensim, and spaCy, we will use spaCy in this tutorial. SpaCy is popular NLP library and it provides built-in support for word vectors.

## What is Word2Vec?
Word2Vec is a technique for natural language processing. The word2vec algorithm uses a neural network model to learn word associations from a large corpus of text. Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence. As the name implies, word2vec represents each distinct word with a particular list of numbers called a vector. The vectors are chosen carefully such that a simple mathematical function (the cosine similarity between the vectors) indicates the level of semantic similarity between the words represented by those vectors.

Our goal is to learn:

* Popular Python machine learning packages (spaCy, sklearn)
* Calculating word similarity using Word2Vec model
* Word analogy analysis
* Calculating sentence similarity using Word2Vec model
* Dimension reduction techniques for high-dimensional vectors
* Visualizing Word2Vec in 2D space
* Sentiment analysis using logistic regression and Word2Vec

### To follow this this tutorial you will need:
- A Google account and a Colab notebook.
- Data downloaded from this repo to upload to Colab (or copy to your Google Drive and mount that to Colab).

### Getting started 

First, let's install the spaCy Python library and download their model for the English language. We only need to do it once. Then we can import the spaCy library and other useful libraries such as numpy (used for linear algebra and vector operations in Python). We can load our downloaded English model in our environment.

```Python
!pip install spacy
!python -m spacy download en_core_web_lg
```
Then, import the following packages: 
```Python
# import packages
import spacy
import numpy as np
import csv
from sklearn.manifold import TSNE
from sklearn import linear_model
import matplotlib.pyplot as plt
%matplotlib inline
from IPython.display import clear_output
```
It is also crucial to load English language model if you are going to work with English. Other language models available [here](https://spacy.io/models). 
There are two ways of downloading the language model. The reason why I specify two of them here is because sometimes we can get an error while loading a model. Therefore, it is good to have two options available. Please remember that yiu need to choose *only one* option!

1. 
```Python
nlp = spacy.load('en_core_web_lg')
```
2. 
```Python
import en_core_web_lg
nlp = en_core_web_lg.load()
```

## Word similarity

By representing words in vectors, we can use linear algebra and vector space models to analyze the relationship between words. One simple task is to calculate the cosine of two word vectors, namely the cosine similarity. This cosine similarity measures the semantic similarity of words. While the value ranges from -1 to 1, it is usually used in the non-negative space [0, 1] where 0 means 0 similarity and 1 means extremely similar or even identical.

In order to calculate the cosine similarity between words, we have to know their vector representations first, which are provided by the Word2Vec model. In the spaCy English model, these vector representations (pretrained using Word2Vec) are already provided. All we need to do is to retrieve these words from the spaCy English model and we will have access to these vector representations.

```Python
# retrieve words from the English model vocabulary
lime = nlp.vocab['lime']
bike = nlp.vocab['bike']
storm = nlp.vocab['storm']

# print the dimension of word vectors
print('vector length:', len(lime.vector))

# print the word vector
print('lime:', lime.vector)
```








### Why this lab is important
In this lab, we'll be using a Pandas dataframe to combine different parts of datasets to create an interactive map. In Pandas, you can think of the dataframe as a version of a spreadsheet: it is the most commonly used Pandas data structure. You can [learn more about Pandas data structures here](https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html). You'll also learn about Folium. I mean... interactive maps, who can resist?!?! Finally, we're going to critically look at some of the choices we're making with our inputs and how the data we're using actually create a lousy map.

# Choropleth Map Using Zip Codes (generally a terrible idea)
This tutorial uses a `.csv` and two `.geojson` files that can be found in the [data folder](data). We'll start by loading our files in Colab (see here for [a variety of other ways you can bring your data into Colab](https://colab.research.google.com/notebooks/io.ipynb))

```Python
# Upload local script to Colab - running this creates a "choose file" button to upload local files.
from google.colab import files
uploaded = files.upload()
```

Next, we'll import all the needed libraries, create a dataframe (variable `df`) into which we'll put our `.csv` data and load up some spatial data in the `laMap` file.

```Python
import folium
import pandas as pd
import json
from folium import plugins

df = pd.read_csv('starbucksInLACounty.csv')

with open('laMap.geojson') as f:
    laArea = json.load(f)
```
The code below pulls together the needed pieces from our data to create one: we need to get a count for the number of stores in each zip code, then attach this to the appropriate geometry for that zip code using a Pandas "dataframe."

```Python
# group the starbucks dataframe by zip code and count the number of stores in each zip code
numStoresSeries = df.groupby('zip').count().id

# initialize an empty dataframe to store this new data
numStoresByZip = pd.DataFrame()

# populate the new dataframe with a 'zipcode' column and a 'numStores' column
numStoresByZip['zipcode'] = [str(i) for i in numStoresSeries.index]
numStoresByZip['numStores'] = numStoresSeries.values

```

Now that we have our data... let's create a map! A lot of the code you see below is explained in [the Quickstart Guide for Folium](https://python-visualization.github.io/folium/quickstart.html).

```Python
# Initiatlize a new map.
laMap = folium.Map(location=[34.0522,-118.2437], tiles='cartodbpositron', zoom_start=9, attr ='<a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors, <a href="https://public.carto.com/viz/397fd294-a82b-4470-90cc-6153ebad5bf6/embed_map">Positron by Carto</a> | Data from <a href = "https://github.com/ritvikmath/StarbucksStoreScraping">Ritvik Kharkar</a>')

# Create the choropleth map. Key components have explanatory comments.
folium.Choropleth(
    geo_data = 'laZips.geojson',         # the geojson which you want to draw on the map [in our case it is the zipcodes in LA County]
    name='choropleth',
    data= numStoresByZip,                # the pandas dataframe which contains the zipcode information and the values of the variable you want to plot on the choropleth
    columns=['zipcode', 'numStores'],    # the columns from the dataframe that you want to use
    nan_fill_color='grey',               # fill color for null values
    nan_fill_opacity=0.4,                # opacity for null values
    key_on='feature.properties.zipcode', # the common key between one of your columns and an attribute in the geojson. This is how python knows which dataframe row matches up to which zipcode in the geojson
    fill_color='YlOrRd',                 # Try some other colors: 'YlGn', 'OrRd', 'BuGn' 'BuPu', 'GnBu', 'PuBu', 'PuBuGn', 'PuRd', 'RdPu', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd'.
    fill_opacity = 0.7,                  # fill color for data values
    line_opacity = 0.2,                  # opacity for data values
    legend_name = 'Number of Starbucks'
).add_to(laMap)

folium.LayerControl().add_to(laMap)

# Display your map
laMap
```
Your map should look something like this:
![LA Zip Map](images/LA-Zip-Map.png)

It can be difficult to understand how to get all the components in place for a choropleth, so let’s take a look at how it works. The choropleth needs to know what color to fill in for zip code 90001, for example. It checks the pandas dataframe referenced by the `data field`, searches the `key_on` column for the zip code and finds the other column listed in columns which is `numStores`. It then knows that it needs to fill in the color corresponding to 3 stores in zip code 90001. It then looks in the GeoJSON referenced by the `geo_data` field, and finds zip code 90001 and its associated shape info, which tells it which shape to draw for that zip code on the map. Through these links, it has all the necessary information.

Another way to think about this is with a visual:
![](images/CSV2GeoJson.png)

### Other things to explore in Folium
When you first establish your map, there are a variety of parameters to set, including the attribution, which accepts basic `HTML` tags. I've purposefully referenced OpenStreetMap, Carto, and Ritvik Kharkar because they all created some part of the map (base layers, data, etc.), even though we've coded it. However, you can change the attribution to acknowledge whatever basemap style or data you are using.

The legend on the upper right is automatically generated for your values using 6 same sized bins. However, passing your own bins (number or list) is simple and can be found in the [Folium Quickstart Guide](https://python-visualization.github.io/folium/quickstart.html).  

### So why is this a terrible idea?
Let's take a close look at your map. Have a look at the northern part of LA county: there appear to be a small number of Starbucky's spread out over a rather large area... and those areas are just north of zip codes that are greyed out because they contain no data (hence, no Starbucky's).

Let's change another variable in your map. Let's change the `tiles = ` parameter in `laMap` from `'cartodbpositron'` to `'Stamen Terrain'`. Now what do you see? You should note that the null value areas make sense because they are basically covering mountains! You should also see that the zip codes covering the towns of Quartz Hill and Palmdale in the north of the county don't really fit around what look like city limits.

In using zip codes, we basically choose a random geography (shape) into which we binned the point data we have for Starbucky's. And in doing so we created a rather random map **because zip codes aren't connected to any real human phenomena other than mail delivery.** When we use other administrative units, such as state, counties, census tracts or blocks, we're using geographies that relate to people and have been specifically designed to make some statistical sense or to relate to something in the 'real world'. This, of course, is why [gerrymandering](https://en.wikipedia.org/wiki/Gerrymandering) is such a big deal in our country: political parties create voting districts using completely made up geographies that don't really represent the underlying population.

In GIS, the geography you choose to aggregate your data into, can radically alter the representation of your data. We call this the "[Modifiable Areal Unit Problem](https://www.e-education.psu.edu/sgam/node/214)" or MAUP. Have a look at the image below: exact same point data, represented as a chorolpeth using 3 different zones.

![MAUP](images/MAUP.png)

Unfortunately, I see really good coders (and even some GISers) who don't understand the concepts of geography who still aggregate data into zip codes and it drives geographers crazy. [Carto has a nice blog post about this](https://carto.com/blog/zip-codes-spatial-analysis/). Whenever you aggregate data into "zonal units" (e.g. shapes, geographies, whatever you want to call it) think carefully about _why_ you are using those particular shapes.

So, what should we do with our map? Well, the Python concepts of creating data frames that link several bits of data together are still valid... but let's create another map... maybe one that's a bit more accurate.

# Keep it Simple
The question that started this tutorial, "How many Starshucks are in each zip code", is conceptually flawed: zip codes are good for mail routes, but not much else. In practice stick to using things like census blocks, tracts, etc... or, why not just stick to addresses and exact locations? If we want to look at distribution of Starflunks in LA... well, then let's just make a map that shows the distribution of Starflunks in LA! Handily enough, creating a basic point map of all Starbucks in LA County from the latitude/longitude pairs in our dataframe is pretty straightforward.

```python
# initialize the map around LA County
laMap = folium.Map(location=[34.0522,-118.2437], tiles='Stamen Toner', zoom_start=9)

# add the shape of LA County to the map
folium.GeoJson(laArea).add_to(laMap)

# for each row in the Starbucks dataset, plot the corresponding latitude and longitude on the map
for i,row in df.iterrows():
    folium.CircleMarker((row.latitude,row.longitude), radius=3, weight=2, color='red', fill_color='red', fill_opacity=.5).add_to(laMap)

# Display your map   
laMap
```
Your map should look something like this:

![LA Zip Map](images/LA-point-map.png)

We can clearly see all the Starbucks in LA County as red points within the LA County region (in blue). We've also used a base map that highlights the road network (because our areas of interest are all located on roads) and that highlights national forest areas (zoom in and you'll see it). Of course, you can customize any of the colors and shapes of the points by passing some different parameters into `folium.CircleMarker`. Compare this to the map above... which one gives a more "real" understanding of the distribution of StarJunks?

# Bonus Choropleth!
Given that we've created a "bad" choropleth using inappropriate geographies, let's quickly bang out a choropleth that uses appropriate geographies. We'll map unemployment by state using 2012 census data.

In this case, states are a great geography to use because each state has a host of laws related to being employed within that particular "zone." So looking at unemployment by state boundaries means that our map would actually show us something meaningful.

The code below is taken from the Folium documentation and uses a nifty trick to access data stored in a Github repo by that repo's URL. If you want to access the data files, see [the data folder in the Folium Github repo here](https://github.com/python-visualization/folium/tree/master/examples/data).

Another valuable thing to note is that the data we're using here map the unemployment rate _as a percent of population_. Whenever possible, [map rates instead of counts](https://www.e-education.psu.edu/natureofgeoinfo/c3_p17.html). Think of it this way. If 10 people in State A and State B are unemployed, and you map straight counts: both states would appear to be the same. But if State A has 100 people (10% unemployment) and State B has 1,000 people (1%), then you actually have very different levels of unemployment!

```python
url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data' #see files at https://github.com/python-visualization/folium/tree/master/examples/data
state_geo = f'{url}/us-states.json'
state_unemployment = f'{url}/US_Unemployment_Oct2012.csv'
state_data = pd.read_csv(state_unemployment)

statemap = folium.Map(location=[48, -102], zoom_start=3)

folium.Choropleth(
    geo_data=state_geo,
    name='choropleth',
    data=state_data,
    columns=['State', 'Unemployment'],
    key_on='feature.id',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='2012 Unemployment Rate (%)'
).add_to(statemap)

folium.LayerControl().add_to(statemap)

#render your map
statemap
```
Your map should look something like this:

![LA Zip Map](images/State_choropleth.png)

Experiment with some of Foliums features: change basemaps, attribution, symbology etc. Happy coding!

# Citation
This tutorial is based on [this Medium Tutorial by Ritvik Kharkar](https://towardsdatascience.com/making-3-easy-maps-with-python-fb7dfb1036) and the associated [Github repo for his tutorial](https://github.com/ritvikmath/StarbucksStoreScraping). I've updated deprecated Folium code; re-orderd some of the steps to explain how his use of zip codes as a geography, while well intentioned and executed, is a huge conceptual problem; and added a brief follow-up showing a solid use of administrative units for choropleth maps.
