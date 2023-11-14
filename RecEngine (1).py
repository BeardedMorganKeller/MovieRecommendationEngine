#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np

df = pd.read_csv('movies.csv')

df.head()


# In[26]:


import pandas as pd
import numpy as np

df = pd.read_csv('movies.csv')

df.describe()


# In[1]:


import pandas as pd
import numpy as np
import tkinter as tk
import random
from difflib import SequenceMatcher
from sklearn.metrics import jaccard_score

df = pd.read_csv('movies.csv')

df['genres'] = df['genres'].apply(lambda x: set(x.split('|')))

pattern = r' \((\d{4})\)'

df['year'] = df['title'].str.extract(pattern)
df = df.dropna(subset=['year'])
df['year'] = df['year'].astype(int)
df['title'] = df['title'].str.replace(pattern, '', regex=True).str.strip()


toChoose = random.sample(df['movieId'].tolist(), 10)
dfSub = df[df['movieId'].isin(toChoose)]

initial_state = dfSub.copy()

selected_movies = []


def select_movie():
    selected_index = listbox.curselection()
    if selected_index:
        selected_movie_index = selected_index[0]
        selected_movie_id = dfSub.iloc[selected_movie_index][0]        
        selected_movies.append(selected_movie_id)
        display_selected_movies()
        find_nearest_neighbors(selected_movie_id)
        
        
def display_selected_movies():
    selected_movies_text.config(state="normal")
    selected_movie_titles = []
    for movie_id in selected_movies:
        movie_info = df[df['movieId'] == movie_id]
        if not movie_info.empty:
            title = movie_info['title'].values[0]
            selected_movie_titles.append(title)
            
    selected_movies_text.delete(1.0, tk.END)
    for title in selected_movie_titles:
        selected_movies_text.insert(tk.END, f"{title}\n")
    selected_movies_text.config(state="disabled")
    
    
def weighted_jaccard_similarity(weighted_dictionary: dict, comparator_genres: set):
    numerator = 0
    denominator = weighted_dictionary['total']
    for genre in comparator_genres:
        if genre in weighted_dictionary:
            numerator += weighted_dictionary[genre]
    if denominator > 0:
        return numerator / denominator
    else:
        return 0


def find_nearest_neighbors(movie_id):
    global dfSub
    global initial_state
    genres_weighted_dictionary = find_weight_for_genre(df[df['movieId'].isin(selected_movies)])
    df['weighted_jaccard'] = df['genres'].map(lambda x: weighted_jaccard_similarity(genres_weighted_dictionary, x))
    sorted_df = df.sort_values(by='weighted_jaccard', ascending=False)

    sorted_df = sorted_df[sorted_df['movieId'] != movie_id]
    sorted_df = sorted_df[~sorted_df['movieId'].isin(selected_movies)]

    top_neighbors = sorted_df.head(10)[['movieId', 'title', 'year']]

    listbox.delete(0, tk.END)

    for _, row in top_neighbors.iterrows():
        listbox.insert(tk.END, f"{row['title']} ({row['year']})")

    listbox.config(state=tk.NORMAL)

    dfSub = top_neighbors.copy()
    initial_state = dfSub.copy()

    
def find_weight_for_genre(selected_movies_df):
    genres_weighted_dictionary = {'total': 0}
    for _, movie in selected_movies_df.iterrows():
        for genre in movie['genres']:
            if genre in genres_weighted_dictionary:
                genres_weighted_dictionary[genre] += 1
            else:
                genres_weighted_dictionary[genre] = 1
                
            genres_weighted_dictionary['total'] += 1
    return genres_weighted_dictionary

def search():
    global initial_state
    global dfSub
    
    initial_state = dfSub.copy()
    query = entry.get()
    query_list = query.lower().split()
    
    df['similarities'] = df['title'].map(lambda x: title_comparison(query_list, x))
    
    sorted_df = df.sort_values(by='similarities',ascending = False)
    top_neighbors = sorted_df.head(10)[['movieId', 'title', 'year']]
    
    listbox.delete(0, tk.END)
    
    for _, row in top_neighbors.iterrows():
        listbox.insert(tk.END, f"{row['title']} ({row['year']})")
    
    listbox.config(state=tk.NORMAL)
    
    dfSub = top_neighbors.copy()
    
def clear_results():
    listbox.delete(0, tk.END)
    entry.delete(0, tk.END)

    dfSub = initial_state.copy()
    
    for title, year in zip(dfSub['title'], dfSub['year']):
        listbox.insert(tk.END, f"{title} ({year})")
    listbox.config(state=tk.NORMAL)
        

def title_comparison(query_list: list, x: str):
    title_list = x.lower().split()
    similarity_ratio = SequenceMatcher(None, query_list, title_list).ratio()
    return similarity_ratio

root = tk.Tk()
root.title("Movie Selection")
root.geometry("600x400")
listbox_width = 40
listbox = tk.Listbox(root, selectmode=tk.SINGLE, width=listbox_width)

for title, year in zip(dfSub['title'], dfSub['year']):
    listbox.insert(tk.END, f"{title} ({year})")

select_button = tk.Button(root, text="Select", command=select_movie)


selected_movies_text = tk.Text(root, state="disabled", height=10, width=40)  # Adjust height and width as needed
nearest_neighbors_text = tk.Text(root, state="disabled", height=10, width=40)  # Text widget for nearest neighbors

label = tk.Label(root, text="Enter search query:")
label.pack(pady=10)

entry = tk.Entry(root, width=30)
entry.pack(pady=10)

search_button = tk.Button(root, text="Search", command=search)
search_button.pack(pady=10, padx=5)

clear_button = tk.Button(root, text="Clear", command=clear_results)
clear_button.pack(pady=10, padx=5)

listbox.pack()
select_button.pack()
selected_movies_text.pack()
nearest_neighbors_text.pack()

root.mainloop()


# In[ ]:





# In[9]:


import pandas as pd
import numpy as np
import tkinter as tk
import random
from difflib import SequenceMatcher
from sklearn.metrics import jaccard_score

df = pd.read_csv('movies.csv')

df['genres'] = df['genres'].apply(lambda x: set(x.split('|')))

pattern = r' \((\d{4})\)'

df['year'] = df['title'].str.extract(pattern)
df = df.dropna(subset=['year'])
df['year'] = df['year'].astype(int)
df['title'] = df['title'].str.replace(pattern, '', regex=True).str.strip()


toChoose = random.sample(df['movieId'].tolist(), 10)
dfSub = df[df['movieId'].isin(toChoose)]

initial_state = dfSub.copy()

current_dictionary = []

selected_movies = []
countOfSelect = 0


def select_movie():
    selected_index = listbox.curselection()
    if selected_index:
        selected_movie_index = selected_index[0]
        selected_movie_id = dfSub.iloc[selected_movie_index][0]     
        selected_movies.append(selected_movie_id)
        selected_movie_title = dfSub.iloc[selected_movie_index]['title']
        
        global countOfSelect
        countOfSelect += 1
        current_dictionary.append(selected_movie_title.lower().split())
        display_selected_movies()
        find_nearest_neighbors(selected_movie_id)
        
        
def display_selected_movies():
    selected_movies_text.config(state="normal")
    selected_movie_titles = []
    for movie_id in selected_movies:
        movie_info = df[df['movieId'] == movie_id]
        if not movie_info.empty:
            title = movie_info['title'].values[0]
            selected_movie_titles.append(title)
            
    selected_movies_text.delete(1.0, tk.END)
    for title in selected_movie_titles:
        selected_movies_text.insert(tk.END, f"{title}\n")
    selected_movies_text.config(state="disabled")
    
    
def weighted_jaccard_similarity(weighted_dictionary: dict, comparator_genres: set):
    numerator = 0
    denominator = weighted_dictionary['total']
    for genre in comparator_genres:
        if genre in weighted_dictionary:
            numerator += weighted_dictionary[genre]
    if denominator > 0:
        return numerator / denominator
    else:
        return 0


def find_nearest_neighbors(movie_id):
    global dfSub
    global initial_state
    genres_weighted_dictionary = find_weight_for_genre(df[df['movieId'].isin(selected_movies)])
    
    df['yearFilter'] = df['year'].map(lambda x: year_filter(x))
    
    df['weighted_jaccard'] = df['genres'].map(lambda x: weighted_jaccard_similarity(genres_weighted_dictionary, x))
    df['titleSim'] = df['title'].map(lambda x: title_similarity(x))
    
    sorted_df = df.sort_values(by=['weighted_jaccard', 'titleSim'], ascending=False)
    sorted_df = sorted_df[(sorted_df['yearFilter'] == 1)]

    sorted_df = sorted_df[sorted_df['movieId'] != movie_id]
    sorted_df = sorted_df[~sorted_df['movieId'].isin(selected_movies)]

    top_neighbors = sorted_df.head(10)[['movieId', 'title', 'year']]

    listbox.delete(0, tk.END)

    for _, row in top_neighbors.iterrows():
        listbox.insert(tk.END, f"{row['title']} ({row['year']})")

    listbox.config(state=tk.NORMAL)

    dfSub = top_neighbors.copy()
    initial_state = dfSub.copy()

    
def find_weight_for_genre(selected_movies_df):
    genres_weighted_dictionary = {'total': 0}
    for _, movie in selected_movies_df.iterrows():
        for genre in movie['genres']:
            if genre in genres_weighted_dictionary:
                genres_weighted_dictionary[genre] += 1
            else:
                genres_weighted_dictionary[genre] = 1
                
            genres_weighted_dictionary['total'] += 1
    return genres_weighted_dictionary

def search():
    global initial_state
    global dfSub
    
    initial_state = dfSub.copy()
    query = entry.get()
    query_list = query.lower().split()
    
    query_list = [word[:-1] if word.endswith(':') else word for word in query.lower().split()]
    
    df['similarities'] = df['title'].map(lambda x: title_comparison(query_list, x))
    df['yearFilter'] = df['year'].map(lambda x: year_filter(x))
    
    sorted_df = df[df['similarities'] > 0].sort_values(by='similarities', ascending=False)
    sorted_df = sorted_df[(sorted_df['yearFilter'] == 1)]
    
    top_neighbors = sorted_df.head(10)[['movieId', 'title', 'year']]
    
    listbox.delete(0, tk.END)
    
    for _, row in top_neighbors.iterrows():
        listbox.insert(tk.END, f"{row['title']} ({row['year']})")
    
    listbox.config(state=tk.NORMAL)
    
    dfSub = top_neighbors.copy()
    
def clear_results():
    listbox.delete(0, tk.END)
    entry.delete(0, tk.END)

    dfSub = initial_state.copy()
    
    for title, year in zip(dfSub['title'], dfSub['year']):
        listbox.insert(tk.END, f"{title} ({year})")
    listbox.config(state=tk.NORMAL)    

def title_comparison(query_list: list, x: str):
    title_list = x.lower().split()
    overall_similarity = SequenceMatcher(None, query_list, title_list).ratio()
    
    title_list = [word[:-1] if word.endswith(':') else word for word in title_list]
    
    max_sequence_length = min(len(query_list), len(title_list))
    sequence_similarities = [
        SequenceMatcher(None, query_list[i:i + max_sequence_length], title_list[i:i + max_sequence_length]).ratio()
        for i in range(len(title_list) - max_sequence_length + 1)
    ]
    
    sequence_similarity = max(sequence_similarities, default=0)
    
    boost_factor = 1.5 
    
    combined_similarity = max(overall_similarity, sequence_similarity) * boost_factor
    
    return combined_similarity

def year_filter(x):
    year = int(x)
    start_year = int(start_slider.get())
    end_year = int(end_slider.get())
    if year < start_year or year > end_year:
        return 0
    else:
        return 1
    
def title_similarity(title):
    ratiocompare = 0
    title_list = title.lower().split()
    title_list = [word[:-1] if word.endswith(':') else word for word in title_list]
    
    for word_set in current_dictionary:
        word_set = [word[:-1] if word.endswith(':') else word for word in word_set]
        ratiocompare += len(set(word_set).intersection(title_list))
    return ratiocompare / countOfSelect

def update_label_start(value):
    start_year = int(value)
    end_year = int(end_slider.get())

def update_label_end(value):
    start_year = int(start_slider.get())
    end_year = int(value)

root = tk.Tk()
root.title("Movie Selection")
root.geometry("600x800")
listbox_width = 40
listbox = tk.Listbox(root, selectmode=tk.SINGLE, width=listbox_width)

for title, year in zip(dfSub['title'], dfSub['year']):
    listbox.insert(tk.END, f"{title} ({year})")

select_button = tk.Button(root, text="Select", command=select_movie)


selected_movies_text = tk.Text(root, state="disabled", height=10, width=40)
nearest_neighbors_text = tk.Text(root, state="disabled", height=10, width=40)

label = tk.Label(root, text="Enter search query:")
label.pack(pady=10)

entry = tk.Entry(root, width=30)
entry.pack(pady=10)

search_button = tk.Button(root, text="Search", command=search)
search_button.pack(pady=10, padx=5)

clear_button = tk.Button(root, text="Clear", command=clear_results)
clear_button.pack(pady=10, padx=5)


start_slider = tk.Scale(root, from_=1912, to=2022, orient="horizontal", length=400, label="Start Year", command=update_label_start)
end_slider = tk.Scale(root, from_=1912, to=2022, orient="horizontal", length=400, label="End Year", command=update_label_end)

start_slider.pack(pady=10)
end_slider.pack(pady=10)

label.pack(pady=10)

listbox.pack()
select_button.pack()
selected_movies_text.pack()
nearest_neighbors_text.pack()

root.mainloop()


# In[ ]:





# In[ ]:




