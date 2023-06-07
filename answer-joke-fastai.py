import streamlit as st
import pandas as pd
from fastai.tabular.all import *
from fastai.collab import *

@st.cache_data
def load_data():
    # Load the data
    data_df = pd.read_excel('jester-data/[final] April 2015 to Nov 30 2019 - Transformed Jester Data - .xlsx')
    data_df.columns = range(data_df.shape[1]) # Rename columns to match with jokes dataframe

    # Convert data_df from wide format to long format
    data_df = data_df.stack().reset_index()
    data_df.columns = ['user', 'joke', 'rating']

    # Filter out missing ratings
    data_df = data_df[data_df['rating'] != 99.0]

    # Load the jokes
    jokes_df = pd.read_excel('jester-data/Dataset4JokeSet.xlsx')
    jokes_df.columns = ['title']
    jokes_df.index.name = 'joke'

    # Merge the ratings with the jokes
    data_df = data_df.merge(jokes_df, left_on='joke', right_index=True)

    return data_df, jokes_df

def train_model(data_df):
    # Create data loaders
    dls = CollabDataLoaders.from_df(data_df, item_name='title', bs=64, user_name='user', rating_name='rating')

    # Train the model
    learn = collab_learner(dls, n_factors=50, y_range=(-10, 10))
    learn.fit_one_cycle(5, 5e-3, wd=0.1)

    return learn

def recommend_jokes(learn, data_df, jokes_df, new_user_id, new_ratings):
    # Convert ratings from 0-5 scale to -10 to 10 scale
    new_ratings = {joke_id: info['rating']*4 - 10 for joke_id, info in new_ratings.items()}

    # Add new user's ratings to the data
    new_ratings_df = pd.DataFrame({
        'user': [new_user_id]*len(new_ratings),
        'joke': list(new_ratings.keys()),
        'rating': list(new_ratings.values()),
        'title': jokes_df.loc[list(new_ratings.keys()), 'title'].values
    })

    data_df = pd.concat([data_df, new_ratings_df])

    # Generate recommendations for the new user
    joke_ids = data_df['joke'].unique()  # Get the list of all joke ids
    joke_ids_new_user = data_df.loc[data_df['user'] == new_user_id, 'joke']  # Get the list of joke ids rated by the new user
    joke_ids_to_pred = np.setdiff1d(joke_ids, joke_ids_new_user)  # Get the list of joke ids the new user has not rated

    # Predict the ratings for all unrated jokes
    testset_new_user = pd.DataFrame({
        'user': [new_user_id] * len(joke_ids_to_pred),
        'joke': joke_ids_to_pred,
        'title': jokes_df.loc[joke_ids_to_pred, 'title'].values
    })

    test_dl = learn.dls.test_dl(testset_new_user)
    preds, _ = learn.get_preds(dl=test_dl)

    # Add predictions to the testset_new_user DataFrame
    testset_new_user['rating'] = preds.numpy()

    # Get the top 5 jokes with highest predicted ratings
    top_5_jokes = testset_new_user.nlargest(5, 'rating')

    return top_5_jokes


def main():
    # Load data
    data_df, jokes_df = load_data()

    # Choose an unused user_id for the new user
    new_user_id = data_df['user'].max() + 1

    # Randomly select 3 jokes for the user to rate
    if 'initial_ratings' not in st.session_state:
        st.session_state.initial_ratings = {}
        random_jokes = jokes_df.sample(3)
        for joke_id, joke in zip(random_jokes.index, random_jokes['title']):
            st.session_state.initial_ratings[joke_id] = {'joke': joke, 'rating': 3}

    # Ask user for ratings
    with st.form(key='initial_ratings_form'):
        for joke_id, info in st.session_state.initial_ratings.items():
            st.write(info['joke'])
            info['rating'] = st.slider('Rate this joke', 0, 5, step=1, value=info['rating'], key=f'first_{joke_id}')
        if st.form_submit_button('Submit Ratings'):
            # Train model
            learn = train_model(data_df)

            # Recommend jokes based on user's ratings
            recommended_jokes = recommend_jokes(learn, data_df, jokes_df, new_user_id, st.session_state.initial_ratings)

            # Save recommended jokes to session state
            st.session_state.recommended_jokes = {}
            for joke_id, joke in zip(recommended_jokes['joke'], recommended_jokes['title']):
                st.session_state.recommended_jokes[joke_id] = {'joke': joke, 'rating': 3}

    # Display recommended jokes and ask for user's ratings
    if 'recommended_jokes' in st.session_state:
        st.write('We recommend the following jokes based on your ratings:')
        with st.form(key='recommended_ratings_form'):
            for joke_id, info in st.session_state.recommended_jokes.items():
                st.write(info['joke'])
                info['rating'] = st.slider('Rate this joke', 0, 5, step=1, value=info['rating'], key=f'rec_{joke_id}')
            if st.form_submit_button('Submit Recommended Ratings'):
                # Calculate the percentage of total possible score
                total_score = sum(info['rating'] for info in st.session_state.recommended_jokes.values())
                percentage_of_total = (total_score / 25) * 100
                st.write(f'You rated the recommended jokes {percentage_of_total}% of the total possible score.')

if __name__ == '__main__':
    main()
