from flask import *
import pandas as pd
import numpy as np
import pickle

popular_df = pickle.load(open('popular.pkl', 'rb'))
pt = pickle.load(open('pt.pkl', 'rb'))
books = pickle.load(open('books.pkl', 'rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl', 'rb'))

app = Flask(__name__,template_folder='templateshruti')


@app.route('/')
def index():
    return render_template('index.html',
                           book_name = list(popular_df['book-title'].values),
                           author=list(popular_df['book-author'].values),
                           image=list(popular_df['image-url-m'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_rating'].values)
                           )


@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')


@app.route('/recommend_books',methods=['post'])
def recommend():
    user_input = request.form.get('user_input')
    index = np.where(pt.index == user_input)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

    data = []
    for i in similar_items:
        items = []
        temp_df = books[books['book-title'] == pt.index[i[0]]]
        items.extend(list(temp_df.drop_duplicates('book-title')['book-title'].values))
        items.extend(list(temp_df.drop_duplicates('book-title')['book-author'].values))
        items.extend(list(temp_df.drop_duplicates('book-title')['image-url-m'].values))

        data.append(items)

    print(data)

    return render_template('recommend.html', data=data)


if __name__ == '__main__':
    app.run(debug=True)
