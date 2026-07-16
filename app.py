from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

movies = joblib.load("movie_list.pkl")
similarity = joblib.load("similarity.pkl")

def recommend(movie):

    movie = movie.lower()

    movie_index = None

    for i, title in enumerate(movies['title']):

        if title.lower() == movie:
            movie_index = i
            break

    if movie_index is None:
        return ["Movie Not Found"]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []

    for i in movie_list:
        recommendations.append(movies.iloc[i[0]].title)

    return recommendations


@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = []

    if request.method == "POST":

        movie_name = request.form["movie"]

        recommendations = recommend(movie_name)

    return render_template(
        "index.html",
        recommendations=recommendations
    )


if __name__ == "__main__":
    app.run(debug=True)