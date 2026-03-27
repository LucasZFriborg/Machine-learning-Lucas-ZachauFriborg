# Report

## Data Exploration
During the data exploration, it became clear that the genres provided in the dataset are relatively limited and coarse. There are only twenty different genres, and many movies share the same combinations of these. This makes it difficult to distinguish between movies using genres alone as it limits the quality of recommendations if only this information is used.

On the other hand, the tags dataset contains much richer and more descriptive information. The tags are user-generated and include a much wider variety of words and phrases such as "dark", "sci-fi", and "psychological", providing deeper insight into the themes and tone (e.g. mood or style) of each movie. However, the number of tags per movie varies significantly, meaning that the information is unevenly distributed across the dataset.

## Method
Based on the observations above, a content-based filtering approach was chosen. Instead of relying on ratings, this method uses the features of the movies themselves to determine similarity. By combining genres and tags, each movie can be represented in a more expressive way.

To process the textual data, TF-IDF vectorization was applied. TF-IDF not only counts how often a word appears in a movie description (term frequency), but also reduces the importance of words that appear frequently across many movies (inverse document frequency). This helps emphasize more distinctive terms, making it easier to differentiate between movies with otherwise similar descriptions.

Similarity between movies was computed using cosine similarity. This metric compares the angle between vectors rather than their length, making it suitable for text-based representations where the length of the vectors may vary depending on the number of tags. Together, TF-IDF and cosine similarity provide an effective way to measure how similar two movies are based on their descriptive features.

## Results
The system was tested using several different input movies, and in general the recommendations were relevant and reasonable. For example, when using movies such as The Godfather or one of the Batman movies, the system returned movies with similar themes and characteristics, comparable to recommendations seen on streaming platforms.

A noticeable pattern was that sequels and movies within the same franchise often appeared at the top of the recommendation list. While this indicates that the system successfully captures strong similarities, it can also reduce the diversity of the recommendations. Despite this, the system was still able to suggest additional movies outside of the specific franchise that were relevant in terms of tone and content.

An additional usability feature implemented in the system was a "Did you mean?" selection step. This allows the user to choose between multiple matching titles, which is helpful in cases of ambiguous input or spelling variations. However, this step could be further refined to avoid prompting the user unnecessarily when the input already matches a movie title exactly.

## Discussion and Limitations
One limitation of this approach is that it does not take user preferences into account. Since the system is based purely on content, it produces the same recommendations regardless of who the user is. This differs from collaborative filtering methods, which can provide personalized recommendations based on user behavior when user interaction data is available over time.

Another limitation is the reliance on tag data. Since the tags are unevenly distributed, some movies are described in much greater detail than others, which can affect the quality and consistency of the recommendations. Additionally, tags are user-generated and therefore subject to bias. However, unlike ratings, which reflect how much users like a movie, tags describe the content and characteristics of a movie. This makes them more suitable for capturing similarity between movies, even though some subjectivity remains in how users assign tags.

This system also does not utilize the ratings dataset. While ratings could provide useful information about user preferences, they are inherently subjective and reflect how much users like a movie rather than how similar it is to another movie. Moreover, the ratings dataset is very large (over 30 million entries) and highly sparse, meaning that most users have only rated a small subset of movies. This results in a sparse dataset, meaning that most values are missing. This would make the system more complex and computationally demanding.

## Possible Improvements
There are several ways the system could be improved. One option would be to implement collaborative filtering, which uses user behavior to identify patterns based on historical user interactions, which can enable personalized recommendations when such data is available over time.

A hybrid system combining content-based filtering and collaborative filtering could further improve performance by incorporating both movie similarity and user interaction data, if such data is available.

Another improvement would be to introduce techniques to increase diversity, such as filtering out movies from the same franchise or applying clustering methods like KMeans to ensure a broader range of recommendations.

From a usability perspective, the system would be extended into a graphical interface, for example using a Dash application. This would improve the user experience by making the system more interactive and visually appealing, potentially including movie posters and other metadata (e.g. release year, cast, or genre). In contrast, such an extension would require additional computational resources when working with large datasets.

## Conclusion
In conclusion, the implemented system demonstrates that combining genres and tags with TF-IDF vectorization and cosine similarity can produce appropriate movie recommendations. While the approach has limitations, particularly in terms of personalization and diversity, it provides a solid foundation that can be further extended with more advanced techniques such as collaborative filtering or hybrid models. Overall, I am satisfied with the outcome, especially considering that this is my first time building such a system.