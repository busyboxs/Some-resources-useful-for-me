# Text feature extraction with sklearn

## The Bag of Words representation(词袋表示)

Text Analysis is a major application field for machine learning algorithms. However the raw data, a sequence of symbols cannot be fed directly to the algorithms themselves as most of them expect numerical feature vectors with a fixed size rather than the raw text documents with variable length.

In order to address this, scikit-learn provides utilities for the most common ways to extract numerical features from text content, namely:

* **tokenizing** strings and giving an integer id for each possible token, for instance by using white-spaces and punctuation as token separators.
* **counting** the occurrences of tokens in each document.
* **normalizing** and weighting with diminishing importance tokens that occur in the majority of samples / documents.

In this scheme, features and samples are defined as follows:

* each **individual token occurrence frequency** (normalized or not) is treated as a feature.
* the vector of all the token frequencies for a given document is considered a multivariate sample.

A corpus of documents can thus be represented by a matrix with one row per document and one column per token (e.g. word) occurring in the corpus.

We call `vectorization` the general process of turning a collection of text documents into numerical feature vectors. This specific strategy (tokenization, counting and normalization) is called the **Bag of Words** or “Bag of n-grams” representation. Documents are described by word occurrences while completely ignoring the relative position information of the words in the document.

## Sparsity(稀疏向量)

As most documents will typically use a very small subset of the words used in the corpus, the resulting matrix will have many feature values that are zeros (typically more than 99% of them).

For instance a collection of 10,000 short text documents (such as emails) will use a vocabulary with a size in the order of 100,000 unique words in total while each document will use 100 to 1000 unique words individually.

In order to be able to store such a matrix in memory but also to speed up algebraic operations matrix / vector, implementations will typically use a sparse representation such as the implementations available in the scipy.sparse package.

## Common Vectorizer usage

`CountVectorizer` implements both tokenization and occurrence counting in a single class

```
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=1)

corpus = [
 			'This is the first document.',
    		'This is the second second document.',
    		'And the third one.',
			'Is this the first document?',
		]
X = vectorizer.fit_transform(corpus)
feature_name = vectorizer.get_feature_names()

print feature_name
print X.toarray()
```

result:

```
[u'and', u'document', u'first', u'is', u'one', u'second', u'the', u'third', u'this']
[[0 1 1 1 0 0 1 0 1]
 [0 1 0 1 0 2 1 0 1]
 [1 0 0 0 1 0 1 1 0]
 [0 1 1 1 0 0 1 0 1]]
```

## Tf–idf term weighting

In a large text corpus, some words will be very present (e.g. “the”, “a”, “is” in English) hence carrying very little meaningful information about the actual contents of the document. If we were to feed the direct count data directly to a classifier those very frequent terms would shadow the frequencies of rarer yet more interesting terms.

In order to re-weight the count features into floating point values suitable for usage by a classifier it is very common to use the tf–idf transform.

Tf means term-frequency while tf–idf means term-frequency times inverse document-frequency: $tfidf(t,d) = tf(t,d) * idf(t)$

Using the `TfidfTransformer`’s default settings, `TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)` the term frequency, the number of times a term occurs in a given document, is multiplied with idf component, which is computed as

$$\text{idf}(t) = log \frac{1+n_d}{1+\text{df}(d,t)}+1$$

where $n_d$ is the total number of documents, and $\text{df}(d,t)$ is the number of documents that contain term t. The resulting tf-idf vectors are then normalized by the Euclidean norm:

$$v_{norm} = \frac{v}{\left\lVert v\right\rVert_2}=\frac{v}{\sqrt {v_1^2+v_2^2+\cdots+v_n^2}}$$

This was originally a term weighting scheme developed for information retrieval (as a ranking function for search engines results) that has also found good use in document classification and clustering.

The following sections contain further explanations and examples that illustrate how the tf-idfs are computed exactly and how the tf-idfs computed in scikit-learn’s `TfidfTransformer` and `TfidfVectorizer` differ slightly from the standard textbook notation that defines the idf as

$$idf(t) = log \frac{n_d}{1+df(d,t)}$$

In the `TfidfTransformer` and `TfidfVectorizer` with `smooth_idf=False`, the “1” count is added to the idf instead of the idf’s denominator:

$$\text{idf}(t) = log \frac{n_d}{\text{df}(d,t)}+1$$

```
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)
counts = [[3, 0, 1], [2, 0, 0], [3, 0, 0], [4, 0, 0], [3, 2, 0], [3, 0, 2]]
tfidf = transformer.fit_transform(counts)
print(tfidf.toarray())
```

result:

```
[[ 0.81940995  0.          0.57320793]
 [ 1.          0.          0.        ]
 [ 1.          0.          0.        ]
 [ 1.          0.          0.        ]
 [ 0.47330339  0.88089948  0.        ]
 [ 0.58149261  0.          0.81355169]]
```

For example, we can compute the tf-idf of the first term in the first document in the counts array as follows:

$$
\begin{array} {ll}
n_{d,term1} = 6 \\
df(d,t)_{term1} = 6 \\
idf(d,t)_{term1} = log \frac{n_d}{df(d,t)}+1 = log(1) + 1 = 1 \\
tf-idf_{term1} = tf * idf = 3 \times 1 \\
\end{array}
$$

Now, if we repeat this computation for the remaining 2 terms in the document, we get

$$
\begin{array}{ll}
tf-idf_{term2} = 0 \times (log(6/1)+1) = 0 \\
tf-idf_{term3} = 1 \times (log(6/2)+1) \approx 2.0986 \\
\end{array}
$$

and the vector of raw tf-idfs:

$$tf-idf_{raw} = [3, 0, 2.0986]$$

Then, applying the Euclidean (L2) norm, we obtain the following tf-idfs for document 1:

$$\frac{[3,0,2.0986]}{\sqrt{(3^2+0^2+2.0986^2)}} = [0.819, 0, 0.573]$$

Furthermore, the default parameter `smooth_idf=True` adds “1” to the numerator and denominator as if an extra document was seen containing every term in the collection exactly once, which prevents zero divisions:

$$\text{idf}(t) = log \frac{n_d}{\text{df}(d,t)}+1$$

Using this modification, the tf-idf of the third term in document 1 changes to $1.8473$:

$$tf-idf_{term3} = 1\times log(7/3)+1\approx1.8473$$

And the L2-normalized tf-idf changes to

$$\frac{[3,0,1.8473]}{\sqrt{(3^2+0^2+1.8473^2)}} = [0.8515, 0, 0.5243]$$

As tf–idf is very often used for text features, there is also another class called `TfidfVectorizer` that combines all the options of `CountVectorizer` and `TfidfTransformer` in a single model

