Download Link: https://assignmentchef.com/product/solved-text-classification-assignment-2
<br>
For this assignment, we’ll be building a text classifier. The goal of our text classifer will be to distinguish between words that are simple and words that are complex. Example simple words are <em>heard, sat, feet, shops, town</em>, and example complex words are <em>abdicate, detained, liaison, vintners</em>. Distinguishing between simple and complex words is the first step in a larger NLP task called text simplification, which aims to replace complex words with simpler synonyms. Text simplification is potentially useful for re-writing texts so that they can be more easily understood by younger readers, people learning English as a second language, or people with learning disabilities.

The learning goals of this assignment are:

<ul>

 <li>Understand an important class of NLP evaluation methods (precision, recall and F1), and implement them yourself.</li>

 <li>Employ common experimental design practices in NLP. Split the annotated data into training/development/test sets, implement simple baselines to determine how difficult the task is, and experiment with a range of features and models.</li>

 <li>Get an introduction to sklearn, an excellent machine learning Python package.</li>

</ul>

We will provide you with training and development data that has been manually labeled. We will also give you a test set without labels. You will build a classifier to predict the labels on our test set. You can upload your classifier’s predictions to Gradescope. We will score its predictions and maintain a leaderboard showing whose classifier has the best performance.

<h2 id="identifying-complex-words">Identifying Complex Words</h2>

Automated text simplification is an NLP task, where the goal is to take as input a complex text, and return a text that is easier to understand. One of the most logical first steps in text simplification, and example of text classification, is identifying which words in a text are hard to understand, and which words are easy to understand.

We have prepared a labeled training set for this assignment. We provide a dataset of words and their corresponding sentences that has been split into training, development, and test sets. The training set is disjoint, so if a word appears in the training set, it will not also appear in the test set or the development set.

This dataset was collected by taking the first 200 tokens in 200 complex texts, and crowdsourcing human judgements. We asked nine human annotators to identify at least 10 complex words in each text. From here, words that were identified as complex by at least 3 annotators were labeled as complex. In addition, words that were identified as complex by zero annotators were labeled as simple. One thing to note is that we kept only nouns, verbs, adjectives, and adverbs, and removed stopwords (i.e. common words like <code class="highlighter-rouge">the</code> or <code class="highlighter-rouge">and</code>) and proper nouns. After this filtering, we were left with 5,922 unique words. For this homework, we split these words up into 4,000 words for training, 1,000 words for development, and the remaining 922 words are reserved for testing.

Shown below is an example of the training data. Note that the training data and development data files have the same formatting, and the test data does not include the label column:

<table>

 <thead>

  <tr>

   <th>WORD</th>

   <th>LABEL</th>

   <th>ANNOTATORS</th>

   <th>SENTENCE</th>

   <th>SENTENCE_INDEX</th>

  </tr>

 </thead>

 <tbody>

  <tr>

   <td>jumping</td>

   <td>0</td>

   <td>0</td>

   <td>“ Coleman with his jumping frog – bet stranger $ 50 – stranger had no frog &amp; C got him one – in the meantime stranger filled C ‘s frog full of shot &amp; he could n’t jump – the stranger ‘s frog won . ‘’</td>

   <td>4</td>

  </tr>

  <tr>

   <td>paths</td>

   <td>0</td>

   <td>0</td>

   <td>The Cannery project will feature drought-tolerant landscaping along its bike paths , and most of the front yards will be landscaped with low-water plants in place of grass .</td>

   <td>10</td>

  </tr>

  <tr>

   <td>banks</td>

   <td>0</td>

   <td>0</td>

   <td>Extending their protests into the workweek , Hong Kong democracy activists continued occupying major thoroughfares Monday , forcing the closure of some schools , banks and other businesses in the semi-autonomous Chinese territory .</td>

   <td>24</td>

  </tr>

  <tr>

   <td>fair-weather</td>

   <td>1</td>

   <td>5</td>

   <td>Months ago , many warned him not to invest in a place where fair-weather tourists flee in the fall and the big lake ‘s waters turn cold and storm-tossed , forcing the 100 or so hardy full-time residents of Cornucopia to hibernate for the winter .</td>

   <td>13</td>

  </tr>

  <tr>

   <td>krill</td>

   <td>1</td>

   <td>7</td>

   <td>But unlike the other whales , the 25-foot-long young adult stuck near the surface – and it did n’t dive down to feast on the blooms of krill that attract humpbacks to the bay .</td>

   <td>27</td>

  </tr>

  <tr>

   <td>affirmed</td>

   <td>1</td>

   <td>8</td>

   <td>CHARLESTON , S.C. – A grand jury affirmed the state of South Carolina ‘s murder charge on Monday against a white former North Charleston police officer who fatally shot an unarmed black man trying to run from a traffic stop .</td>

   <td>7</td>

  </tr>

 </tbody>

</table>

Here is what the different fields in the file mean:

<ul>

 <li>WORD: The word to be classified</li>

 <li>LABEL: 0 for simple words, 1 for complex words</li>

 <li>ANNOTATORS: The number of annotators who labeled the word as complex</li>

 <li>SENTENCE: The sentence that was shown to annotators when they labeled the word as simple or complex</li>

 <li>SENTENCE_INDEX: The index of the word in the sentence (0 indexed, space delimited).</li>

</ul>

We have provided the function <code class="highlighter-rouge">load_file(data_file)</code>, which takes in the file name (<code class="highlighter-rouge">data_file</code>) of one of the datasets, and reads in the words and labels from these files. CLARIFICATION: You should make sure your load_file() function makes every word lowercase. We have edited the skeleton to do so as of 1/18.

Note: While the context from which each word was found is provided, you do not need it for the majority of the assignment. The only time you may need this is if you choose to implement any context-based features in your own classifier in Section 4.

<h2 id="1-implement-the-evaluation-metrics">1. Implement the Evaluation Metrics</h2>

Before we start with this text classification task, we need to first determine how we will evaluate our results. The most common metrics for evaluating binary classification (especially in cases of class imbalance) are precision, recall, and f-score. For this assignment, complex words are considered positive examples, and simple words are considered negative examples.

For this problem, you will fill in the following functions:

<ul>

 <li><code class="highlighter-rouge">get_precision(y_pred, y_true)</code></li>

 <li><code class="highlighter-rouge">get_recall(y_pred, y_true)</code></li>

 <li><code class="highlighter-rouge">get_fscore(y_pred, y_true)</code></li>

</ul>

Here, <code class="highlighter-rouge">y_pred</code> is list of predicted labels from a classifier, and <code class="highlighter-rouge">y_true</code> is a list of the true labels.

You may <strong>not</strong> use sklearn’s built-in functions for this, you must instead write your own code to calculate these metrics. You will be using these functions to evaluate your classifiers later on in this assignment.

We recommend that you also write a function <code class="highlighter-rouge">test_predictions(y_pred, y_true)</code>, which prints out the precision, recall, and f-score. This function will be helpful later on!

<h2 id="2-baselines">2. Baselines</h2>

<h3 id="implement-a-majority-class-baseline">Implement a majority class baseline</h3>

You should start by implementing simple baselines as classifiers. Your first baseline is a majority class baseline which is one of the most simple classifier. You should complete the function <code class="highlighter-rouge">all_complex(data_file)</code>, which takes in the file name of one of the datasets, labels each word in the dataset as complex, and returns out the precision, recall, and f-score.

Please report the precision, recall, and f-score on both the training data and the development data individually to be graded.

<h3 id="word-length-baseline">Word length baseline</h3>

For our next baseline, we will use a slightly complex baseline, the length of each word to predict its complexity.

For the word length baseline, you should try setting various thresholds for word length to classify them as simple or otherwise. For example, you might set a threshold of 9, meaning that any words with less than 9 characters will be labeled simple, and any words with 9 characters or more will be labeled complex. Once you find the best threshold using the training data, use this same threshold for the development data as well.

You will be filling in the function <code class="highlighter-rouge">word_length_threshold(training_file, development_file)</code>. This function takes in both the training and development data files, and returns out the precision, recall, and f-score for your best threshold’s performance on both the training and development data.

Usually, Precision and Recall are inversely related and while building binary-classification systems we try to find a good balance between them (by maximizing f-score, for example). It is often useful to plot the Precision-Recall curve for various settings of the classifier to gauge its performance and compare it to other classifiers. For example, for this baseline, a Precision-Recall curve can be plotted by plotting the Precision (on the y-axis) and Recall (on the X-axis) for different values of word-length threshold.

In your write-up, please report the precision, recall, and f-score for the training and development data individually, along with the range of thresholds you tried. Also plot the Precision-Recall curve for the various thresholds you tried. For plotting, <a href="https://matplotlib.org/">matplotlib</a> is a useful python library.

<h3 id="word-frequency-baseline">Word frequency baseline</h3>

Our final baseline is a classifier similar to the last one, but thresholds on word frequency instead of length. We have provided Google NGram frequencies in the text file <code class="highlighter-rouge">ngram_counts.txt</code>, along with the helper function <code class="highlighter-rouge">load_ngram_counts(ngram_counts_file)</code> to load them into Python as a dictionary.

You will be filling in the function <code class="highlighter-rouge">word_frequency_threshold(training_file, development_file, counts)</code>, where <code class="highlighter-rouge">counts</code> is the dictionary of word frequencies. This function again returns the precision, recall, and fscore for your best threshold’s performance on both the training and development data.

Please again report the precision, recall, and f-score on the training and development data individually, along with the range of thresholds you tried, and the best threshold to be graded. Similar to the previous baseline, plot the Precision-Recall curve for range of thresholds you tried. Also, make a third plot that contains the P-R curve for both the baseline classifier. Which classifier looks better <em>on average</em>?

Note: Due to its size, loading the ngram counts into Python takes around 20 seconds, and finding the correct threshold may take a few minutes to run.

<h2 id="3-classifiers">3. Classifiers</h2>

<h3 id="naive-bayes-classification">Naive Bayes classification</h3>

Now, let’s move on to actual machine learning classifiers! For our first classifier, you will use the built-in Naive Bayes model from sklearn, to train a classifier. You should refer to the online <a href="http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html">sklearn documentation</a> when you are building your classifier.

The first thing to note is that sklearn classifiers take in <code class="highlighter-rouge">numpy</code> arrays, rather than regular lists. You may use the online <code class="highlighter-rouge">numpy</code> documentation. To create a <code class="highlighter-rouge">numpy</code> list of length 5, you can use the following Python commands:

<figure class="highlight">

 <pre><code class="language-python" data-lang="python"><span class="o">&gt;&gt;&gt;</span> <span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="n">np</span><span class="o">&gt;&gt;&gt;</span> <span class="n">X</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">,</span><span class="mi">3</span><span class="p">,</span><span class="mi">4</span><span class="p">,</span><span class="mi">5</span><span class="p">])</span></code></pre>

</figure>

To train a classifier, you need two <code class="highlighter-rouge">numpy</code> arrays: <code class="highlighter-rouge">X_train</code>, an <code class="highlighter-rouge">m</code> by <code class="highlighter-rouge">n</code> array, where <code class="highlighter-rouge">m</code> is the number of words in the dataset, and <code class="highlighter-rouge">n</code> is the number of features for each word; and <code class="highlighter-rouge">Y</code>, an array of length <code class="highlighter-rouge">m</code> for the labels of each of the words. Once we have these two arrays, we can fit a Naive Bayes classifier using the following commands:

<figure class="highlight">

 <pre><code class="language-python" data-lang="python"><span class="o">&gt;&gt;&gt;</span> <span class="kn">from</span> <span class="nn">sklearn.naive_bayes</span> <span class="kn">import</span> <span class="n">GaussianNB</span><span class="o">&gt;&gt;&gt;</span> <span class="n">clf</span> <span class="o">=</span> <span class="n">GaussianNB</span><span class="p">()</span><span class="o">&gt;&gt;&gt;</span> <span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">Y</span><span class="p">)</span></code></pre>

</figure>

Finally, to use your model to predict the labels for a set of words, you only need one <code class="highlighter-rouge">numpy</code> array: <code class="highlighter-rouge">X_test</code>, an <code class="highlighter-rouge">m'</code> by <code class="highlighter-rouge">n</code> array, where <code class="highlighter-rouge">m'</code> is the number of words in the test set, and <code class="highlighter-rouge">n</code> is the number of features for each word. Note that the <code class="highlighter-rouge">n</code> used here is the same as the <code class="highlighter-rouge">n</code> in <code class="highlighter-rouge">X_train</code>. Then, we can use our classifier to predict labels using the following command:

<figure class="highlight">

 <pre><code class="language-python" data-lang="python"><span class="o">&gt;&gt;&gt;</span> <span class="n">Y_pred</span> <span class="o">=</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span></code></pre>

</figure>

You should fill in the function <code class="highlighter-rouge">naive_bayes(training_file, development_file, counts)</code>. This function will train a <code class="highlighter-rouge">Naive Bayes</code> classifier on the training data using word length and word frequency as features, and returns your model’s precision, recall, and f-score on the training data and the development data individually.

In your write-up, please report the precision, recall, and f-score on the training and development data for your Naive Bayes classifier that uses word length and word frequency.

NOTE: Before training and testing a classifier, it is generally important to normalize your features. This means that you need to find the mean and standard deviation (sd) of a feature. Then, for each row, perform the following transformation:

<code class="highlighter-rouge">X_scaled = (X_original - mean)/sd</code>

Be sure to always use the means and standard deviations from the <code class="highlighter-rouge">training data</code>.

<h3 id="logistic-regression">Logistic Regression</h3>

Next, you will use sklearn’s built-in Logistic Regression classifier. Again, we will use word length and word frequency as your two features. You should refer to the online <a href="http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html">sklearn documentation</a> when you are building your classifier. To import and use this model, use the following command:

<figure class="highlight">

 <pre><code class="language-python" data-lang="python"><span class="o">&gt;&gt;&gt;</span> <span class="kn">from</span> <span class="nn">sklearn.linear_model</span> <span class="kn">import</span> <span class="n">LogisticRegression</span><span class="o">&gt;&gt;&gt;</span> <span class="n">clf</span> <span class="o">=</span> <span class="n">LogisticRegression</span><span class="p">()</span></code></pre>

</figure>

For this problem, you will be filling in the function <code class="highlighter-rouge">logistic_regression(training_file, development_file, counts)</code>. This function will train a <code class="highlighter-rouge">Logistic Regression</code> classifier on the training data, and returns your model’s precision, recall, and f-score on the training data and the development data individually.

Again, please report the precision, recall, and f-score on the training and development data.

<h3 id="comparing-naive-bayes-and-logistic-regression">Comparing Naive Bayes and Logistic Regression</h3>

After implementing Naive Bayes and Logistic Regression classifiers, you will notice that their performance is not identical, even though they are given the same data. Add a paragraph to your write up that discusses which model performed better on this task.

<h2 id="4-build-your-own-model">4. Build your own model</h2>

Finally, the fun part! In this section, you will build your own classifier for the complex word identification task, and compare your results to that of your classmates. You will also perform an error analysis for your best performing model.

You can choose any other types of classifier, and any additional features you can think of! For classifiers, beyond Naive Bayes and Logistic Regression, you might consider trying <code class="highlighter-rouge">SVM</code>, <code class="highlighter-rouge">Decision Trees</code>, and <code class="highlighter-rouge">Random Forests</code>, among others. Additional word features that you might consider include number of syllables, number of WordNet synonyms, and number of WordNet senses . For counting the number of syllables, we have provided a python script <a href="http://computational-linguistics-class.org/downloads/hw2/syllables.py">syllables.py</a> that contains the function <code class="highlighter-rouge">count_syllables(word)</code>, which you may use. To use WordNet in Python, refer to this <a href="http://www.nltk.org/howto/wordnet.html">documentation</a>. You could also include sentence-based complexity features, such as length of the sentence, average word length, and average word frequency.

When trying different classifiers, we recommend that you train on training data, and test on the development data, like the previous sections.

In your writeup, please include a description of all of the models and features that you tried. To receive full credit, you MUST try at least 1 type of classifier (not including Naive Bayes and Logistic Regression), and at least two features (not including length and frequency).

Note: You can also tune the parameters of your model, e.g. what type of kernel to use. This is NOT required, as some of you may not be that familiar with this.

<h3 id="analyze-your-model">Analyze your model</h3>

An important part of text classification tasks is to determine what your model is getting correct, and what your model is getting wrong. For this problem, you must train your best model on the training data, and report the precision, recall, and f-score on the development data.

In addition, need to perform a detailed error analysis of your models. Give several examples of words on which your best model performs well. Also give examples of words which your best model performs poorly on, and identify at least TWO categories of words on which your model is making errors.

<h3 id="leaderboard">Leaderboard</h3>

Finally, train your best model on both the training and development data. You will use this classifier to predict labels for the test data, and will submit these labels in a text file named <code class="highlighter-rouge">test_labels.txt</code> (with one label per line) to the leaderboard; be sure NOT to shuffle the order of the test examples. Instructions for how to post to the leaderboard will be posted on Piazza soon.

The performances of the baselines will be included on the leaderboard. In order to receive full credit, your model must be able to outperform all of the baselines. In addition, the top 3 teams will receive 5 bonus points!

<h3 id="optional-leaderboard-using-outside-data">(Optional) Leaderboard using outside data</h3>

While the training data we have provided is sufficient for completing this assignment, it is not the only data for the task of identifying complex words. As an optional addition to this homework, you may look for and use any additional training data, and submit your predicted labels in a text file named <code class="highlighter-rouge">test_labels.txt</code> to a separate leaderboard.

As a start, we recommend looking at the <a href="http://alt.qcri.org/semeval2016/task11/">SemEval 2016 dataset</a>, a dataset that was used in a complex words identification competition. In addition, you can try to use data from <a href="https://newsela.com/">Newsela</a>. Newsela’s editors re-write newspaper articles to be appropriate for students at different grade levels. The company has generously shared a dataset with us. The Newsela data <strong>may not</strong> be re-distributed outside of Penn. You can find the data on eniac at <code class="highlighter-rouge">/home1/c/ccb/data/newsela/newsela_article_corpus_with_scripts_2016-01-29.1.zip</code>.

Good luck, and have fun!

<h2 id="5-deliverables">5. Deliverables</h2>

<h2 id="6-recommended-readings">6. Recommended readings</h2>

<table>

 <tbody>

  <tr>

   <td><a href="https://web.stanford.edu/~jurafsky/slp3/6.pdf">Naive Bayes Classification and Sentiment.</a> Dan Jurafsky and James H. Martin. Speech and Language Processing (3rd edition draft) .</td>

  </tr>

  <tr>

   <td><a href="https://web.stanford.edu/~jurafsky/slp3/7.pdf">Logistic Regression.</a> Dan Jurafsky and James H. Martin. Speech and Language Processing (3rd edition draft) .</td>

  </tr>

  <tr>

   <td><a href="https://www.cis.upenn.edu/~ccb/publications/new-data-for-text-simplification.pdf">Problems in Current Text Simplification Research: New Data Can Help.</a> Wei Xu, Chris Callison-Burch, and Courtney Napoles. TACL 2015. <a class="label label-success" href="http://computational-linguistics-class.org/assignment2.html#new-data-for-text-simplification-abstract" data-toggle="modal">Abstract</a><a class="label label-default" href="http://computational-linguistics-class.org/assignment2.html#new-data-for-text-simplification-bibtex" data-toggle="modal">BibTex</a></td>

  </tr>

  <tr>

   <td><a href="http://aclweb.org/anthology/P/P13/P13-3015.pdf">Comparison of Techniques to Automatically Identify Complex Words.</a> Matthew Shardlow. ACL 2013.</td>

  </tr>

  <tr>

   <td><a href="https://www.researchgate.net/profile/Gustavo_Paetzold/publication/305334627_SemEval_2016_Task_11_Complex_Word_Identification/links/57bab70a08ae14f440bd9722/SemEval-2016-Task-11-Complex-Word-Identification.pdf">SemEval 2016 Task 11: Complex Word Identification.</a> Gustavo Paetzold and Lucia Specia. ACL 2016.</td>

  </tr>

 </tbody>

</table>

5/5 - (5 votes)

Here are the materials that you should download for this assignment:

<ul>

 <li><a href="http://computational-linguistics-class.org/downloads/hw2/hw2_skeleton.py">Skeleton code</a> – this provides some of the functions that you should implement.</li>

 <li><a href="http://computational-linguistics-class.org/downloads/hw2/data.tar.gz">Data sets</a> – this is a tarball with the training/dev/test sets.</li>

 <li><a href="https://www.cis.upenn.edu/~cis530/18sp/data/ngram_counts.txt.gz">Unigram counts</a> from the <a href="https://research.googleblog.com/2006/08/all-our-n-gram-are-belong-to-you.html">Google N-gram corpus</a>.</li>

</ul>

Here are the deliverables that you will need to submit:

<ul>

 <li>Your code. This should implement the skeleton files that we provide. It should be written in Python 3.</li>

 <li>Your model’s output for the test set using only the provided training and development data.</li>

 <li>(Optional) your model’s output for the test set, using any data that you want.</li>

 <li>Your writeup in the form of a PDF.</li>