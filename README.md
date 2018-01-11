A spam classifier based on machine learning concepts, It uses Artificial neural network model.
It preprocess data by Lower-casing, Stripping HTML, Normalizing URLs, Normalizing Email Addresses, Normalizing Numbers, Word Stemming, Removal of non-words.
Trained a classifier to classify whether a given email, x, is spam (y = 1) or non-spam (y = 0). In particular, it converted each email into a feature vector x ∈ R n . Then developed a predefined Vocabulary list which matches indices with the spam mail and stores the indices in a vector and makes the feature x i ∈ {0, 1} for an email corresponds to whether the i-th word in the dictionary occurs in the email. That is, x i = 1 if the i-th word is in the email and x i = 0 if the i-th word is not present in the email.
The neural network used here was a 5 layer neural network (deep neural network) with the activation functions such as RELU for first four layer and sigmoid for the last layer.
After loading the dataset, it will proceed to train our neural network to classify between spam (y = 1) and non-spam (y = 0) emails. Once the training completes, saw that the classifier gets a training accuracy of about 90.8% and a test accuracy of about 88.5%.
further hyperparameter tuning can stretch the accuracy to more than 95%.
