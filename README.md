# DeepLearningDNA
1D ConvNet for DNA Classification (implementation of Nguyen's 2016 paper)

### Introduction:

A Keras implementation of the paper: "DNA Sequence Classification by Convolutional Neural Network" by Nguyen, et al. (2016)

The DNA sequence classifier is tested using the UCI Splice dataset. Requires the splice dataset, available here:
https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/splice-junction-gene-sequences/

The target variable is categorical: EI, IE, or N (exon-intron junction, intron-exon junction, or neither). Predictors are Primate DNA sequences of 60 nucleotides, 30 preceding the junction, and 30 after the junction.

### Files:

cnn_classify_crossval.py: Performs 10-fold cross-validation on the classifier.\
cnn_classify.py: Demonstrates classifier performance on a single 80-20 Train/Test split of the dataset.

### Example Outputs: 

#### cnn_classify_crossval.py 

<pre>
Loading Dataset...
Processing Sequences...
Generating Models...
Fold 1 - Validation Accuracy: 0.9594
Fold 2 - Validation Accuracy: 0.9563
Fold 3 - Validation Accuracy: 0.9563
Fold 4 - Validation Accuracy: 0.9344
Fold 5 - Validation Accuracy: 0.9844
Fold 6 - Validation Accuracy: 0.9467
Fold 7 - Validation Accuracy: 0.9561
Fold 8 - Validation Accuracy: 0.9717
Fold 9 - Validation Accuracy: 0.9590
Fold 10 - Validation Accuracy: 0.9558

mean accuracy (10 folds): 0.9580
</pre>

#### cnn_classify.py

<pre>
Loading Dataset...
Processing Sequences...
Generating Model...
Test Set Accuracy: 0.9608
Test Set F1 Score: 0.9608

Test Set Confusion Matrix:
     EI   IE    N
EI  125    1    4
IE    3  142    6
N     4    7  346
</pre>
