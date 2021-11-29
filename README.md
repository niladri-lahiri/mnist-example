# Line chart that has on x-axis the percentage of training and y-axis test set macro f1
![alt text](https://github.com/niladri-lahiri/mnist-example/blob/feature/assignment_11/images/line_plot.png)

We can see that with the increase in the training percentage, the macro f1-score also increases drastically. Just with very little increase in the training percentage, we can see a steep spike in the macro f1 score and from there on, the as we increase the training percentage, the difference isnt too much.

# Compare the actual predictions on test set using the model trained with  20% training data vs 10% training data, and 30% vs 20% so-on.
![alt text](https://github.com/niladri-lahiri/mnist-example/blob/feature/assignment_11/images/roc_plot.png)

We can see that as we increase the training data, the change in ROC score is drastic in the initial changes, but as we keep on changing the difference in ROC score subsides and is almost close to null.
