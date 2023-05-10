we take best performing model of mara ( the one of fold 0) and we retrain it on marta data and compare it to the result of best marta model (fold 1). in the pickle file only the accuracy of fold 1 is accurate since in the other four the model was trained on the fold of of the test, hence the high accuracies.

In this case, the weight of the first layer are frozen and only the last layer is retrained
