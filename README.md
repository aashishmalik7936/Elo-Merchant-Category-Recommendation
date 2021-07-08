# ELO MERCHANT CATEGORY RECOMMENDATION COMPETITION #
This was a machine learning competition hosted by **Kaggle**. In this participants had to make a regression model that will predict the loyality scores given for each transactions. Link to competition: https://www.kaggle.com/c/elo-merchant-category-recommendation

## About Elo ##
Elo is one of the largest payment brands in Brazil, has built partnerships with merchants in order to offer promotions or discounts to cardholders. Elo asked the kagglers to build a regression model which can predict the loyality scores given to custormers for that particular transaction, given the details about the transaction. 

## Performance ##
This was one of the largest competition hosted on kaggle and I secured **140th position (top 4%) out of 4110 participants thereby winning a silver medal**. Link to my kaggle profile: https://www.kaggle.com/aashish7936

## Libraries of python ##
Numpy, Pandas, Seaborn, Matplotlib, scikit-learn, lightgbm, xgboost, datetime, gc.

## Approach ##
Initially, I did some exploratory data analysis to become familiar with the feature that are present in the dataset. The dataset contained
Historical transactions data, new transactions data, merchant data, train data(contained loyality scores), test data. While doing EDA I found that the target column contains **about 1.5 percent of the outliers** and the competition metric was **root mean squared error** which means one wrong prediction can cost a lot.

Then I did preprocessing like filling continuous null values with mean & categorical null values with mode and did categorical encoding. In preprocessing, I made a columns which **classified the outlier target from non outliers**.

Now, I made a **5 fold stratified fold cross validation in which each fold contained same ratio of outlier and non outlier targets** and then trained the dataset on lightgbm.

For hyperparameter tuning, I used **optuna** to get best set of hyperparamters.

I mostly made **aggregated features** with the **Historical and new transactions of the costumers and merchants**. Most impactful features were those which were derived either from **purchase amount** or from the **difference in the purchase dates**. For checking usefulness of features, I was **recording the CV score after the adding the new feature and then I was shuffling that feature and then again checking the CV score, if the shuffling CV score is better than the normal CV score then the feature not useful otherwise it is useful**. With this technique I was able to generate **108 useful features**.

Now comes the post processing part, For I basically made three models:
#First model was made on both outlier targets as well as non outlier targets.
#Second model was made on non outliers targets.
#Third model was made to predict the the outliers in the testset.

**I replaced the second model predicted values with first model values which were predicted as outliers by the third model**. This technique gave a lot of boost to my model.

I also made a **xgboost model** with similar technique as mentioned above. Xgboost model had a total of **73 features** and used **10 fold CV instead of 5 fold**.

Finally, I took **the average of both the models as the final submission**.

## Things that I didn't tried ##
I didn't go for stacking because of less time. I think more models could have been made and then use their predictions as features for a **meta model** to predict the final values.

I didn't tried neural network which could be done.
