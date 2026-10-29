# Ensemble Learning, Bagging & Boosting — Viva Questions

## Basic Questions

**1. What is Ensemble Learning?**  
Combining multiple models to make a better prediction.

**2. Why do we use Ensemble Learning?**  
To improve accuracy and make predictions more reliable.

**3. What is a base learner?**  
An individual model used in an ensemble.

**4. What are the two common ensemble techniques?**  
Bagging and Boosting.

## Bagging

**5. What is Bagging?**  
Bagging trains multiple models independently on different samples of the data.

**6. What does Bagging stand for?**  
Bootstrap Aggregating.

**7. How are models trained in Bagging?**  
In parallel, independently of one another.

**8. How is the final prediction made in Bagging?**  
Majority voting for classification; averaging for regression.

**9. What is Bootstrap Sampling?**  
Randomly selecting data rows with replacement.

**10. Give an example of Bagging.**  
Random Forest.

**11. What is Random Forest?**  
A collection of decision trees whose predictions are combined.

**12. What is feature sampling in Random Forest?**  
Each split considers a random subset of features.

**13. What problem does Bagging mainly reduce?**  
Variance.

## Boosting

**14. What is Boosting?**  
Boosting trains models one after another, with later models focusing on earlier errors.

**15. How are models trained in Boosting?**  
Sequentially.

**16. How is Boosting different from Bagging?**  
Bagging trains models independently; Boosting learns from previous errors.

**17. What is AdaBoost?**  
A boosting method that gives more weight to misclassified data points.

**18. What is a Decision Stump?**  
A decision tree with only one split (depth 1).

**19. What happens to wrongly classified points in AdaBoost?**  
Their weights are increased so later models focus more on them.

**20. How is the final prediction made in Boosting?**  
By combining model predictions, often using weighted votes or sums.

**21. What is Gradient Boosting?**  
A method that adds models to reduce the errors (residuals) of earlier predictions.

**22. What does Boosting mainly try to reduce?**  
Bias, though the effect depends on the model and settings.

## Quick Comparison

| Bagging | Boosting |
|---|---|
| Models train in parallel | Models train sequentially |
| Models are independent | Later models focus on earlier errors |
| Uses voting or averaging | Often uses weighted combination |
| Mainly reduces variance | Often reduces bias |
| Example: Random Forest | Examples: AdaBoost, Gradient Boosting |

## Last-Minute Memory Trick

- **Ensemble** = many models together
- **Bagging** = parallel + independent
- **Boosting** = sequential + correct errors
- **Random Forest** = many decision trees
- **AdaBoost** = increase weight of misclassified points
- **Gradient Boosting** = learn from residuals
