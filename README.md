# OWL Factory Hackathon
### **Authors:** Karl Johannes, Christian Wergers, Philipp Ohm, Eric Thor

**Link to Hackathon Website:** https://smartfactory-owl.de/eventer/ai-challenge-days-2023/

### **Problem:** 
Predict the temperature at position 6 of the chocolate melting and cooling machine to replace the current sensor with a predictive model to reduce costs.

### **Our Solution:** 
After an exploratory data analysis (EDA), we decided to train two separate regression models. One day and one night cycle model. In addition to the day/night cycle split, we added hours and minutes as features. The use of a standard scaler was also very important since the sensor data scales are very different. We then performed PCA, where the number of principal components was chosen based on the PVE/CPVE plot, as 3 components already explained <98% of the variance. The hackathon was evaluated based on the mean absolute error (MAE). We decided to use an LSTM for this regression task because it gave the best MAE results.

### **Our obtained scores (sensor 6 temperature deviation)**:
- Mean Absolute Errors (MAE):
    - Day MAE: 0.071-0.105
    - Night MAE: 0.01
    - Overall MAE: 0.041-0.058

### Used Technologies:
- Python, Pytorch 
### Used Preprocessing Techniques:
- PCA, MDS, StandardScaler, PVE/CPVE
### Used/Tested Models:
- Ridge Regression, Lasso Regression, XGBoost, Support Vector Regression, Long short-term memory NNs (LSTM), inear Regression


### How we evaluated our models
- Model Evaluation Dataset Splits: (week 1-4 excl. testset/week 5)
- **Full Data**
    - Train on full data and evaluate on Testset
- **Holdout Split 1**
    - Train on (Tuesdays, Wednesdays) | Evaluate on (Thursdays)
- **Holdout Split 2**
    - Train on (Wednesdays, Thursdays) | Evaluate on (Tuesdays)
- **Holdout Split 3**
    - Train on (Tuesdays, Thursdays) | Evaluate on (Wednesdays)
- **Single Holdout 1**
    - Train on (Tuesdays) | Evaluate on (Wednesdays, Thursdays)
- **Single Holdout 2**
    - Train on (Wednesdays) | Evaluate on (Tuesdays, Thursdays)
- **Single Holdout 3**
    - Train on (Thursdays) | Evaluate on (Tuesdays, Wednesdays)
- **Testset (DO NOT USE)**
    - Week 5 Data from tuesday, wednesday, thursday