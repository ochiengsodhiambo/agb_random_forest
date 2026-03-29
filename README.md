## AGB Prediction Using Random Forest

This project applies a Random Forest regression model to estimate **Above Ground Biomass (AGB)** using vegetation indices (**NDVI and EVI**). It demonstrates the use of data science and machine learning for environmental and ecological analysis.

---

### Project Overview
- Built a predictive model using **scikit-learn**
- Processed and cleaned environmental data using **pandas**
- Evaluated model performance using **R² and Mean Squared Error**
- Visualized model predictions vs actual values

---

### Data
The dataset contains:
- **NDVI (Normalized Difference Vegetation Index)**
- **EVI (Enhanced Vegetation Index)**
- **AGB (Above Ground Biomass)**

---

### Methodology
1. Data cleaning (handling missing values)
2. Feature selection (NDVI, EVI)
3. Train-test split (80/20)
4. Model training using Random Forest Regressor
5. Model evaluation (R², MSE)
6. Visualization of predictions

---

### Results
- R² Score: **
- MSE: **

The model shows the effectiveness of vegetation indices in estimating biomass.

---

### How to Run

```bash
pip install -r requirements.txt
python agb_random_forest.py

