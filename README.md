# 🎓 AI-Driven Career Success Predictor

**Optimizing Student Employability in an AI-Transformed Workforce**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://amenauril78-career-success-predictor-app-vnwdxj.streamlit.app)

---

## 📋 Project Overview

An end-to-end machine learning system that predicts student placement outcomes and provides personalized career recommendations. This project demonstrates comprehensive model exploration, rigorous hyperparameter tuning, and production deployment.

**Course:** CIS 508 - Machine Learning in Business  
**Institution:** Arizona State University  
**Semester:** Fall 2025

---

## 🎯 Business Problem

**Challenge:**
- 32% of MBA graduates remain unemployed after graduation
- Students lack personalized career guidance
- Career services cannot identify at-risk students early
- AI is transforming job markets faster than curricula adapt

**Solution:**
- Predict placement probability with 86% accuracy
- Identify specific risk factors for each student
- Provide data-driven, actionable recommendations
- Enable early intervention for at-risk students

---

## 🚀 Live Application

**🌐 [Try the Career Success Predictor](https://amenauril78-career-success-predictor-app-vnwdxj.streamlit.app)**

### Features:
- ✅ Real-time placement probability predictions
- ✅ Personalized risk factor analysis
- ✅ Actionable career recommendations
- ✅ Skill development priorities
- ✅ Interactive, user-friendly interface

---

## 🤖 Model Development

### Comprehensive Model Exploration

This project involved systematic comparison of **7 classification algorithms** with extensive hyperparameter tuning:

| Rank | Model | F1-Score | AUC-ROC | Accuracy | Status |
|------|-------|----------|---------|----------|--------|
| 1 | **Random Forest** | **0.9000** 🥇 | **0.9187** | **86.05%** | **🚀 Deployed** |
| 2 | Logistic Regression | 0.8966 | 0.9282 | 88.37% | Evaluated |
| 3 | Naive Bayes | 0.8966 | 0.9231 | 88.37% | Evaluated |
| 4 | SVM | 0.8772 | 0.9256 | 86.05% | Evaluated |
| 5 | XGBoost | 0.8667 | **0.9410** 🏆 | 86.05% | Evaluated |
| 6 | Neural Network | 0.8387 | 0.8179 | 83.72% | Evaluated |
| 7 | k-NN | 0.8214 | 0.8744 | 81.40% | Evaluated |

Random Forest figures are from the saved deployment package (`models/deployment_package_rf.pkl`).

### Model Selection Rationale

**Random Forest** was selected for production deployment. It has the highest F1-score (0.9000), the primary optimization metric, and:

1. **Interpretability:** Provides transparent feature importance for user trust and explainability
2. **Balanced Performance:** Strong metrics across all dimensions (F1: 0.9000, AUC: 0.9187, recall: 93.1%)
3. **Production Stability:** Robust to outliers and handles non-linear relationships effectively
4. **Close Competition:** Logistic Regression and Naive Bayes are within 0.004 F1; with a 43-student test set, the top three models perform about the same

### Hyperparameter Optimization

All models underwent systematic tuning:
- **GridSearchCV** for Logistic Regression, k-NN, Naive Bayes, SVM
- **RandomizedSearchCV** for Random Forest, XGBoost, Neural Network (larger parameter spaces)
- **3-fold cross-validation** throughout
- **F1-score** as primary optimization metric (handles class imbalance)
- **SMOTE oversampling** for balanced training data

---

## 📊 Dataset & Features

**Source:** Campus Placement Data (Kaggle)  
**Size:** 215 student records  
**Features:** 24 total (13 original + 11 engineered)  
**Target:** Binary classification (Placed / Not Placed)  
**Split:** 60% train / 20% validation / 20% test

### Feature Engineering

Created 11 advanced features:
- **Academic Metrics:** Average, trend, consistency across all education levels
- **Career Readiness Score:** Weighted composite of test scores, grades, and experience
- **Performance Indicators:** Strong test performer, declining performance flags
- **Experience Markers:** Work experience binary encoding
- **Risk Flags:** Low test score, weak degree performance indicators

### Key Insights

**Top Predictive Factors:**
1. **Work Experience** (40% weight) - Students WITH experience: 85% placement rate vs 45% without
2. **Employability Test Score** (30% weight) - Scores >70% strongly predict success
3. **MBA Performance** (30% weight) - Recent academic achievement matters
4. **Academic Consistency** - Stable performance valued over sporadic excellence
5. **Career Readiness Composite** - Holistic success indicator

---

## 💼 Potential Business Impact (Illustrative Estimate)

> These figures are a hypothetical scenario, not measured results. They assume a 15-point placement lift and $25,000 of value per placement; neither has been tested.

### Illustrative ROI Scenario (500 graduates/year)

**Current State:**
- 68% placement rate (340 placed, 160 unemployed)
- Lost alumni engagement value: ~$2M/year

**With AI System (assumed):**
- Assumed 15-point improvement in placement rate
- New rate: 83% (415 placed, 85 unemployed)
- **75 additional successful placements**

**Financial Impact:**
| Metric | Value |
|--------|-------|
| Additional Placements | 75 students |
| Value per Placement | $25,000 |
| Annual Benefit | $1,875,000 |
| System Cost (Year 1) | $50,000 |
| Net Benefit | $1,825,000 |
| **ROI** | **3,650%** |

**Beyond ROI:**
- Improved student satisfaction and outcomes
- Enhanced university reputation
- Stronger alumni networks
- Data-driven resource allocation

---

## 🛠️ Technical Stack

### Development
- **Language:** Python 3.9+
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn
- **Environment:** Google Colab
- **Version Control:** Git, GitHub

### Machine Learning
- **Framework:** scikit-learn
- **Algorithms:** Logistic Regression, k-NN, Naive Bayes, SVM, Random Forest, XGBoost, Neural Network
- **Experiment Tracking:** MLflow (Databricks)
- **Model Selection:** Random Forest Classifier

### Deployment
- **Framework:** Streamlit
- **Hosting:** Streamlit Cloud
- **CI/CD:** Automated deployment from GitHub

---

## 📁 Project Structure
```
career-success-predictor/
├── models/
│   ├── career_predictor_model_rf.pkl      # Trained Random Forest model
│   ├── preprocessing_objects.pkl          # Feature encoders & scalers
│   └── deployment_package_rf.pkl          # Complete deployment package
├── app.py                                 # Streamlit web application
├── requirements.txt                       # Python dependencies
├── README.md                              # Project documentation
└── .gitignore                             # Git ignore file
```
---

## 📈 Model Performance

### Test Set Results (Unseen Data)

**Random Forest Classifier:**
- **Accuracy:** 86.05%
- **Precision:** 87.10%
- **Recall:** 93.10%
- **F1-Score:** 0.9000
- **AUC-ROC:** 0.9187 ⭐ (Excellent)

**Confusion Matrix:**
| **Actual** | **Predicted Not Placed** | **Predicted Placed** |
| :---: | :---: | :---: |
| Not Placed | 10 | 4 |
| Placed | 2 | 27 |

**Interpretation:**
- 37 correct predictions out of 43 (86% accuracy)
- Only 2 false negatives (missed placements)
- 4 false positives (incorrect placement predictions)

---

## 🔬 Experiment Tracking

All experiments logged in **MLflow (Databricks)** with complete tracking:
- ✅ All model parameters and hyperparameters
- ✅ Performance metrics (accuracy, precision, recall, F1, AUC-ROC)
- ✅ Confusion matrices and ROC curves
- ✅ Model artifacts for reproducibility
- ✅ Full experiment lineage

**MLflow Experiment:** `Career_Success_Predictor_Final`  
**Total Runs:** 40+ (including hyperparameter search iterations)

---

## 🎓 Key Learnings

1. **Comprehensive Model Exploration:** Testing multiple algorithms reveals optimal solutions
2. **Feature Engineering Impact:** Engineered features significantly improved model performance
3. **Production Considerations:** Model selection requires balancing performance with interpretability
4. **Business Value:** ML systems must deliver measurable ROI beyond technical metrics
5. **End-to-End Pipeline:** Real value comes from deployed, accessible solutions

---

## 🔮 Future Enhancements

### Short-term (3-6 months)
- Real-time model monitoring and drift detection
- A/B testing framework for model improvements
- Integration with university CRM systems
- Mobile application (iOS/Android)

### Long-term (6-12 months)
- Multi-class prediction (specific job roles/industries)
- Salary prediction module
- Extended feature set (soft skills, extracurriculars)
- Alumni outcome tracking
- Automated model retraining pipeline

### Research Opportunities
- AI impact on career trajectories
- Skill gap analysis at scale
- Intervention effectiveness studies
- Long-term placement outcome tracking

---

## 📄 License

This project is for educational purposes as part of CIS 508 coursework at Arizona State University.

---

## 👤 Author

**Amena Uril**  
Master's Student - Business Analytics  
Arizona State University  
📧 auril@asu.edu  
🔗 [GitHub](https://github.com/AmenaUril78))  
🔗 [LinkedIn][(https://www.linkedin.com/in/amena-uril/)]

---

## 🙏 Acknowledgments

- **Professor:** Sang Pil Han - CIS 508: Machine Learning in Business
- **Institution:** Arizona State University
- **Dataset:** Campus Placement Data (Kaggle)
- **Tools:** Streamlit, scikit-learn, MLflow (Databricks)

---

## 📞 Contact & Support

**Live Application:** [Career Success Predictor](https://amenauril78-career-success-predictor-app-vnwdxj.streamlit.app)  
**GitHub Repository:** [career-success-predictor](https://github.com/AmenaUril78/career-success-predictor)    
**Email:** auril@asu.edu

---

<div align="center">

**⚠️ Disclaimer**

This tool is for educational and research purposes. Final placement decisions should consider multiple factors beyond model predictions. The model provides probability-based guidance, not definitive outcomes.

---

**Made with ❤️ by Amena Uril | Arizona State University | Fall 2025**

</div>
