XGBoost (seed=42, 5-fold, scoring=average_precision)
Test ROC-AUC: 0.9750
Test PR-AUC: 0.8820
Threshold=0.5: F1=0.8586, Precision=0.8817, Recall=0.8367; Confusion [[56853,11],[16,82]]
Best-F1 Thr ≈ 0.9728: F1=0.8852, Precision=0.9529, Recall=0.8265; Confusion [[56860,4],[17,81]]
Best params: {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 600, 'scale_pos_weight': ~577.29, 'subsample': 0.8}
Logistic Regression (seed=42, 5-fold)

Test ROC-AUC: 0.9720
Test PR-AUC: 0.7189
Threshold=0.5: F1=0.1144, Precision=0.0610, Recall=0.9184; Confusion [[55479,1385],[8,90]]
Best-F1 Thr ≈ 1.0000: F1=0.8247, Precision=0.8333, Recall=0.8163; Confusion [[56848,16],[18,80]]
Best params: {'C': 0.5}
