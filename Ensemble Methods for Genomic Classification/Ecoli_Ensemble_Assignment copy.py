{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "1c6bd4dd",
   "metadata": {},
   "source": [
    "\n",
    "Course: Machine Learning\n",
    "\n",
    "Assignment-2: Comparative Analysis of Random Forest and XGBoost using the Ecoli Dataset\n",
    "\n",
    "Name : Aschalew Zerihun\n",
    "\n",
    "Date : May 25, 2026\n",
    "\n",
    "Student ID: GSR/8706/18"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "016d92a7",
   "metadata": {},
   "source": [
    "\n",
    "# Ensemble Methods for Genomic Classification\n",
    "## Comparative Analysis of Random Forest and XGBoost using the Ecoli Dataset\n",
    "\n",
    "This notebook provides a complete solution for the assignment on ensemble learning methods in bioinformatics.\n",
    "\n",
    "The notebook includes:\n",
    "\n",
    "1. Exploratory Data Analysis (EDA)\n",
    "2. Data preprocessing\n",
    "3. Random Forest implementation\n",
    "4. XGBoost implementation\n",
    "5. Comparative evaluation\n",
    "6. Biological interpretation of feature importance\n",
    "\n",
    "Dataset:\n",
    "- `ecoli.data`\n",
    "- `ecoli.names`\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "988f69b6",
   "metadata": {},
   "source": [
    "\n",
    "# Import Required Libraries\n",
    "\n",
    "The following libraries are used for:\n",
    "- Data manipulation\n",
    "- Visualization\n",
    "- Machine learning\n",
    "- Model evaluation\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "f1b1118d",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from sklearn.model_selection import train_test_split, GridSearchCV\n",
    "from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import (\n",
    "    confusion_matrix,\n",
    "    classification_report,\n",
    "    f1_score,\n",
    "    accuracy_score\n",
    ")\n",
    "\n",
    "from xgboost import XGBClassifier\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Display plots inside notebook\n",
    "%matplotlib inline\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6c5a1ced",
   "metadata": {},
   "source": [
    "\n",
    "# Load the Ecoli Dataset\n",
    "\n",
    "The Ecoli dataset contains protein localization information for *Escherichia coli* proteins.\n",
    "\n",
    "Features:\n",
    "- mcg\n",
    "- gvh\n",
    "- lip\n",
    "- chg\n",
    "- aac\n",
    "- alm1\n",
    "- alm2\n",
    "\n",
    "Target:\n",
    "- Protein localization class\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "a2325208",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sequence_name</th>\n",
       "      <th>mcg</th>\n",
       "      <th>gvh</th>\n",
       "      <th>lip</th>\n",
       "      <th>chg</th>\n",
       "      <th>aac</th>\n",
       "      <th>alm1</th>\n",
       "      <th>alm2</th>\n",
       "      <th>class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>AAT_ECOLI</td>\n",
       "      <td>0.49</td>\n",
       "      <td>0.29</td>\n",
       "      <td>0.48</td>\n",
       "      <td>0.5</td>\n",
       "      <td>0.56</td>\n",
       "      <td>0.24</td>\n",
       "      <td>0.35</td>\n",
       "      <td>cp</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ACEA_ECOLI</td>\n",
       "      <td>0.07</td>\n",
       "      <td>0.40</td>\n",
       "      <td>0.48</td>\n",
       "      <td>0.5</td>\n",
       "      <td>0.54</td>\n",
       "      <td>0.35</td>\n",
       "      <td>0.44</td>\n",
       "      <td>cp</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>ACEK_ECOLI</td>\n",
       "      <td>0.56</td>\n",
       "      <td>0.40</td>\n",
       "      <td>0.48</td>\n",
       "      <td>0.5</td>\n",
       "      <td>0.49</td>\n",
       "      <td>0.37</td>\n",
       "      <td>0.46</td>\n",
       "      <td>cp</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>ACKA_ECOLI</td>\n",
       "      <td>0.59</td>\n",
       "      <td>0.49</td>\n",
       "      <td>0.48</td>\n",
       "      <td>0.5</td>\n",
       "      <td>0.52</td>\n",
       "      <td>0.45</td>\n",
       "      <td>0.36</td>\n",
       "      <td>cp</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ADI_ECOLI</td>\n",
       "      <td>0.23</td>\n",
       "      <td>0.32</td>\n",
       "      <td>0.48</td>\n",
       "      <td>0.5</td>\n",
       "      <td>0.55</td>\n",
       "      <td>0.25</td>\n",
       "      <td>0.35</td>\n",
       "      <td>cp</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  sequence_name   mcg   gvh   lip  chg   aac  alm1  alm2 class\n",
       "0     AAT_ECOLI  0.49  0.29  0.48  0.5  0.56  0.24  0.35    cp\n",
       "1    ACEA_ECOLI  0.07  0.40  0.48  0.5  0.54  0.35  0.44    cp\n",
       "2    ACEK_ECOLI  0.56  0.40  0.48  0.5  0.49  0.37  0.46    cp\n",
       "3    ACKA_ECOLI  0.59  0.49  0.48  0.5  0.52  0.45  0.36    cp\n",
       "4     ADI_ECOLI  0.23  0.32  0.48  0.5  0.55  0.25  0.35    cp"
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "# Define column names\n",
    "columns = [\n",
    "    'sequence_name',\n",
    "    'mcg',\n",
    "    'gvh',\n",
    "    'lip',\n",
    "    'chg',\n",
    "    'aac',\n",
    "    'alm1',\n",
    "    'alm2',\n",
    "    'class'\n",
    "]\n",
    "\n",
    "# Load dataset\n",
    "df = pd.read_csv(\n",
    "    'ecoli.data',\n",
    "    delim_whitespace=True,\n",
    "    header=None,\n",
    "    names=columns\n",
    ")\n",
    "\n",
    "# Display first rows\n",
    "df.head()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "90131e29",
   "metadata": {},
   "source": [
    "\n",
    "# Exploratory Data Analysis (EDA)\n",
    "\n",
    "We first inspect the dataset structure and class distribution.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "dad579ee",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dataset Shape: (336, 9)\n",
      "\n",
      "Missing Values:\n",
      "sequence_name    0\n",
      "mcg              0\n",
      "gvh              0\n",
      "lip              0\n",
      "chg              0\n",
      "aac              0\n",
      "alm1             0\n",
      "alm2             0\n",
      "class            0\n",
      "dtype: int64\n",
      "\n",
      "Class Distribution:\n",
      "class\n",
      "cp     143\n",
      "im      77\n",
      "pp      52\n",
      "imU     35\n",
      "om      20\n",
      "omL      5\n",
      "imS      2\n",
      "imL      2\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# Dataset information\n",
    "print(\"Dataset Shape:\", df.shape)\n",
    "\n",
    "print(\"\\nMissing Values:\")\n",
    "print(df.isnull().sum())\n",
    "\n",
    "print(\"\\nClass Distribution:\")\n",
    "print(df['class'].value_counts())\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "82e5c2ec",
   "metadata": {},
   "source": [
    "\n",
    "## Class Distribution Visualization\n",
    "\n",
    "This plot helps identify whether the dataset is balanced or imbalanced.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "afc467b0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": 
      "text/plain": [
       "<Figure size 1000x800 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "# Remove sequence name and class\n",
    "features = df.drop(columns=['sequence_name', 'class'])\n",
    "\n",
    "plt.figure(figsize=(10,8))\n",
    "\n",
    "sns.heatmap(\n",
    "    features.corr(),\n",
    "    annot=True,\n",
    "    cmap='coolwarm'\n",
    ")\n",
    "\n",
    "plt.title('Correlation Heatmap')\n",
    "\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e503ca04",
   "metadata": {},
   "source": [
    "\n",
    "## Boxplots Across Localization Classes\n",
    "\n",
    "Boxplots help compare feature distributions across different protein localization sites.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "id": "d64bca86",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": 
      "text/plain": [
       "<Figure size 1000x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": 
      "text/plain": [
       "<Figure size 1000x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "selected_features = ['mcg', 'gvh', 'aac', 'alm2', 'lip']\n",
    "\n",
    "for feature in selected_features:\n",
    "    \n",
    "    plt.figure(figsize=(10,5))\n",
    "    \n",
    "    sns.boxplot(\n",
    "        x='class',\n",
    "        y=feature,\n",
    "        data=df\n",
    "    )\n",
    "    \n",
    "    plt.title(f'Boxplot of {feature} Across Classes')\n",
    "    \n",
    "    plt.xticks(rotation=45)\n",
    "    \n",
    "    plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "09ef2755",
   "metadata": {},
   "source": [
    "\n",
    "# Data Preprocessing\n",
    "\n",
    "We perform:\n",
    "1. Feature scaling\n",
    "2. Label encoding\n",
    "3. Train-test split\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "id": "dc328ca3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training Shape: (268, 7)\n",
      "Testing Shape: (68, 7)\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# Prepare features and labels\n",
    "X = df.drop(columns=['sequence_name', 'class'])\n",
    "y = df['class']\n",
    "\n",
    "# Encode target labels\n",
    "label_encoder = LabelEncoder()\n",
    "y_encoded = label_encoder.fit_transform(y)\n",
    "\n",
    "# Standardization\n",
    "standard_scaler = StandardScaler()\n",
    "X_standardized = standard_scaler.fit_transform(X)\n",
    "\n",
    "# Min-Max normalization\n",
    "minmax_scaler = MinMaxScaler()\n",
    "X_normalized = minmax_scaler.fit_transform(X)\n",
    "\n",
    "# Train-test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X_standardized,\n",
    "    y_encoded,\n",
    "    test_size=0.2,\n",
    "    stratify=y_encoded,\n",
    "    random_state=42\n",
    ")\n",
    "\n",
    "print(\"Training Shape:\", X_train.shape)\n",
    "print(\"Testing Shape:\", X_test.shape)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0210703c",
   "metadata": {},
   "source": [
    "\n",
    "# Random Forest Classification\n",
    "\n",
    "Random Forest is a bagging-based ensemble learning algorithm.\n",
    "\n",
    "Key ideas:\n",
    "- Bootstrap sampling\n",
    "- Multiple decision trees\n",
    "- Random feature selection\n",
    "- Majority voting\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "faa79563",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best Parameters: {'max_depth': 5, 'max_features': 'sqrt', 'n_estimators': 100}\n",
      "OOB Score: 0.8544776119402985\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# Define parameter grid\n",
    "rf_params = {\n",
    "    'n_estimators': [50, 100],\n",
    "    'max_depth': [5, 10, None],\n",
    "    'max_features': ['sqrt', 'log2']\n",
    "}\n",
    "\n",
    "# Initialize Random Forest\n",
    "rf_model = RandomForestClassifier(\n",
    "    random_state=42,\n",
    "    oob_score=True\n",
    ")\n",
    "\n",
    "# Grid Search\n",
    "rf_grid = GridSearchCV(\n",
    "    rf_model,\n",
    "    rf_params,\n",
    "    cv=5,\n",
    "    scoring='f1_weighted'\n",
    ")\n",
    "\n",
    "# Train model\n",
    "rf_grid.fit(X_train, y_train)\n",
    "\n",
    "# Best model\n",
    "best_rf = rf_grid.best_estimator_\n",
    "\n",
    "print(\"Best Parameters:\", rf_grid.best_params_)\n",
    "print(\"OOB Score:\", best_rf.oob_score_)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "78205f10",
   "metadata": {},
   "source": [
    "\n",
    "## Random Forest Evaluation\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "ee4e82e4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest Accuracy: 0.8676470588235294\n",
      "Random Forest F1 Score: 0.8714426270061993\n",
      "\n",
      "Classification Report\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.97      1.00      0.98        29\n",
      "           1       0.83      0.62      0.71        16\n",
      "           4       0.50      0.86      0.63         7\n",
      "           5       1.00      1.00      1.00         4\n",
      "           6       1.00      1.00      1.00         1\n",
      "           7       1.00      0.82      0.90        11\n",
      "\n",
      "    accuracy                           0.87        68\n",
      "   macro avg       0.88      0.88      0.87        68\n",
      "weighted avg       0.90      0.87      0.87        68\n",
      "\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# Predictions\n",
    "rf_predictions = best_rf.predict(X_test)\n",
    "\n",
    "# Accuracy\n",
    "rf_accuracy = accuracy_score(y_test, rf_predictions)\n",
    "\n",
    "# F1 Score\n",
    "rf_f1 = f1_score(\n",
    "    y_test,\n",
    "    rf_predictions,\n",
    "    average='weighted'\n",
    ")\n",
    "\n",
    "print(\"Random Forest Accuracy:\", rf_accuracy)\n",
    "print(\"Random Forest F1 Score:\", rf_f1)\n",
    "\n",
    "# Classification Report\n",
    "print(\"\\nClassification Report\")\n",
    "print(\n",
    "    classification_report(\n",
    "        y_test,\n",
    "        rf_predictions,\n",
    "        # target_names=label_encoder.classes_\n",
    "    )\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "40ed1ad5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0 7 4 0 7 1 4 1 4 4 0 4 0 1 4 7 0 7 6 1 0 0 1 5 0 0 0 4 1 4 0 0 0 4 0 0 0\n",
      " 0 0 7 0 0 1 0 5 7 7 1 0 5 4 4 1 1 0 7 1 0 7 0 0 1 0 5 0 4 0 0]\n",
      "Prediction Class Distribution: Counter({np.int64(0): 30, np.int64(4): 12, np.int64(1): 12, np.int64(7): 9, np.int64(5): 4, np.int64(6): 1})\n",
      "6\n"
     ]
    }
   ],
   "source": [
    "print(rf_predictions)\n",
    "\n",
    "from collections import Counter\n",
    "print(\"Prediction Class Distribution:\", Counter(rf_predictions))\n",
    "\n",
    "print(len(Counter(rf_predictions)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "25632a17",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Counter({np.int64(0): 29, np.int64(1): 16, np.int64(7): 11, np.int64(4): 7, np.int64(5): 4, np.int64(6): 1})\n"
     ]
    }
   ],
   "source": [
    "print(Counter(y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "481b41fb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['cp' 'im' 'imL' 'imS' 'imU' 'om' 'omL' 'pp']\n"
     ]
    }
   ],
   "source": [
    "print(label_encoder.classes_) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "a02f63f9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": 
      "text/plain": [
       "<Figure size 800x600 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "# Confusion Matrix\n",
    "rf_cm = confusion_matrix(y_test, rf_predictions)\n",
    "\n",
    "plt.figure(figsize=(8,6))\n",
    "\n",
    "sns.heatmap(\n",
    "    rf_cm,\n",
    "    annot=True,\n",
    "    fmt='d',\n",
    "    cmap='Blues'\n",
    ")\n",
    "\n",
    "plt.title('Random Forest Confusion Matrix')\n",
    "plt.xlabel('Predicted')\n",
    "plt.ylabel('Actual')\n",
    "\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "33e75b46",
   "metadata": {},
   "source": [
    "\n",
    "# XGBoost Classification\n",
    "\n",
    "XGBoost is a gradient boosting algorithm.\n",
    "\n",
    "Key ideas:\n",
    "- Sequential learning\n",
    "- Error correction\n",
    "- Gradient optimization\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "8ca2abfc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best Parameters: {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3}\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# Initialize XGBoost model\n",
    "xgb_model = XGBClassifier(\n",
    "    objective='multi:softmax',\n",
    "    num_class=len(np.unique(y_encoded)),\n",
    "    eval_metric='mlogloss',\n",
    "    random_state=42\n",
    ")\n",
    "\n",
    "# Parameter grid\n",
    "xgb_params = {\n",
    "    'learning_rate': [0.01, 0.1],\n",
    "    'max_depth': [3, 5],\n",
    "    'gamma': [0, 1]\n",
    "}\n",
    "\n",
    "# Grid Search\n",
    "xgb_grid = GridSearchCV(\n",
    "    xgb_model,\n",
    "    xgb_params,\n",
    "    cv=3,\n",
    "    scoring='f1_weighted'\n",
    ")\n",
    "\n",
    "# Train model\n",
    "xgb_grid.fit(X_train, y_train)\n",
    "\n",
    "# Best model\n",
    "best_xgb = xgb_grid.best_estimator_\n",
    "\n",
    "print(\"Best Parameters:\", xgb_grid.best_params_)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "id": "4841344b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "XGBoost Accuracy: 0.8823529411764706\n",
      "XGBoost F1 Score: 0.8856288278023073\n",
      "\n",
      "Classification Report\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.97      1.00      0.98        29\n",
      "           1       0.92      0.69      0.79        16\n",
      "           3       0.00      0.00      0.00         0\n",
      "           4       0.55      0.86      0.67         7\n",
      "           5       1.00      1.00      1.00         4\n",
      "           6       0.00      0.00      0.00         1\n",
      "           7       1.00      0.91      0.95        11\n",
      "\n",
      "    accuracy                           0.88        68\n",
      "   macro avg       0.63      0.64      0.63        68\n",
      "weighted avg       0.90      0.88      0.89        68\n",
      "\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# Predictions\n",
    "xgb_predictions = best_xgb.predict(X_test)\n",
    "\n",
    "# Accuracy\n",
    "xgb_accuracy = accuracy_score(y_test, xgb_predictions)\n",
    "\n",
    "# F1 Score\n",
    "xgb_f1 = f1_score(\n",
    "    y_test,\n",
    "    xgb_predictions,\n",
    "    average='weighted'\n",
    ")\n",
    "\n",
    "print(\"XGBoost Accuracy:\", xgb_accuracy)\n",
    "print(\"XGBoost F1 Score:\", xgb_f1)\n",
    "\n",
    "# Classification Report\n",
    "print(\"\\nClassification Report\")\n",
    "print(\n",
    "    classification_report(\n",
    "        y_test,\n",
    "        xgb_predictions,\n",
    "        # target_names=label_encoder.classes_\n",
    "    )\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "id": "4dd32ec7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": 
      "text/plain": [
       "<Figure size 800x600 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "# Confusion Matrix\n",
    "xgb_cm = confusion_matrix(y_test, xgb_predictions)\n",
    "\n",
    "plt.figure(figsize=(8,6))\n",
    "\n",
    "sns.heatmap(\n",
    "    xgb_cm,\n",
    "    annot=True,\n",
    "    fmt='d',\n",
    "    cmap='Greens'\n",
    ")\n",
    "\n",
    "plt.title('XGBoost Confusion Matrix')\n",
    "plt.xlabel('Predicted')\n",
    "plt.ylabel('Actual')\n",
    "\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "99ba494c",
   "metadata": {},
   "source": [
    "\n",
    "# Comparative Evaluation\n",
    "\n",
    "We compare:\n",
    "- Accuracy\n",
    "- F1-score\n",
    "- Feature importance\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "id": "d4ca4d35",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Model</th>\n",
       "      <th>Accuracy</th>\n",
       "      <th>F1 Score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Random Forest</td>\n",
       "      <td>0.867647</td>\n",
       "      <td>0.871443</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>XGBoost</td>\n",
       "      <td>0.882353</td>\n",
       "      <td>0.885629</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           Model  Accuracy  F1 Score\n",
       "0  Random Forest  0.867647  0.871443\n",
       "1        XGBoost  0.882353  0.885629"
      ]
     },
     "execution_count": 101,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "comparison = pd.DataFrame({\n",
    "    'Model': ['Random Forest', 'XGBoost'],\n",
    "    'Accuracy': [rf_accuracy, xgb_accuracy],\n",
    "    'F1 Score': [rf_f1, xgb_f1]\n",
    "})\n",
    "\n",
    "comparison\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d5464481",
   "metadata": {},
   "source": [
    "\n",
    "# Feature Importance Analysis\n",
    "\n",
    "Feature importance helps identify biological variables contributing most strongly to protein localization prediction.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "a3982885",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": 
      "text/plain": [
       "<Figure size 1000x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Feature  Importance\n",
      "5    alm1    0.275753\n",
      "0     mcg    0.229922\n",
      "6    alm2    0.175257\n",
      "1     gvh    0.173075\n",
      "4     aac    0.118144\n",
      "2     lip    0.024712\n",
      "3     chg    0.003138\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# Random Forest feature importance\n",
    "rf_importance = pd.DataFrame({\n",
    "    'Feature': X.columns,\n",
    "    'Importance': best_rf.feature_importances_\n",
    "}).sort_values(by='Importance', ascending=False)\n",
    "\n",
    "plt.figure(figsize=(10,5))\n",
    "\n",
    "sns.barplot(\n",
    "    x='Importance',\n",
    "    y='Feature',\n",
    "    data=rf_importance\n",
    ")\n",
    "\n",
    "plt.title('Random Forest Feature Importance')\n",
    "\n",
    "plt.show()\n",
    "\n",
    "print(rf_importance)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "id": "f02de94a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": 
      "text/plain": [
       "<Figure size 1000x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Feature  Importance\n",
      "5    alm1    0.339243\n",
      "1     gvh    0.151930\n",
      "0     mcg    0.140793\n",
      "6    alm2    0.127448\n",
      "4     aac    0.121397\n",
      "2     lip    0.119188\n",
      "3     chg    0.000000\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# XGBoost feature importance\n",
    "xgb_importance = pd.DataFrame({\n",
    "    'Feature': X.columns,\n",
    "    'Importance': best_xgb.feature_importances_\n",
    "}).sort_values(by='Importance', ascending=False)\n",
    "\n",
    "plt.figure(figsize=(10,5))\n",
    "\n",
    "sns.barplot(\n",
    "    x='Importance',\n",
    "    y='Feature',\n",
    "    data=xgb_importance\n",
    ")\n",
    "\n",
    "plt.title('XGBoost Feature Importance')\n",
    "\n",
    "plt.show()\n",
    "\n",
    "print(xgb_importance)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "525cf10a",
   "metadata": {},
   "source": [
    "\n",
    "# Scientific Discussion\n",
    "\n",
    "## Random Forest\n",
    "Random Forest uses bagging and bootstrap aggregation to reduce variance and improve prediction stability. Random feature selection helps avoid overfitting and improves biological classification performance.\n",
    "\n",
    "## XGBoost\n",
    "XGBoost uses gradient boosting, where models are trained sequentially to correct previous errors. This often improves predictive accuracy for complex biological relationships.\n",
    "\n",
    "## Biological Interpretation\n",
    "Features with higher importance values contribute more strongly to localization prediction. These variables may represent biological signals associated with membrane transport, protein charge, and amino acid composition.\n",
    "\n",
    "## Conclusion\n",
    "Both Random Forest and XGBoost are powerful ensemble learning methods for bioinformatics classification tasks. XGBoost often achieves higher predictive performance, while Random Forest provides robust and interpretable results.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e79fd3bf",
   "metadata": {},
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
