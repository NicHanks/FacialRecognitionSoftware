Facial Recognition with LFW Dataset

60/20/20 Split, Two Models (Logistic Regression & Decision Tree), Full Metrics

This project demonstrates facial recognition classification using the Labeled Faces in the Wild (LFW) dataset. Two machine learning models—Logistic Regression (OvR) and a Decision Tree—are trained and evaluated using a stratified 60/20/20 split (train/validation/test).

We evaluate the models using classification reports, confusion matrices, and ROC curves, with proper preprocessing and scaling to ensure fair comparison.

📂 Dataset

Source: fetch_lfw_people
 (built into scikit-learn)

Filtering: Only people with ≥70 images are included to avoid extremely small classes.

Shape:

Samples: n_samples

Features: flattened grayscale images (n_features)

Classes: multiple public figures (n_classes)

⚙️ Methodology

Data Preparation

Load LFW dataset

Apply stratified split:

60% training

20% validation

20% test

Preprocessing

Features scaled with StandardScaler (important for linear models).

Scaling fitted only on training set to avoid data leakage.

Models

Logistic Regression (One-vs-Rest)

Solver: liblinear

Max iterations: 2000

Decision Tree

Depth: max 25

Min samples split: 4

Evaluation

Final models trained on train + validation (common practice after fixing hyperparameters).

Metrics:

Classification report (precision, recall, F1)

Confusion matrix visualization

ROC curve (OvR, per-class AUC)

📊 Results Summary
Model	Accuracy	Macro F1	Weighted F1	Notes
Logistic Regression OvR	~85%	~0.82	~0.84	Performs better on high-dimensional data
Decision Tree	~70%	~0.67	~0.69	More interpretable, but prone to overfitting

Values are approximate and may vary slightly per run due to random splits.

🖼️ Example Visualizations
Confusion Matrices

Logistic Regression


Decision Tree


ROC Curves

Logistic Regression OvR


Decision Tree OvR


▶️ Usage
1. Clone Repository
git clone https://github.com/yourusername/facial-recognition-lfw.git
cd facial-recognition-lfw

2. Install Dependencies
pip install numpy matplotlib scikit-learn


(Optional, for extended use):

pip install opencv-python face_recognition Pillow

3. Run Script
python facial_recognition.py


This will:

Download the LFW dataset (first run only)

Train both models

Display classification reports

Save confusion matrices & ROC plots to /images/

🧾 Example Output Snippet
Samples: 1288, Features: 1850, Classes: 7
Names: ['Ariel Sharon' 'Colin Powell' 'Donald Rumsfeld' ... ]
=== Logistic Regression: classification report ===
              precision    recall  f1-score   support
...
macro avg      0.8231     0.8154     0.8189       200
weighted avg   0.8420     0.8500     0.8457       200

📌 Notes

Models are not tuned; hyperparameters are fixed for simplicity.

Validation set included to reflect realistic ML workflow.

Scaling only applied to models requiring it.

📜 Attribution

Dataset: Labeled Faces in the Wild (LFW), University of Massachusetts, Amherst.

Portions of code assisted by OpenAI’s ChatGPT (2025), reviewed and modified for correctness.