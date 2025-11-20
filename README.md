# This resource package contains the entire dataset used in the study, the source code, and a detailed list of dependency versions.
[ACGAN-SA.bmp](https://github.com/user-attachments/files/23641378/ACGAN-SA.bmp)
The code is divided into several scripts; here is a brief overview of the workflow in plain language:
Before running, the dataset should be cleaned by removing outliers—samples with identical elemental composition and heat treatment but drastically different tensile strengths must be deleted; if the difference is small, take the average.
Next, physical feature descriptors are calculated from the composition (Elements_properties.py).
Highly correlated features are then removed based on Pearson correlation coefficients (Initial Feature Elimination.py).
The optimal number and names of features are determined via recursive feature elimination (recursive_feature_elimination.py).
The machine-learning regression model with the strongest generalization ability is selected from Artificial Neural Network.py, Gradient Boosting Regression Tree.py, Random Forest.py, and Support Vector Regression.py.
An ACGAN is trained (ACGAN.py) to obtain a generator (generator.pt) that can produce new aluminum-alloy samples with higher specific strength conditioned on labels.
Finally, ACGAN-SA.py is executed; by modifying the code according to the decision-maker’s preference, it can generate novel aluminum-alloy samples that favor either high strength or low density, with specific strength greater than 142.
