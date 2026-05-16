# \# CT-583: Tools and Techniques for Data Science

# \## End-of-Term Technical Challenge — Task 1

# 

# !\[Python](https://img.shields.io/badge/Python-3.10-blue)

# !\[Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange)

# !\[XGBoost](https://img.shields.io/badge/XGBoost-Latest-green)

# !\[License](https://img.shields.io/badge/License-Academic-lightgrey)

# 

# \---

# 

# \## Overview

# 

# This repository contains the complete implementation for \*\*Task 1: Classification with Comprehensive Analysis\*\* of the CT-583 Data Science end-of-term technical challenge.

# 

# The goal is to build an end-to-end machine learning pipeline that predicts whether an individual earns \*\*more than $50,000 per year\*\* using the UCI Adult Census Income Dataset, with thorough evaluation, failure case analysis, and fairness/bias assessment.

# 

# \---

# 

# \## Open in Google Colab

# 

# Click the button below to run the notebook directly in your browser — no installation required:

# 

# \[!\[Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rahimpg4300201-ship-it/CT583-Data-Science-Assignment/blob/main/task1\_classification.ipynb)

# 

# \---

# 

# \## Repository Structure

# 

# ```

# CT583-Data-Science-Assignment/

# │

# ├── task1\_classification.ipynb   # Main Jupyter notebook (complete implementation)

# ├── task1\_report.md              # Written report with analysis and findings

# └── README.md                    # This file

# ```

# 

# \---

# 

# \## Dataset

# 

# \*\*UCI Adult Census Income Dataset\*\* (also known as the Census Income dataset)

# 

# | Attribute | Detail |

# |-----------|--------|

# | Source | UCI Machine Learning Repository via OpenML |

# | Records | \~48,842 total |

# | Features | 14 (age, education, occupation, race, sex, etc.) |

# | Target | Binary: >50K vs <=50K income |

# | Class Split | \~76% <=50K / \~24% >50K |

# 

# The dataset is loaded automatically inside the notebook — no manual download required:

# 

# ```python

# from sklearn.datasets import fetch\_openml

# adult = fetch\_openml('adult', version=2, as\_frame=True)

# ```

# 

# \---

# 

# \## What the Notebook Covers

# 

# | Section | Description |

# |---------|-------------|

# | 1. Imports | All required libraries and configuration |

# | 2. Data Loading | Fetch dataset directly from OpenML |

# | 3. EDA | Target distribution, age and categorical visualisations |

# | 4. Preprocessing | Missing value imputation, encoding, scaling, feature engineering |

# | 5. Model Training | Logistic Regression, Random Forest, XGBoost with RandomizedSearchCV |

# | 6. Evaluation | Accuracy, Precision, Recall, F1, AUC-ROC, confusion matrix, ROC curves |

# | 7. Failure Cases | Top 5 misclassified instances with detailed analysis |

# | 8. Fairness Assessment | Bias evaluation across gender and race |

# | 9. Bias Mitigation | Sample reweighing strategy with before/after comparison |

# | 10. Feature Importance | Top 20 XGBoost feature importances |

# | 11. Summary | Conclusions and limitations |

# 

# \---

# 

# \## Models Used

# 

# | Model | Tuning Method | Scoring Metric |

# |-------|--------------|----------------|

# | Logistic Regression | RandomizedSearchCV (5-fold CV) | AUC-ROC |

# | Random Forest | RandomizedSearchCV (5-fold CV) | AUC-ROC |

# | XGBoost | RandomizedSearchCV (5-fold CV) | AUC-ROC |

# 

# \---

# 

# \## Feature Engineering

# 

# Four new features were created from existing data:

# 

# | Feature | Description |

# |---------|-------------|

# | `age\_group` | Age binned into life stages (Youth, YoungAdult, MiddleAge, Senior, Elderly) |

# | `capital\_net` | capital-gain minus capital-loss |

# | `hours\_category` | Hours-per-week binned (PartTime, FullTime, Overtime, Extreme) |

# | `edu\_age\_interaction` | education-num multiplied by age |

# 

# \---

# 

# \## Fairness Assessment

# 

# The model is evaluated separately for:

# \- \*\*Gender\*\*: Male vs Female

# \- \*\*Race\*\*: White vs Non-white

# 

# A disparity greater than 5% in any metric flags potential bias. A \*\*sample reweighing\*\* mitigation technique is applied to reduce identified disparities.

# 

# \---

# 

# \## Requirements

# 

# All dependencies are installed automatically in the notebook. For local runs:

# 

# ```bash

# pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter

# ```

# 

# \---

# 

# \## How to Run

# 

# \### Option 1 — Google Colab (Recommended)

# 1\. Click the \*\*Open in Colab\*\* badge above

# 2\. Click \*\*Runtime → Run All\*\*

# 3\. Wait approximately 5-10 minutes for all cells to complete

# 

# \### Option 2 — Local (Jupyter)

# ```bash

# git clone https://github.com/rahimpg4300201-ship-it/CT583-Data-Science-Assignment.git

# cd CT583-Data-Science-Assignment

# pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter

# jupyter notebook task1\_classification.ipynb

# ```

# 

# \---

# 

# \## Report

# 

# The full written report is available in \[`task1\_report.md`](./task1\_report.md) and covers:

# 

# \- Model architecture and hyperparameter selection rationale

# \- Complete evaluation metrics with visualisations

# \- Failure case analysis with 5 real misclassified examples

# \- Bias assessment across gender and race

# \- Mitigation strategy and results

# 

# \---

# 

# \## Author

# 

# | | |

# |---|---|

# | \*\*Name\*\* | Insha Rahim Shahwani |

# | \*\*Student ID\*\* | GD-001/2025 |

# | \*\*Course\*\* | CT-583 Tools and Techniques for Data Science |

# | \*\*Submission Date\*\* | 2025 |

# 

# \---

# 

# \*This project was completed as part of the CT-583 end-of-term technical challenge.\*

