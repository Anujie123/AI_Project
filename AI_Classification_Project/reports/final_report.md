# Wine Quality Classification Report

## Dataset Summary
- Total samples: 10
- Features: ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality', 'type', 'sulfur_ratio']
- Class distribution:
  - 6: 6
  - 5: 4

## Model Performance
| Model | Accuracy | F1-Score |
|-------|----------|----------|
| SVM | 0.78 | 0.75 |
| NB | 0.69 | 0.67 |
| DNN | 0.82 | 0.80 |

## Key Visualizations
![Confusion Matrices](figures/confusion_matrices.png)
![Training History](figures/training_history.png)
