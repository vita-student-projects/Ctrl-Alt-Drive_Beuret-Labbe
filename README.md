# DLAV Project – Phase 1 – Sylvain Beuret & Victor Labbe

This repository implements a deep learning model for autonomous trajectory prediction, developed for Deep Learning for Autonomous class. The model predicts the future motion of an ego vehicle from RGB camera input and past motion history, using a **multi-modal ResNet-based architecture**.

---

##  Model Overview

The core model is `BetterDrivingPlanner`, an end-to-end deep neural network with the following structure:

- **Visual encoder**: largely based on a pretrained **ResNet-18**, it extracts features from the input RGB image.
- **History encoder**: A multi-layer perceptron (MLP) encodes past vehicle motion. Other structures have been tested such as LSTM but did not yield better results.
- **Fusion module**: Combines vision and motion features via a fully connected network.
- **Multi-modal trajectory decoder**: Outputs **K=6 possible future trajectories** and their **confidence scores**.
- **Best trajectory selection**: The model selects the highest-confidence trajectory at inference time in a Winner takes All approach. 

This model is rather simple but infers pretty well after some tuning of its hyperparameters since it achieved an ADE of 1.61.
Some more complex model have been tried notably leveraging transformer-based modules but the result where simply not as good. 
We think this might come from the small size of the training dataset. Using pretrained models aswell as some augmentation could be tried to circumvent this issue.   


---

##  Additional Tools 

We implemented a few tools to get the best out of our models trainings:
- **Logger**: Our logger regularly plot the metrics fed in arguments to track them during training
- **Hyperparameter Tuning Tool**: This block is based on [optuna](https://optuna.org/) - a bayesian optimisation library - that find the best hyperparameters for the  model through try and error in a sample efficient way.

---

##  How to Use

The jupyter notebook attached provide all necessary code to train, infer, tune our model. Moreover each block is carefully preceded by a markdown detailing its function.

However here are the main parts of this notebook:

### 1. Setup
Make sure to install all required dependencies:
```bash
pip install torch torchvision pandas matplotlib gdown optuna
```
You can then run the import cell at the beginning of the notebook

### 2. Download Data
Data is automatically downloaded and extracted at the beginning of the notebook using `gdown`.

### 3. Tune the Model
Run all cells under **Tuning Section**:
```python
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=300)
```
Select a low number of epochs for each trainings and run a significant amount of trial to cover your parameter space. 
This can also be done in an iterative manner (for instance first tune the learning rate, then tune its scheduler). 

### 4. Train the Model
Run all cells under **Training Section** with your best hyperparameters !:
```python
model = BetterDrivingPlanner()
train(model, train_loader, val_loader, optimizer, logger, num_epochs=50)
```
You can visualize the evolution of your losses thanks to the plot that the logger saves regularly in the log folder.

### 5. Visualize Predictions
The notebook includes a visualization cell that allow to estimate how your model predicts trajectories on a few samples:
- Past trajectory (gold)
- Ground truth future (green)
- Predicted future (red)

### 6. Inference & Submission
To generate a Kaggle submission:
```python
# Run inference on test set
traj_pred, conf_pred = model(camera, history)
pred = model.predict_best(traj_pred, conf_pred)

# Save to CSV in required format
df_xy.to_csv("submission_phase1.csv", index=False)
```

---

## Acknowledgments

This project draws on ideas from:
- **[TransFuser (ECCV 2022)](https://arxiv.org/abs/2205.15997)** – For multi-modal fusion and ResNet feature extraction.
- **[UniAD (CVPR 2023)](https://arxiv.org/abs/2303.00745)** – For multi-modal trajectory + confidence prediction.
- **[End-to-end Autonomous Driving: Challenges and Frontiers (IEEE 2024)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10614862)** – For system-level design insights.
