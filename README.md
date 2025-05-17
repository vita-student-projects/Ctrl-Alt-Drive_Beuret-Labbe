# DLAV Project – Phase 2 – Sylvain Beuret & Victor Labbe

This repository implements a deep learning model for autonomous trajectory prediction, developed for Deep Learning for Autonomous class : https://github.com/vita-student-projects/Ctrl-Alt-Drive_Beuret-Labbe/tree/Milestone-II


The model predicts the future motion of an ego vehicle from RGB camera input and past motion history. The model also compute depth maps and semantics to improve the common visual encoder based on a ResNet architecture

We managed to obtained the following results:
| Metric       | Value |
|--------------|--------|
| ADE (Validation) | **1.40** |
| FDE (Validation) | 3.97     	|

---

##  Model Overview

The core model is `BetterDrivingPlanner`, an end-to-end deep neural network with the following structure:

- **Visual encoder**: Largely based on a pretrained **ResNet-34**, this encoder extract the important features in the camera image. The last 2 layers are replaced and the layer is fine-tuned on our dataset. 

- **Depth and Semantics Decoder**: The depth and semantics maps are computed from the visual features extracted by the encoder. Those features are then processed in **small ConvNets**. The resulting maps are pretty rough and do not match closely the ground truth. We managed to obtained very good maps by leveraging a **more complex architecture inspired of a U-net with residual connections to a ResNet34 encoder** but this cost us in term of performance on the planner. We also tried to embed and feed the auxialiary maps in the fusion MLP but the performance was degraded as well. Those tries can be accessed in the 'Extra' folder.

- **History encoder**: A multi-layer perceptron (MLP) encodes past vehicle motion. Many other structures have been tested such as LSTM, GRU and small Transformers but we obtained the better results with our simple MLP.

- **Fusion module**: Combines vision and motion features via a fully connected network. Here again we tried to implement a Transformer architecture to leverage the attention mechanism for learning the relations between camera and history but this was not concluding probably due to the size of our dataset. 

- **Multi-modal trajectory decoder**: Fully connected networks outputs **multiple possible future trajectories** and their **confidence scores**.
- **Best trajectory selection**: The model selects the highest-confidence trajectory at inference time in a Winner takes All approach. 


---

##  Additional Tools 

We implemented a few tools to get the best out of our models trainings:
- **Logger**: Our logger regularly plot the metrics fed in arguments to track them during training
- **Hyperparameter Tuning Tool**: This block is based on [optuna](https://optuna.org/) - a bayesian optimisation library - that find the best hyperparameters for the  model through try and error in a sample efficient way.
- **Dataset Augmentation**: To obtain better generalization of our model, we doubled the size of the training dataset by flipping cameras and adapting accordingly trajectories. We also apply different lightning variations and Gaussian blur at every epoch during training to augment our performance on unseen data.

---

##  How we achieved our best model 

Our best model was achieved with the following parameters:
- Our model focus on only one auxiliary task : depth estimation with a weight of 0.05 
- 6 Trajectories are predicted for 60 steps, the best one is selected in a Winner-Takes-All approach
- LayerNorm, Dropout and Weight-Decay are used to stabilize and regularize. 
- A Pretrained Resnet34 visual encoder is used frozen with the exception of its initial and final layers.
- A learning rate scheduler is adopted to process the learning rate from 2e-4 to 5e-5.
- Our model is trained on small amounts of epochs and then retrained with data augmentation starting from the best model previous weights

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
