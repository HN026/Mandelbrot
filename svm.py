import torch
from sklearn.svm import SVR 
from sklearn.metrics import mean_squared_error
from sklearn.datasets import MandelbrotDataSet
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

dataset = MandelbrotDataSet(10000, max_depth=1500, gpu=True)

x = dataset.inputs.numpy()
y = torch.unsqueeze(dataset.outputs, 1).numpy()

model = SVR(C=50, epsilon=0.1, gamma='scale', kernel='rbf')

model.fit(x, y.ravel())

y_pred = model.predict(x)
print("After training, MSE: ", mean_squared_error(y, y_pred))

x_range = np.linspace(-1,1,300)
y_range = np.linspace(-1,1,200)
x_grid, y_grid = np.meshgrid(x_range, y_range)

grid = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

z_pred = model.predict(grid)
z_grid = z_pred.reshape(x_grid.shape)

fig, axs = plt.subplots(1,2,figsize=(10,5))

axs[1].imshow(z_grid, origin='lower', extent=(0,1,0,1), cmap='viridis')
axs[1].set_title('Predicted Output')

plt.tight_layout()
plt.show()