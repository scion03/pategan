import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def data_generator(no, dim, noise_rate):
    """Generate train/test dataset for PATE-GAN evaluation
  
  Args:
    no: The number of train/test samples
    dim: The number of dimensions in train/test features
    noise_rate: The amount of noise for generating labels
    
  Returns:
    train_data: Training data (feature + label)
    test_data: Testing data (feature + label)
  """

    # Define symmetric covariance matrix for generating features
    # cov_matrix = np.random.uniform(0, 1, [dim, dim])
    # cov_matrix = 0.5 * (cov_matrix + np.transpose(cov_matrix))
    df = pd.read_csv('diabetes.csv')
 
    # Generate train/test features
    # x_train = np.random.multivariate_normal(
    #     np.zeros(
    #         [
    #             dim,
    #         ]
    #     ),
    #     cov_matrix,
    #     [
    #         no,
    #     ],
    # )
    # x_test = np.random.multivariate_normal(
    #     np.zeros(
    #         [
    #             dim,
    #         ]
    #     ),
    #     cov_matrix,
    #     [
    #         no,
    #     ],
    # )

    # Define feature label relationship
    W = np.random.uniform(
        0,
        1,
        [
            dim,
        ],
    )
    b = np.random.uniform(0, 1, 1)

    # Generate train/test labels
    # y_train = np.matmul(x_train, W) + b + np.random.normal(0, noise_rate, no)
    # y_train = np.reshape(1 * (y_train), [-1, 1])

    # y_test = np.matmul(x_test, W) + b + np.random.normal(0, noise_rate, no)
    # y_test = np.reshape(1 * (y_test), [-1, 1])

    df = pd.read_csv('Real estate.csv')
    # get the locations
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, random_state=0
    )
    train_data = np.concatenate((x_train, y_train), axis=1)
    test_data = np.concatenate((x_test, y_test), axis=1)
    # Normalization
    for i in range(dim + 1):
        train_data[:, i] = train_data[:, i] - np.min(train_data[:, i])
        train_data[:, i] = train_data[:, i] / (np.max(train_data[:, i]) + 1e-8)

    # Normalization
    for i in range(dim + 1):
        test_data[:, i] = test_data[:, i] - np.min(test_data[:, i])
        test_data[:, i] = test_data[:, i] / (np.max(test_data[:, i]) + 1e-8)

    return train_data, test_data
