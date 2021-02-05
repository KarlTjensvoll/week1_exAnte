import numpy as np
from numpy import linalg as la
from tabulate import tabulate


def estimate(
        y: np.array, x: np.array, transform='', n=None, t=None
    ) -> list:
    
    b_hat = est_ols(y, x)
    resid = y - x@b_hat
    u_hat = resid@resid.T
    SSR = resid.T@resid
    SST = (y - np.mean(y)).T@(y - np.mean(y))
    R2 = 1 - SSR/SST

    sigma, cov, se = variance(transform, SSR, x, n, t)
    t_values = b_hat/se
    
    names = ['b_hat', 'se', 'sigma', 't_values', 'R2', 'cov']
    results = [b_hat, se, sigma, t_values, R2, cov]
    return dict(zip(names, results))

    
def est_ols( y: np.array, x: np.array) -> np.array:
    return la.inv(x.T@x)@(x.T@y)

def variance( 
        transform: str, 
        SSR: float, 
        x: np.array, 
        n: int,
        t: int
    ) -> tuple:

    k = x.shape[1]
    if not n:
        n = x.shape[0]

    if transform in ('', 'be'):
        sigma = SSR/(n - k))
    elif transform.lower() == 'fe':
        sigma = SSR/(n * (t - 1) - k)
    elif transform.lower() == 're':
        sigma = SSR/(t * (n - k))
    else:
        raise Exception('Invalid transform provided.')

    cov = sigma*la.inv(x.T@x)
    se = np.sqrt(cov.diagonal()).reshape(-1, 1)
    return sigma, cov, se


def print_table(
        labels: tuple,
        results: dict,
        headers=["", "Beta", "Se", "t-values"],
        title="Results",
        _lambda=None,
        **kwargs
    ) -> None:
    label_y, label_x = labels
    # Create table for data on coefficients
    table = []
    for i, name in enumerate(label_x):
        row = [
            name, 
            results.get('b_hat')[i], 
            results.get('se')[i], 
            results.get('t_values')[i]
        ]
        table.append(row)
    
    # Print table
    print(title)
    print(f"Dependent variable: {label_y}\n")
    print(tabulate(table, headers, **kwargs))
    
    # Print data for model specification
    print(f"R\u00b2 = {results.get('R2').item():.3f}")
    print(f"\u03C3\u00b2 = {results.get('sigma').item():.3f}")
    
    if _lambda:
        print(f'\u03bb = {_lambda.item():.3f}')


def perm( Q_T: np.array, A: np.array, t=0) -> np.array:
    """Takes a transformation matrix and performs the transformation on 
    the given vector or matrix.

    Args:
        Q_T (np.array): The transformation matrix. Needs to have the same
        dimensions as number of years a person is in the sample.
        
        A (np.array): The vector or matrix that is to be transformed. Has
        to be a 2d array.

    Returns:
        np.array: Returns the transformed vector or matrix.
    """
    # We can infer t from the shape of the transformation matrix.
    if t==0:
        t = Q_T.shape[0]

    # Initialize the numpy array
    Z = np.zeros(A.shape)

    # Loop over the individuals, and permutate their values.
    for i in range(int(A.shape[0]/t)):
        Z[i*t: (i + 1)*t] = Q_T@A[i*t: (i + 1)*t]
    return Z


def load_example_data():
    # First, import the data into numpy.
    data = np.loadtxt('wagepan.txt', delimiter=",")
    id_array = np.array(data[:, 0])

    # Count how many persons we have. This returns a tuple with the 
    # unique IDs, and the number of times each person is observed.
    unique_id = np.unique(id_array, return_counts=True)
    n = unique_id[0].size
    t = int(unique_id[1].mean())
    year = np.array(data[:, 1], dtype=int)

    # Load the rest of the data into arrays.
    y = np.array(data[:, 8]).reshape(-1, 1)
    x = np.array(
        [np.ones((y.shape[0])),
            data[:, 2],
            data[:, 4],
            data[:, 6],
            data[:, 3],
            data[:, 9],
            data[:, 5],
            data[:, 7]]
    ).T

    # Lets also make some variable names
    label_y = 'Log wage'
    label_x = [
        'Constant',
        'Black',
        'Hispanic',
        'Education',
        'Experience',
        'Experience sqr',
        'Married',
        'Union'
    ]
    return y, x, n, t, year, label_y, label_x