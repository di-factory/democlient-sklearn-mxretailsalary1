import numpy as np


class Pytorch_OneHot_Tx:
    def __init__(self):
        self.metadata = []

    def fit(self, X, y=None):
        for col in range(X.shape[1]):
            dict = {}
            dict["col"] = col
            dict["value_range"] = {}
            for num, val in enumerate(np.unique(X[:, col])):
                dict["value_range"][val] = num
            self.metadata.append(dict)
            print(dict)

    def transform(self, X):
        num_columns = sum(len(col["value_range"]) for col in self.metadata)
        X_result = np.zeros((X.shape[0], num_columns))  # Initialize the result matrix

        current_col = 0
        for col_metadata in self.metadata:
            col = col_metadata["col"]
            col_mapping = col_metadata["value_range"]

            for r in range(X.shape[0]):
                value = X[r, col]
                value_index = col_mapping.get(
                    value, -1
                )  # Get the index from the mapping

                if value_index != -1:  # If value exists in mapping
                    X_result[
                        r, current_col + value_index
                    ] = 1  # Set the corresponding value to 1

            current_col += len(col_mapping)  # Move to the next set of columns

        return X_result


m = np.array([["A", "Z", 2, 3], ["B", "Y", 3, 4], ["C", "X", 6, 7]])

p = Pytorch_OneHot_Tx()
p.fit(m[:, :2])
print(p.transform(m[:, 0:2]))
print(p.transform(np.array([["A", "X"]])))
print(p.metadata)
