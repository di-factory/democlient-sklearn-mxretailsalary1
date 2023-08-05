import numpy as np

    
m = np.array([
        ['A', 'Z', 2, 3],
        ['B', 'Y', 3, 4],
        ['C', 'X', 6, 7]
    ])

selec = [2]

print(m[selec])


selected_cols = [0, 1]  # Select columns 0 and 1

selected_data = m[:, selected_cols]
print(selected_data)
