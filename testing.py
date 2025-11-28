import numpy as np
import plotly.express as px

a = np.array(
    [
        [
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 0],
        ],
        [
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0],
        ],
        [
            [1, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ],
    ]
)
b = np.array([np.float32(6), np.float32(0), np.float32(6)])

c = a[b == 6]
print(f"c:\n{c}")

d = c.sum(axis=0)
print(f"d:\n{d}")

# d = c.nonzero()
# print(f"d:\n{d}")
# x = d[1]
# y = d[2]


# fig = px.scatter(x=x, y=y)
# fig.show()
