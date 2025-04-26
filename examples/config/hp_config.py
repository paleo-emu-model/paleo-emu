import pandas as pd

# 参数数量
nkeep = 15

# 高斯过程超参数表
hp = pd.DataFrame({
    'l.co2':     [0.523323] * nkeep,
    'l.esinw':   [2.791735] * nkeep,
    'l.ecosw':   [1.310285] * nkeep,
    'l.obl':     [1.663824] * nkeep,
    'l.icevol':  [10.000000] * nkeep,
    'nugget':    [0.000000000224038] * nkeep
}).T  # transpose to shape (6, nkeep) for GP routine