### 环境配置

python 版本 3.7

推荐使用 anaconda 配置环境

若需要使用cupy，参考 https://docs.cupy.dev/en/stable/install.html 进行安装
```bash
conda install -c conda-forge cupy=10.2 cudatoolkit=11.6
```

### numpy 切换为 cupy
在 my_import.py 中将 `import numpy as np` 改为 `import cupy as np`

将函数 `np.add.at` 修改为 `cupyx` 库的 `scatter_add`，如下
```python
# numpy
np.add.at(x, (slice(None), k, i, j), rows_reshaped)
# cupy
import cupyx as npx
npx.scatter_add(x, (slice(None), k, i, j), rows_reshaped)
```
