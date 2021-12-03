# Deep Hough Transform

### Build and Compile
Please verify your environment by running `python3 -m torch.utils.collect_env` and make sure you have proper environment.
We suggest `torch>=1.4.0` and `gcc>=7.3.0`. Please note that the gcc should be compatitable with torch
else it will raise an error.


### Example
```
dh = DeepHough(num_angle=180, num_bias=100)
x = torch.zeros(100, 100).cuda()
x[40, :] = 1
x[60, :] = 1
x[:, [40, 50, 60]] = 1
y = dh(x.view(1, 1, 100, 100))
```
![](./results.png)

### Pre-built wheels

| Python  | CUDA | Torch | Wheel |
| ------------- | -------- | -------- | -------- |
| 3.8  | 11.0  | 1.9.0 | <http://data.kaizhao.net/projects/deep-hough-transform/wheels/deephough-0.1.0-cp38-cp38-linux_x86_64.whl> |