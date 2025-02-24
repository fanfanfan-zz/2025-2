
# PyTorch + MNIST 手写数字识别

一个基于 PyTorch 的 MNIST 手写数字 1～10 的识别模型，支持训练、验证和保存最佳模型。最终得到的模型准确率 **acc > 99%**。

## 功能特性
- 使用卷积神经网络（CNN）进行图像分类。
- 自动保存训练过程中的最佳模型。
- 实时监控训练损失和验证准确率。

---

## 安装步骤

1. **克隆仓库**：
   ```bash
   git clone https://github.com/your_username/mnist-pytorch.git
   cd mnist-pytorch
   ```

2. **修改路径**：
   - 找到并打开源代码 `train_and_test.py`。
   - 修改第 18 行和第 26 行的 `root` 路径为你自己的绝对路径（注意是 `train_and_test.py` 的上一级路径）。

3. **激活虚拟环境**
   ```bash
   conda activate env_name
   
   ```
4. **安装依赖库**：
   ```env
   pip install torch==2.0.1 torchvision==0.15.2 matplotlib==3.7.1 opencv-python==4.7.0.72 numpy==1.24.3
   ```

5. **运行代码**：
   ```env
   python train_and_test.py
   ```

---

## 结果示例
- 训练过程中会输出日志，包括每个 epoch 的损失和验证准确率。
- 最终模型会保存在 `best_cnn.pkl` 文件中。

---

## 依赖库
- `torch==2.0.1`
- `torchvision==0.15.2`
- `matplotlib==3.7.1`
- `opencv-python==4.7.0.72`
- `numpy==1.24.3`

---

## 项目结构
```text
mnist-pytorch/
├── train_and_test.py  # 训练与测试脚本
├── best_cnn.pkl       # 保存好的最佳模型
├── README.md          # 项目说明
└── MNIST/              # MNIST 数据集
```

---

## 注意事项
- 确保 `root` 路径设置正确。
- 训练过程中会实时显示损失和准确率，最终模型准确率可达 **99%** 以上。



