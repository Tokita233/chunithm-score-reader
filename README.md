# CHUNITHM 分表识别器

## 手机/云端部署

本项目包含 `render.yaml`。将仓库连接到 Render 后即可获得手机可访问的 HTTPS 地址：

1. 登录 Render，选择 **New + → Blueprint**。
2. 连接本 GitHub 仓库。
3. 确认创建 `chunithm-score-reader` 服务并等待部署完成。

免费实例在一段时间无人访问后可能休眠，首次打开需要等待启动。上传的图片仅在内存中识别，不会保存到服务器磁盘。

## 使用

1. 安装 64 位 Python 3.11 或 3.12，并勾选安装器中的 “Add Python to PATH”。
2. 双击 `启动.bat`。首次运行会下载并安装 OCR 组件。
3. 浏览器打开后选择分表图片，默认使用自动检测，也可手动限制为 45/50 首，然后点击“开始识别”。
4. 对照每行左侧的小图校正结果，再点该行“复制”按钮。

识别和显示顺序固定为：同一行从左到右，完成一行后再从下一行左侧开始，即整体从上到下。

输出格式固定为：

```text
upsert 分数 名字 难度
```

例如：

```text
upsert 1007649 taboo tears you up Ultima
upsert 1007647 Cult future Master
```

## 说明

- 自动判断并支持两种五列分表：Diving-Fish / Tippy Bot 旧 UI，以及 CHUNITHM MATE Best 50 新 UI。
- 歌名使用日英混合 OCR，支持平假名、片假名、日文汉字、英文和数字。
- 难度主要根据顶部色块识别：紫色为 Master，黑底红纹为 Ultima。
- OCR 无法保证每一首歌名都完全正确，所以页面保留了可编辑的分数、歌名和难度字段。
- 第一次运行需要联网安装依赖；之后可离线运行。
