# 绿顶智析 · 完整在线版的 Streamlit 移植包

此包恢复原 Sites 在线系统（RoofScope v2）的实际 HTML、CSS、JavaScript、真实样例与完整训练模型资源。不是只有图片上传的 Streamlit 框架，也不是嵌入旧网站的网址。原网站关闭后此包仍能独立运行。

## 包含的实际功能

1. 批量任务：影像、建筑掩膜、绿顶掩膜、NDVI 文件名配对；队列、进度、失败原因和结果汇总。
2. 真实 GR-Net：用户 checkpoint_epoch50.pth 转换的 ONNX float32 模型，内置完整模型分片与哈希校验；浏览器本地运行，不使用随机模型。
3. 地图与研究范围：GeoTIFF、坐标与地理叠加、区域圈定、建筑连通域及网格统计。在线底图需网络。
4. 人工校核：屋顶/绿顶画笔修正、多边形修正、撤销、恢复原始结果和重新计算。
5. 双口径覆盖率：绿顶/屋顶、绿顶/有效研究区域；像素统计及已知分辨率时的面积。
6. 温度配对：带地理参考的 LST、日期/尺度校验、摄氏/开尔文与比例系数转换、生成配对表；也支持 CSV。
7. Pearson 显著性、Fisher 95% CI、散点与回归线、Spearman 对照、异常值、分组检验、自定义四档关联评级。
8. 项目保存、恢复、备份、样本管理、分割掩膜/CSV/JSON/完整研究报告导出。

保持原版七个导航页面：影像分析、批量任务、地图、热环境评估、样本库、项目与成果、方法与模型。报告为可打印 HTML：打开导出的报告，按 Ctrl+P，选择“保存为 PDF”。并未将浏览器打印冒称直接服务端 PDF 生成。

## 在已有 GitHub 项目中更新（适用于 Sunshineperm/GreenRoof-AI-System）

1. 解压本包。
2. 将包内所有内容复制到 D:\green\GreenRoof-AI-System-Final，替换旧 app.py、requirements.txt、README.md，并确保新增的 frontend 文件夹与 app.py 同级。
3. 保留你已有的 models_checkpoint_epoch50.pth、grnet、data、modules 和 Git 历史；本包不会删除这些文件。
4. 在项目目录打开 Git Bash，执行：

```bash
git add app.py requirements.txt README.md frontend backend .streamlit .gitignore .gitattributes package.json package-lock.json tests scripts
git commit -m "Restore complete RoofScope workbench and trained model"
git push
```

5. Streamlit 的仓库、main 分支和 app.py 入口保持不变。推送后它会重新部署。若仍显示旧页面，在 Manage app 中重启，然后刷新。
6. 请使用 Python 3.12。requirements.txt 固定了本次启动验证使用的 Streamlit 1.64.0。

重要：只替换 app.py 不够。frontend 中的页面、脚本、样例、WASM 和模型分片必须全部上传。每个模型分片为约 8 MiB，不超过 GitHub 普通单文件限制。旧 PTH 权重继续通过已有 Git LFS 管理，不需要重复下载或修改路径。

## 本地运行

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

直接双击 app.py 不能启动网站。此部署方式无需 npm 构建、无需 CUDA、无需额外模型服务。

## GR-Net 输入与运行

默认使用浏览器内置模型：不填写“模型服务地址”。需要现代桌面浏览器、真实可见光/NIR 波段和训练编码相同的 NDVI。四波段 BGRN/RGBN TIFF 为首选；普通 RGB 图片只能做辅助识别，不能伪造近红外输入。

点击“载入真实测试样例”，再将分析方式改为 GR-Net，并点击分析即可验证。样例配有真实 NDVI；初始标注覆盖率不等于模型预测覆盖率。

模型约 115 MiB，首次加载下载并缓存，运行使用设备 CPU/WASM。浏览器模型限制 400 万像素，大图请切片。若填写独立服务地址，将切换到模型服务；backend 目录保留了 Python 服务和模型结构，部署它需安装 backend/requirements.txt 并提供原 PTH。

## 数据及科学解释

项目保存使用当前浏览器 IndexedDB，本地缓存不会自动跨设备同步；请下载项目备份。更换网站域名不会迁移原域名的本地项目。

没有真实配对温度样本时，系统不生成真实热岛结论。Pearson 与四档评级描述关联证据，不能证明因果降温；等级是可调整的项目规则。建筑连通域不是经测绘确认的独立建筑。

## 源码结构

- app.py：完整工作台的 Streamlit 宿主
- frontend/：原在线版的完整前端和训练模型
- frontend/streamlit-bridge.js：仅负责与 Streamlit 握手和设置显示高度
- backend/：可选 Python 模型服务、模型结构、导出脚本
- tests/：覆盖率、配对、统计、项目恢复、批量和修正的原版测试
- scripts/check.mjs：语法、DOM 和资源检查

开发验证（部署不需要）：npm ci，然后 npm test 和 npm run check。
