# GitHub 开源项目最佳实践指南

本文档总结了将本地项目上传并开源到 GitHub 的完整流程和最佳实践，可作为后续项目的参考模板。

---

## 📋 目录

1. [开源前的准备工作](#1-开源前的准备工作)
2. [必备文件清单](#2-必备文件清单)
3. [.gitignore 配置规范](#3-gitignore-配置规范)
4. [README 文档写作规范](#4-readme-文档写作规范)
5. [Git 操作流程](#5-git-操作流程)
6. [开源许可证选择](#6-开源许可证选择)
7. [项目维护建议](#7-项目维护建议)

---

## 1. 开源前的准备工作

### 1.1 代码审查清单

在开源之前，请确保检查以下内容：

- [ ] **敏感信息清理**：移除所有密码、API 密钥、私有 IP 地址、内部服务器地址
- [ ] **配置文件处理**：将敏感配置移至 `.gitignore`，提供示例配置文件（如 `config.example.json`）
- [ ] **内部脚本清理**：移除或忽略内部同步脚本、部署脚本等
- [ ] **代码注释检查**：移除不适合公开的注释内容
- [ ] **依赖项整理**：确保所有依赖都在 `requirements.txt` 或 `package.json` 中列出

### 1.2 文件命名规范

| 文件类型 | 推荐命名 | 说明 |
|---------|---------|------|
| 英文 README | `README.md` | 主文档，GitHub 默认显示 |
| 中文 README | `README_CN.md` 或 `README_zh.md` | 中文版本 |
| 许可证 | `LICENSE` 或 `LICENSE.txt` | 开源许可证 |
| 贡献指南 | `CONTRIBUTING.md` | 贡献者指南 |
| 变更日志 | `CHANGELOG.md` | 版本更新记录 |

---

## 2. 必备文件清单

### 2.1 最小必备文件

```
project/
├── README.md              # 项目说明文档（必须）
├── LICENSE                # 开源许可证（必须）
├── .gitignore             # Git 忽略配置（必须）
├── requirements.txt       # Python 依赖（如适用）
└── package.json           # Node.js 依赖（如适用）
```

### 2.2 推荐添加的文件

```
project/
├── README_CN.md           # 中文文档
├── CHANGELOG.md           # 变更日志
├── CONTRIBUTING.md        # 贡献指南
├── CODE_OF_CONDUCT.md     # 行为准则
├── docs/                  # 详细文档目录
│   └── ...
└── examples/              # 示例代码或配置
    └── config.example.json
```

---

## 3. .gitignore 配置规范

### 3.1 通用模板

```gitignore
# ==================== 敏感文件 ====================
# 配置文件（包含密码、API 密钥等）
config.json
*.config.local
.env
.env.local
*.pem
*.key

# 内部脚本
sync.sh
deploy.sh
internal/

# ==================== Python ====================
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
ENV/
.venv/
*.egg-info/
dist/
build/

# ==================== Node.js ====================
node_modules/
npm-debug.log
yarn-error.log

# ==================== IDE ====================
.idea/
.vscode/
*.swp
*.swo
*~
.project
.settings/

# ==================== OS ====================
.DS_Store
Thumbs.db
Desktop.ini

# ==================== 日志和临时文件 ====================
*.log
*.tmp
*.temp
logs/
temp/

# ==================== 模型和大文件 ====================
*.pth
*.ckpt
*.safetensors
*.bin
models/
weights/
```

### 3.2 注意事项

1. **先配置再 add**：在 `git add` 之前配置好 `.gitignore`
2. **已跟踪文件**：如果文件已被跟踪，需要先执行 `git rm --cached <file>`
3. **保留空目录**：在空目录中添加 `.gitkeep` 文件

---

## 4. README 文档写作规范

### 4.1 推荐结构

```markdown
# 项目名称

<!-- 徽章区域 -->
<p align="center">
  <img src="https://img.shields.io/badge/..." alt="badge">
</p>

<!-- 简短描述 -->
一句话描述项目功能和用途。

## ✨ 功能特点
- 功能 1
- 功能 2

## 📋 节点/API 说明（如适用）
| 名称 | 说明 |
|------|------|
| ... | ... |

## 🛠️ 安装指南
### 方法一：推荐方式
### 方法二：手动安装

## 📦 依赖/模型下载
### 依赖项
### 模型文件（如适用）

## 🚀 使用方法
### 基本使用
### 参数说明

## 💻 系统要求

## 📝 依赖项列表

## 🙏 致谢

## 📄 许可证

## ⚠️ 免责声明
```

### 4.2 写作要点

#### 标题和描述
- 使用清晰的项目名称
- 一句话说明项目用途
- 添加适当的徽章（License、Python 版本等）

#### 安装指南
- **多种安装方式**：提供包管理器和手动安装两种方式
- **完整命令**：给出可直接复制执行的命令
- **依赖说明**：明确列出所有依赖及版本要求

```markdown
## 🛠️ 安装指南

### 方法一：通过包管理器安装（推荐）
1. 安装 XXX Manager
2. 搜索 `项目名称`
3. 点击安装

### 方法二：手动安装
```bash
cd target_directory
git clone https://github.com/username/project.git
cd project
pip install -r requirements.txt
```
```

#### 模型/资源下载
- **表格展示**：使用表格清晰列出下载链接
- **目录结构**：说明文件应放置的位置
- **引用来源**：链接到官方来源

```markdown
## 📦 模型下载

参考 [官方指南](链接)：

| 模型 | 下载链接 | 说明 |
|------|----------|------|
| Model A | 🤗 [Huggingface](链接) | 说明 |
| Model B | 🤗 [Huggingface](链接) | 说明 |

### 目录结构
```
project/
└── models/
    ├── model_a.pth
    └── model_b.pth
```
```

#### 使用方法
- **参数表格**：使用表格说明参数
- **代码示例**：提供可运行的示例代码
- **截图/GIF**：如适用，添加效果展示

```markdown
## ⚙️ 参数说明

| 参数 | 说明 | 默认值 | 取值范围 |
|-----|------|--------|---------|
| size | 输出尺寸 | 512 | 256-1024 |
| steps | 采样步数 | 20 | 1-100 |
```

### 4.3 多语言支持

```markdown
# Project Name

[English](README.md) | [中文](README_CN.md)

...
```

### 4.4 常用徽章

```markdown
<!-- 许可证 -->
![License](https://img.shields.io/badge/License-Apache%202.0-green)

<!-- Python 版本 -->
![Python](https://img.shields.io/badge/Python-3.8+-blue)

<!-- 平台 -->
![Platform](https://img.shields.io/badge/Platform-ComfyUI-orange)

<!-- Stars -->
![Stars](https://img.shields.io/github/stars/username/repo)
```

---

## 5. Git 操作流程

### 5.1 首次上传流程

```bash
# 1. 进入项目目录
cd /path/to/project

# 2. 初始化 Git 仓库
git init

# 3. 配置 .gitignore（重要！在 add 之前）
# 编辑 .gitignore 文件

# 4. 添加所有文件
git add .

# 5. 检查状态，确认没有敏感文件
git status

# 6. 首次提交
git commit -m "Initial release: 项目简述"

# 7. 添加远程仓库
git remote add origin https://github.com/username/repo.git

# 8. 推送到 GitHub
git push -u origin main
```

### 5.2 从已有远程仓库推送到新仓库

```bash
# 添加新的远程仓库
git remote add github https://github.com/username/new-repo.git

# 推送本地分支到新仓库的 main 分支
git push -u github local-branch:main
```

### 5.3 常用 Git 命令

```bash
# 查看远程仓库
git remote -v

# 查看状态
git status

# 查看提交历史
git log --oneline

# 取消已暂存的文件
git restore --staged <file>

# 移除已跟踪但需忽略的文件
git rm --cached <file>

# 修改最后一次提交信息
git commit --amend -m "新的提交信息"
```

### 5.4 提交信息规范

推荐使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
<type>: <description>

[optional body]

[optional footer]
```

常用 type：
- `feat`: 新功能
- `fix`: 修复 bug
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建/工具相关

示例：
```bash
git commit -m "feat: add video face swapping support"
git commit -m "fix: resolve memory leak in model loader"
git commit -m "docs: update installation guide"
```

---

## 6. 开源许可证选择

### 6.1 常用许可证对比

| 许可证 | 特点 | 适用场景 |
|-------|------|---------|
| **MIT** | 最宽松，几乎无限制 | 希望代码被广泛使用 |
| **Apache-2.0** | 宽松，有专利保护 | 企业项目推荐 |
| **GPL-3.0** | 强制开源衍生作品 | 希望保持开源 |
| **BSD-3-Clause** | 类似 MIT，禁止用名义背书 | 学术项目 |

### 6.2 添加许可证

1. 在 GitHub 创建仓库时选择许可证
2. 或手动创建 `LICENSE` 文件，从 [choosealicense.com](https://choosealicense.com/) 复制内容

---

## 7. 项目维护建议

### 7.1 Issue 和 PR 模板

创建 `.github/ISSUE_TEMPLATE/bug_report.md`：

```markdown
---
name: Bug 报告
about: 报告一个 bug
---

**问题描述**
简要描述遇到的问题。

**复现步骤**
1. ...
2. ...

**期望行为**
描述期望的正确行为。

**环境信息**
- OS: 
- Python: 
- GPU: 

**日志/截图**
如有，请附上相关日志或截图。
```

### 7.2 版本发布

1. 更新 `CHANGELOG.md`
2. 更新版本号
3. 创建 Git tag：`git tag v1.0.0`
4. 推送 tag：`git push origin v1.0.0`
5. 在 GitHub 创建 Release

### 7.3 持续维护

- 及时回复 Issue
- 定期更新依赖
- 保持文档更新
- 添加 CI/CD（可选）

---

## 📝 检查清单

开源前的最终检查：

- [ ] `.gitignore` 已正确配置
- [ ] 无敏感信息（密码、密钥、内部地址）
- [ ] `README.md` 完整且准确
- [ ] 许可证文件存在
- [ ] 依赖文件存在且版本明确
- [ ] 模型/资源下载链接正确
- [ ] 代码可以正常运行
- [ ] 提交历史清晰

---

## 🔗 参考资源

- [GitHub 官方文档](https://docs.github.com/)
- [Choose a License](https://choosealicense.com/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Shields.io 徽章生成](https://shields.io/)
- [Markdown 指南](https://www.markdownguide.org/)

---

*文档版本：v1.0 | 更新日期：2026-01-16*
