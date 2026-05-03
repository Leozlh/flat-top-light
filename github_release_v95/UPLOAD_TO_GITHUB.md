# Upload To GitHub

## 方式一：Git 本地上传（推荐）

### 1. 进入文件夹

```powershell
cd "D:\Trae products\flat_top light\github_release_v95"
```

### 2. 初始化 Git 仓库

```powershell
git init
git config user.name "你的GitHub用户名"
git config user.email "你的GitHub邮箱"
```

### 3. 提交

```powershell
git add .
git commit -m "v95 release: Pareto-knee pipeline for flat-top holography"
```

### 4. 在 GitHub 新建仓库

建议仓库名：`flat-top-light` 或 `holography-flat-top-v95`

创建时不要勾选 README / .gitignore / license（本地已有）。

### 5. 推送

```powershell
git remote add origin https://github.com/你的用户名/你的仓库名.git
git branch -M main
git push -u origin main
```

## 方式二：网页上传

1. GitHub 新建空仓库
2. 选择 `uploading an existing file`
3. 把 `github_release_v95` 内容拖入
4. 提交

## 上传前检查

- [ ] 没有泄露账号、密钥、私有路径
- [ ] 没有超大 checkpoint 文件
- [ ] notebook 输出已清理（可选：`Cell > All Output > Clear`）
- [ ] .npy 文件在 50MB 以下

## 上传后建议

1. 在仓库首页补充项目简介
2. 检查 `docs/` PDF 能否正常下载
3. 说明当前版本是 v95
4. 后续更新维护 `CHANGELOG.md`
