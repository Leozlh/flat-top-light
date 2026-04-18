# Upload To GitHub

下面给你两种方式。推荐用第一种：本地 `git` 上传，后续更新最方便。

## 方式一：用 Git 本地上传

### 1. 进入整理好的文件夹

在 PowerShell 里执行：

```powershell
cd "D:\Trae products\flat_top light\github_release_20260418"
```

### 2. 初始化一个新的 Git 仓库

```powershell
git init
```

### 3. 把文件加入版本管理

```powershell
git add .
git commit -m "Initial curated release"
```

如果第一次提交时报用户名或邮箱未设置，就先执行：

```powershell
git config user.name "你的GitHub用户名"
git config user.email "你的GitHub邮箱"
```

然后再重新执行：

```powershell
git add .
git commit -m "Initial curated release"
```

### 4. 在 GitHub 网站新建仓库

去 GitHub 新建一个仓库，建议名字例如：

- `flat-top-light`
- `holography-flat-top-light`
- `flat-top-light-release`

创建时建议：

- 不要勾选 `Add a README file`
- 不要勾选 `.gitignore`
- 不要勾选 `license`

因为这些内容我们已经在本地准备了。

### 5. 把本地仓库连到 GitHub 远程仓库

把下面的地址替换成你自己的仓库地址：

```powershell
git remote add origin https://github.com/你的用户名/你的仓库名.git
```

### 6. 推送到 GitHub

```powershell
git branch -M main
git push -u origin main
```

上传完成后，刷新 GitHub 页面就能看到内容。

## 方式二：直接网页上传

如果你不想用命令行，也可以：

1. 在 GitHub 新建一个空仓库
2. 进入仓库页面
3. 选择 `uploading an existing file`
4. 把 `github_release_20260418` 里的内容整体拖进去
5. 填写提交说明，例如 `Initial curated release`
6. 点击提交

这种方式适合第一次快速上传，但后续反复更新会不如 `git` 方式方便。

## 上传前建议检查

上传前你最好确认这几件事：

1. 没有把不想公开的账号、路径、密钥、私有数据放进去
2. 没有把超大的 checkpoint 或运行输出一起放进去
3. notebook 和文档里没有需要隐藏的个人信息
4. 引用的第三方内容可以公开展示

## 上传后建议做的事

1. 在 GitHub 仓库首页补一段项目简介
2. 打开 `docs/` 里的 PDF 检查是否能正常下载
3. 在仓库首页说明当前推荐 notebook 是 `v58`
4. 如果后面还会继续更新，建议单独维护一个 `CHANGELOG.md`

## 我建议你的实际做法

最稳妥的方式是：

1. 先上传这个整理好的文件夹
2. 确认 GitHub 展示效果正常
3. 再决定要不要继续补环境依赖文件、许可证和更正式的首页介绍
