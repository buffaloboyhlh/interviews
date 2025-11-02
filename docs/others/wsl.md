# Windows WSL 详细教程

## 什么是 WSL？

Windows Subsystem for Linux（WSL）是微软在 Windows 10/11 中推出的功能，允许用户在 Windows 上直接运行 Linux 环境，无需虚拟机或双系统。

## WSL 1 vs WSL 2

| 特性 | WSL 1 | WSL 2 |
|------|-------|-------|
| 架构 | 转换层 | 完整 Linux 内核 |
| 性能 | 文件系统操作较慢 | 文件系统操作快 |
| 兼容性 | 较好 | 完全系统调用兼容 |
| 启动速度 | 快 | 稍慢 |

## 安装 WSL

### 方法一：简单安装（推荐）
```powershell
# 以管理员身份运行 PowerShell
wsl --install
```
此命令会：
- 启用 WSL 功能
- 安装默认的 Ubuntu 发行版
- 安装 WSL 2 内核

### 方法二：分步安装
```powershell
# 启用 WSL 功能
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

# 启用虚拟机平台
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# 重启计算机
```

### 安装特定 Linux 发行版
```powershell
# 查看可用的发行版
wsl --list --online

# 安装特定发行版
wsl --install -d Ubuntu-22.04
wsl --install -d Debian
wsl --install -d Kali-Linux
```

## 基本使用

### 启动 WSL
```powershell
# 启动默认发行版
wsl

# 启动特定发行版
wsl -d Ubuntu-22.04

# 以特定用户启动
wsl -u username
```

### 在 WSL 中操作
```bash
# 更新软件包
sudo apt update && sudo apt upgrade

# 安装软件
sudo apt install git python3 nodejs

# 查看 WSL 信息
cat /etc/os-release

# 访问 Windows 文件
cd /mnt/c/Users/YourUsername
```

## WSL 配置

### 查看 WSL 状态
```powershell
# 查看已安装的发行版
wsl --list --verbose

# 查看 WSL 版本
wsl --status
```

### 设置默认 WSL 版本
```powershell
# 设置 WSL 2 为默认版本
wsl --set-default-version 2
```

### 转换发行版版本
```powershell
# 将发行版转换为 WSL 2
wsl --set-version Ubuntu-22.04 2

# 将发行版转换为 WSL 1
wsl --set-version Ubuntu-22.04 1
```

## 文件系统互操作

### 在 WSL 中访问 Windows 文件
```bash
# Windows C 盘挂载在 /mnt/c
cd /mnt/c/Users/YourUsername

# 在 WSL 中运行 Windows 程序
/mnt/c/Windows/System32/notepad.exe
```

### 在 Windows 中访问 WSL 文件
```
# 在文件资源管理器中访问
\\wsl$\Ubuntu-22.04\home\username

# 或在 WSL 中打开 Windows 文件管理器
explorer.exe .
```

## 网络功能

### 端口转发
WSL 2 有独立的 IP，但 Windows 会自动转发端口。

```bash
# 在 WSL 中启动 web 服务器
python3 -m http.server 8000

# 在 Windows 浏览器中访问
# http://localhost:8000
```

## 高级配置

### WSL 配置文件
创建 `%UserProfile%\.wslconfig` 文件：

```ini
[wsl2]
memory=4GB
processors=2
localhostForwarding=true
```

### 导出和导入发行版
```powershell
# 导出发行版
wsl --export Ubuntu-22.04 Ubuntu-backup.tar

# 导入发行版
wsl --import Ubuntu-new .\Ubuntu-new\ Ubuntu-backup.tar
```

### 卸载发行版
```powershell
# 注销并删除发行版
wsl --unregister Ubuntu-22.04
```

## 开发环境配置

### 安装开发工具
```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装开发工具
sudo apt install build-essential git curl wget

# 安装 Node.js
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# 安装 Python 开发环境
sudo apt install python3-pip python3-venv
```

### 配置 VS Code 与 WSL
1. 安装 VS Code 的 "Remote - WSL" 扩展
2. 在 WSL 终端中输入 `code .`
3. VS Code 将在 WSL 环境中打开

## 常见问题解决

### 1. WSL 安装失败
```powershell
# 重置 WSL
wsl --shutdown
wsl --unregister Ubuntu-22.04
wsl --install
```

### 2. 网络连接问题
```powershell
# 重启 WSL 服务
wsl --shutdown
```

### 3. 文件权限问题
```bash
# 在 WSL 中修复 Windows 文件权限
sudo chmod -R 755 /mnt/c/your/project/path
```

## 性能优化技巧

1. **将项目文件放在 WSL 文件系统中** 而不是 Windows 文件系统
2. **使用 WSL 2** 获得更好的性能
3. **合理配置内存和 CPU 限制** 在 `.wslconfig` 中
4. **定期更新** WSL 和 Linux 发行版

## 实用命令总结

```powershell
# WSL 管理命令
wsl --list --verbose          # 列出所有发行版
wsl --shutdown               # 关闭所有 WSL 实例
wsl --terminate <Distro>     # 终止特定发行版
wsl --set-version <Distro> 2 # 设置版本

# 在 WSL 中
wsl --update                # 更新 WSL 内核
wsl --status                # 查看 WSL 状态
```

这个教程涵盖了 WSL 的主要功能和使用方法。WSL 为开发者和系统管理员提供了强大的 Linux 环境，同时保持了 Windows 的便利性。