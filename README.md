# FASTER

FASTER 是一个用 Rust 实现的高性能 TSN 时间触发业务流快速调度工具

## 安装

### Windows工具链配置指南

- 如已安装Windows Git, 可直接启动Git Bash并执行以下命令安装 Rust 工具链：`pacman -S mingw-w64-x86_64-rust`
- 否则：
  1. 安装msys2：https://www.msys2.org/
  2. 启动msys2，执行以下命令安装 Rust 工具链和git：`pacman -S mingw-w64-x86_64-rust git`
- 配置cargo镜像源：
  1. 创建并打开`%USERPROFILE%\.cargo\config.toml`
  2. 添加以下内容：
```toml
[source.crates-io]
replace-with = 'tuna'

[source.tuna]
registry = "https://mirrors.tuna.tsinghua.edu.cn/git/crates.io-index.git"
```
### 编译

```bash
git clone [repository-url]
cd faster
cargo build --release
```
## 使用方法

该工具支持以下命令行参数：

### JSON 输入模式

```bash
./faster --input-json <input-file.json> --output-console
```

### INET 输入模式

```bash
./faster --input-inet <input-file> [--sequence <sequence-file>] --output-inet <output-file>
```

### FAST 输入模式

```bash
./faster --input-fast <device-txt> <flow-txt> <flowlink-txt> --output-console
```

## 输入格式

### JSON 格式
JSON 输入文件需要包含以下主要字段：
- devices: 网络设备信息
- links: 链路信息
- flows: 流量信息

### INET 格式
支持标准的 INET 网络拓扑描述格式，可选择性地包含业务流时序信息。

## 项目结构

```
src/
├── main.rs       # 主程序入口
├── model.rs      # 核心数据模型
├── input_json.rs # JSON 输入处理
├── input_inet.rs # INET 输入处理
└── output_inet.rs # INET 输出生成
```
