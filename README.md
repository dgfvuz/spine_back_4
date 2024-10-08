### spine_back
这是脊柱X光系统的后端, 在linux系统上运行, 采用python版本为3.11.9, 文件依赖项已经导出到requirements.txt
建议使用anaconda虚拟环境运行, 或者使用docker来构建容器运行

### 如何启动
启动方法如下, 在manage.py文件所在目录下执行
```sh
python manage.py runserver
```
默认在127.0.0.1:8000启动服务器

如果需要在本机ip地址上启动服务器,则需要通过
```sh
ipconfig
```
命令查看当前主机的网络ip

以当前网络ip为192.168.31.241为例子

则需要在命令行下执行
```sh
python manage.py runserver 192.168.31.241:8000
```

出现以下信息则是服务器启动成功

```
Watching for file changes with StatReloader
Performing system checks...
System check identified no issues (0 silenced).
June 25, 2024 - 15:06:19
Django version 5.0.6, using settings 'app.settings'
Starting development server at http://192.168.31.241:8000/
Quit the server with CTRL-BREAK.
```

服务器接口文档查看方法: 
启动服务器以后再浏览器中输入
http://localhost:8000/docs/

### createsuperuser 服务端创建超级用户(管理员)的方法
```sh
python manage.py createsuperuser
```
### migrate 进行数据库迁移的方法: 可以提供一种对后端数据库无痛更改的方法, 即更改数据库的表的结构, 但是不改变原来的数据和原来的功能
```sh
python manage.py makemigrations

python manage.py migrate
```

### 必须要做的内容 apps: 这边是app, 每一个app对应一个数据库的表格, 并且定义了各种api,可以通过查阅接口文档得到结果
user(已完成) ```python manage.py startapp user``` 主要完成用户登录,token,鉴权等功能

patient(已完成) ```python manage.py startapp patient``` 主要完成病人管理的功能

report(已完成) ```python manage.py startapp report``` 主要完成报告生成的功能(这也是系统的主要功能)

collect(已完成) ```python manage.py startapp collect``` 收藏功能, 使得用户能够收藏患者

advice(已完成) ```python manage.py startapp advice``` 建议功能, 使得用户能够向系统开发这提供建议, 管理员能够查看所有建议

### 可以做但是没有要求做的内容:
email (未完成) 定期发送消息至邮箱(可以是患者的信息, 每个月的月报, 以及将用户提交的建议发送给开发者邮箱)

数据集生成 (未完成) 由于这个系统实际上是给算法搭建一个平台, 那么我们可以将用户审核的结果自动生成数据集反馈到开发者, 或者在服务器定期训练新数据集, 提高模型的性能

模型自动训练, 自动替换最佳权重(未完成) 在服务器生成数据集后模型定期自动训练, 自动更新权重

日志 (未完成) 服务端保存用户操作日志, 管理员查看日志

服务端错误自动重启 (未完成), 在本地跑的时候由于电脑性能不够， 模型预测时会导致后端程序直接结束, 所以需要一个能够保存错误, 自动重启的功能(或许用docker可以实现?)




#### 2024.10.8更新日志
在report选项添加了一个type字段, 在上传报告的时候需要包含report_type字段, 否则会默认为冠状位