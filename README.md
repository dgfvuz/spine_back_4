### spine_back
这是脊柱X光系统的后端

### run
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

### createsuperuser
```sh
python manage.py createsuperuser
```
### migrate
```sh
python manage.py makemigrations

python manage.py migrate
```

### apps:
user(已完成) ```python manage.py startapp user```

patient(已完成) ```python manage.py startapp patient```

report(已完成) ```python manage.py startapp report```

collect(待完成) ```python manage.py startapp collect```

advice(已完成) ```python manage.py startapp advice```