# Install Mysql8 with c++ connection in Ubuntu 18.04

## 1. Install Mysql 8

- First you need to update your system:

```bash
sudo apt-get update
sudo apt-get upgrade
```

- Install MySQL 8 by adding APT Repository. 

Download MySQL APT Repository from: MySQL APT Repository (Click Download on the bottom right, then copy the link on the next page ( from No thanks, just start my download)). Sample code:

```
wget https://dev.mysql.com/get/mysql-apt-config_0.8.13-1_all.deb 
```

- Install the APT by dpkg

At this step you need to select what to be installed and press OK.

```
sudo dpkg -i mysql-apt-config_0.8.13-1_all.deb 
```

- Install the MySQL Server 8

```
sudo apt-get update
sudo apt-get install mysql-server
sudo apt-get install mysql-client
sudo apt-get install libmysqlclient-dev
```

During installation you will be asked for root password. Select a good one and continue installation

## 2.  Install C++ connection for mysql

```
sudo apt-get install libmysqlcppconn-dev
```

如果遇到如下错误，

```
下列软件包有未满足的依赖关系：
 libmysqlcppconn-dev : 依赖: libmysqlcppconn7 (= 8.0.17-1ubuntu18.04) 但是它将不会被安装
E: 无法修正错误，因为您要求某些软件包保持现状，就是它们破坏了软件包间的依赖关系。
```

则需要手动安装依赖库，执行下面代码

```
sudo apt-get install libmysqlcppconn7
```

## 3. use Mysql create a database for test

run

```
mysql -u root -p
```

and input your mysql password(set during install)

then run following code to create database and table to used next.

```
CREATE DATABASE testdb;
USE testdb;
CREATE TABLE test (id int, name varchar(32), score int);
INSERT INTO test (id, name, score) VALUES (1, "Marianne", 89);
INSERT INTO test (id, name, score) VALUES (2, "Jimmy", 62);
INSERT INTO test (id, name, score) VALUES (3, "Ling", 78);
SELECT * FROM test;
```
So we create a database named "testdb" and a table named "test" in it.

## 4.  configure CMakeLists to run code

In your CMakeLists.txt file, code should be like follow. We add `target_link_libraries`.

```
cmake_minimum_required(VERSION 3.14)
project(mysql_pros)

set(CMAKE_CXX_STANDARD 14)

add_executable(mysql_pros main.cpp)
target_link_libraries(mysql_pros mysqlcppconn mysqlclient)
```

## 5. mysql with c++ in clion

```
#include <iostream>
#include <mysql/mysql.h>

using namespace std;
int qstate;

int main() {
    MYSQL* conn;
    MYSQL_ROW row;
    MYSQL_RES *res;
    conn = mysql_init(0);
    conn = mysql_real_connect(conn, "localhost", "root", "password", "testdb", 3306, NULL, 0);

    if(conn) {
        puts("Successful connection to database!");

        string query = "select * from test";
        const char *q = query.c_str();
        qstate = mysql_query(conn, q);
        if(!qstate) {
            res = mysql_store_result(conn);
            while(row = mysql_fetch_row(res)) {
                cout << "ID: " << row[0] << ", ";
                cout << "Name: " << row[1] << ", ";
                cout << "Score: " << row[2] << endl;
            }
        }else {
            cout << "Query failed: " << mysql_error(conn) << endl;
        }
    }else {
        puts("Connection to database has faild!");
    }
    return 0;
}
```

After run , you will get result below:

```
Successful connection to database!
ID: 1, Name: Marianne, Score: 89
ID: 2, Name: Jimmy, Score: 62
ID: 3, Name: Ling, Score: 78
```
