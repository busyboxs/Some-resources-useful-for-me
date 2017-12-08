# 使用git上传文件或文件夹到github repository

----

## 通过SSH连接GitHub

使用SSH协议，您可以连接并验证远程服务器和服务。使用SSH密钥，您可以连接到GitHub，而无需在每次访问时提供您的用户名或密码。

### **生成一个新的SSH密钥并将其添加到ssh-agent**

在检查了现有的SSH密钥后，可以生成一个新的SSH密钥用于身份验证，然后将其添加到ssh-agent。

如果您还没有SSH密钥，则必须生成一个新的SSH密钥。如果您不确定是否已有SSH密钥，请检查现有密钥。

#### **检查现有的SSH密钥**

1. 打开终端;

2. 输入`ls -al ~/.ssh`以查看是否存在现有的SSH密钥：

```
$ ls -al ~/.ssh
# Lists the files in your .ssh directory, if they exist
```

3. 检查目录列表，看看你是否已经有一个公共的SSH密钥。

默认情况下，公钥的文件名是以下之一：

* id_dsa.pub
* id_ecdsa.pub
* id_ed25519.pub
* id_rsa.pub

如果您没有现有的公钥和私钥对，或者不希望使用任何可用于连接到GitHub的公钥和私钥，请生成一个新的SSH密钥。如果您看到列出的现有公钥和私钥对（例如id_rsa.pub和id_rsa），那么您可以将SSH密钥添加到ssh-agent中。

#### **生成一个新的SSH密钥**

1.  打开终端;

2.  粘贴下面的文本，替换您的GitHub电子邮件地址。这将创建一个新的ssh密钥，使用提供的电子邮件作为标签。

```
$ ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

```powershell
Generating public/private rsa key pair.
```

3. 当系统提示您输入要保存密钥的文件时，请按Enter键。这接受默认的文件位置。

```powershell
Enter a file in which to save the key (/home/you/.ssh/id_rsa): [Press enter]
```

4. 在提示符下，输入安全密码。有关更多信息，请参阅“[使用SSH密钥密码](https://help.github.com/articles/working-with-ssh-key-passphrases)”。

```powershell
Enter passphrase (empty for no passphrase): [Type a passphrase]
Enter same passphrase again: [Type passphrase again]
```

#### **将您的SSH密钥添加到ssh-agent**

在将新的SSH密钥添加到ssh-agent来管理您的密钥之前，您应该检查现有的SSH密钥并生成一个新的SSH密钥。

1. 在后台启动ssh-agent

```powershell
$ eval "$(ssh-agent -s)"
Agent pid 59566
```

2.  将你的SSH私钥添加到ssh-agent。如果使用不同的名称创建密钥，或者要添加具有不同名称的现有密钥，请将命令中的id_rsa替换为私钥文件的名称。

```powershell
$ ssh-add ~/.ssh/id_rsa
```

3. 将SSH密钥复制到剪贴板。如果您的SSH密钥文件与示例代码名称不同，请修改文件名以匹配您当前的设置。复制密钥时，请勿添加任何换行符或空格。

```powershell
sudo apt-get install xclip
# Downloads and installs xclip. If you don't have `apt-get`, you might need to use another installer (like `yum`)

$ xclip -sel clip < ~/.ssh/id_rsa.pub
# Copies the contents of the id_rsa.pub file to your clipboard
```

>提示：如果xclip不起作用，您可以找到隐藏的.ssh文件夹，在您喜欢的文本编辑器中打开该文件，并将其复制到剪贴板。

4. 在任何页面的右上角，点击你的个人资料照片，然后点击  **Settings**。

![](https://help.github.com/assets/images/help/settings/userbar-account-settings.png)

5. 在用户设置侧栏中，单击 **SSH and GPG keys**。

![](https://help.github.com/assets/images/help/settings/settings-sidebar-ssh-keys.png)

6. 单击 **New SSH key** 或 **Add SSH key**

![](https://help.github.com/assets/images/help/settings/ssh-add-ssh-key.png)

7. 在“Title”字段中，为新密钥添加一个描述性标签。例如，如果您使用的是个人Mac，则可以将其称为“Personal MacBook Air”。

8. 将您的密钥粘贴到“Key”字段中

![](https://help.github.com/assets/images/help/settings/ssh-key-paste.png)

9. 点击 **Add SSH key**

![](https://help.github.com/assets/images/help/settings/ssh-add-key.png)

10. 如果出现提示，请确认您的GitHub密码。

![](https://help.github.com/assets/images/help/settings/sudo_mode_popup.png)


#### **测试你的SSH连接**

在你设置你的SSH密钥并将其添加到你的GitHub帐户后，你可以测试你的连接。在测试连接时，您需要使用您的密码（这是您之前创建的SSH密钥）验证此操作。

1. 打开终端。

2. 输入以下内容：

```powershell
$ ssh -T git@github.com
# Attempts to ssh to GitHub
```

您可能会看到以下警告之一：

```powershell
The authenticity of host 'github.com (192.30.252.1)' can't be established.
RSA key fingerprint is 16:27:ac:a5:76:28:2d:36:63:1b:56:4d:eb:df:a6:48.
Are you sure you want to continue connecting (yes/no)?
```

```powershell
The authenticity of host 'github.com (192.30.252.1)' can't be established.
RSA key fingerprint is SHA256:nThbg6kXUpJWGl7E1IGOCspRomTxdCARLviKw6E5SY8.
Are you sure you want to continue connecting (yes/no)?
```

>注意：上面的例子列出了GitHub的IP地址192.30.252.1。在ping GitHub的时候，你可能会看到一系列的IP地址。有关更多信息，请参阅“[GitHub使用哪些IP地址作为白名单？](https://help.github.com/articles/what-ip-addresses-does-github-use-that-i-should-whitelist)”

3. 确认您看到的消息中的 fingerprint 与步骤2中的其中一条消息相匹配，然后输入`yes`：

```powershell
Hi username! You've successfully authenticated, but GitHub does not
provide shell access.
```

您可能会看到以下错误消息：

```
Agent admitted failure to sign using the key.
debug1: No more authentication methods to try.
Permission denied (publickey).
```

这是某些Linux发行版的已知问题。有关更多信息，请参阅“[错误：代理承认失败签名](https://help.github.com/articles/error-agent-admitted-failure-to-sign)”。

4. 验证结果消息是否包含您的用户名。如果收到“权限被拒绝”消息，请参阅“[错误：权限被拒绝（公钥）](https://help.github.com/articles/error-permission-denied-publickey)”。

----

## 上传文件到github

### 1.创建一个新的存储库

您可以在您的个人帐户或拥有足够权限的任何组织中创建新的存储库。

1. 在任何页面的右上角，单击 **+**，然后单击 **New repository**

![](https://help.github.com/assets/images/help/repository/repo-create.png)

2.  在“所有者”下拉列表中，选择您想要创建存储库的帐户。

![](https://help.github.com/assets/images/help/repository/create-repository-owner.png)

3.  为您的存储库键入一个名称，以及一个可选的说明。

![](https://help.github.com/assets/images/help/repository/create-repository-name.png)

4. 您可以选择使存储库公开或私有。公共存储库对公众是可见的，而私人存储库只有你和你分享的人才能访问。您的帐户必须是付费才能创建私人存储库。

5. 有许多可选的项目，您可以预先使用您的存储库。如果您要将现有的存储库导入GitHub，请不要选择这些选项中的任何一个，因为您可能会引入合并冲突。您可以选择稍后使用命令行添加这些文件。

>* 您可以创建一个[README](https://help.github.com/articles/about-readmes/)，它是描述项目的文档。
>* 您可以创建一个[CODEOWNERS](https://help.github.com/articles/about-codeowners/)文件，该文件描述哪些个人或团队拥有存储库中的某些文件。
>* 您可以创建一个.gitignore文件，这是一组[忽略规则](https://help.github.com/articles/ignoring-files/)。
>* 您可以选择为您的项目[添加软件许可证](https://help.github.com/articles/licensing-a-repository/)。

6. 完成后，点击 **Create repository**。

7. 在 Quick Setup page 页面底部的 “Import code from an old repository” 下，可以选择将项目导入到新的资源库。为此，请点击 **Import code**。

### 2.将文件添加到存储库

你可以上传并提交一个现有的文件到GitHub仓库。将文件拖放到文件树中的任意目录，或者从存储库的主页面上传文件。

您通过浏览器添加到存储库的文件被限制为每个文件25 MB。您可以通过命令行添加更大的文件，每个文件最多100 MB。

>提示：
>* 您可以同时将多个文件上传到GitHub。 
>* 如果存储库具有任何受保护的分支，则无法使用Web界面在受保护的分支中编辑或上载文件。

1. 在GitHub上，导航到存储库的主页面。

2. 在您的存储库名称下，单击 **Upload files**。

![](https://help.github.com/assets/images/help/repository/upload-files-button.png)

3. 将要上传到存储库的文件或文件夹拖放到文件树上。

![](https://help.github.com/assets/images/help/repository/upload-files-drag-and-drop.png)

4. 在页面的底部，键入一个简短而有意义的提交消息，描述您对该文件所做的更改。

![](https://help.github.com/assets/images/help/repository/write-commit-message-quick-pull.png)

5. 在提交消息字段下方，决定是否将提交添加到当前分支或新分支。如果你当前的分支是`master`，你应该选择为你的分支创建一个新的分支，然后创建一个pull请求。

![](https://help.github.com/assets/images/help/repository/choose-commit-branch.png)

6. 点击 **Commit changes**.

![](https://help.github.com/assets/images/help/repository/commit-changes-button.png)


### 3.使用命令行将文件添加到存储库

您可以使用命令行将现有文件上传到GitHub存储库。

此过程假定您已经：

* 在GitHub上创建一个存储库，或者让一个你想要贡献的其他人拥有一个现有的存储库
* 在您的计算机上本地克隆存储库

>Warning: Never git add, commit, or push sensitive information to a remote repository. Sensitive information can include, but is not limited to:
>* Passwords
>* SSH keys
>* AWS access keys
>* API keys
>* Credit card numbers
>* PIN numbers


1. 在你的计算机上，将你想要上传到GitHub的文件移动到克隆存储库时创建的本地目录中。

2. 打开终端。

3. 将当前工作目录更改为本地存储库。

4. 将要上传的文件提交到本地存储库。

```powershell
$ git add .
# Adds the file to your local repository and stages it for commit. To unstage a file, use 'git reset HEAD YOUR-FILE'.
```

5. 添加注释

```powershell
$ git commit -m "Add existing file"
# Commits the tracked changes and prepares them to be pushed to a remote repository. To remove this commit and modify the file, use 'git reset --soft HEAD~1' and commit and add the file again.
```

6. 将本地存储库中的更改推送到GitHub。

```powershell
$ git push origin your-branch
# Pushes the changes in your local repository up to the remote repository you specified as the origin
```

### 4.Pushing to a remote

使用`git push`将本地分支上的提交推送到远程存储库。 

`git push`命令有两个参数： 
* 远程名称，例如，`origin`
* 分支名称，例如，`master`

例如：

```powershell
git push  <REMOTENAME> <BRANCHNAME> 
```

举个例子，你通常运行`git push origin master`来把你的本地修改推送到你的在线仓库。

#### 重命名分支

重命名一个分支，你可以使用相同的`git push`命令，但是你可以添加一个参数：新分支的名字。例如：

```powershell
git push  <REMOTENAME> <LOCALBRANCHNAME>:<REMOTEBRANCHNAME> 
```

这会将`LOCALBRANCHNAME`推送到您的`REMOTENAME`，但将其重命名为`REMOTEBRANCHNAME`。


### 5.使用命令行将现有项目添加到GitHub

把你现有的工作放在GitHub上，可以让你分享和协作。

1. 在GitHub上创建一个新的存储库。为避免错误，请勿使用README，许可证或`gitignore`文件初始化新的存储库。您的项目被推送到GitHub后，您可以添加这些文件。

![](https://help.github.com/assets/images/help/repository/repo-create.png)

2. 打开终端。

3. 将当前工作目录更改为您的本地项目。

4. 将本地目录初始化为Git存储库。

```powershell
$ git init
```

5. 将文件添加到新的本地存储库中。这为他们的第一次提交阶段。

```powershell
$ git add .
# Adds the files in the local repository and stages them for commit. To unstage a file, use 'git reset HEAD YOUR-FILE'.
```

6. 提交您在本地存储库中分发的文件。(添加注释)

```powershell
$ git commit -m "First commit"
# Commits the tracked changes and prepares them to be pushed to a remote repository. To remove this commit and modify the file, use 'git reset --soft HEAD~1' and commit and add the file again.
```

7. 在你的GitHub仓库的快速设置页面的顶部，点击复制远程仓库的URL。

![](https://help.github.com/assets/images/help/repository/copy-remote-repository-url-quick-setup.png)

8. 在终端中，添加本地存储库将被推送到的远程存储库的URL。

```powershell
$ git remote add origin remote repository URL
# Sets the new remote
$ git remote -v
# Verifies the new remote URL
```

9. 将本地存储库中的更改推送到GitHub。

```powershell
$ git push origin master
# Pushes the changes in your local repository up to the remote repository you specified as the origin
```
