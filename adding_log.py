#!/usr/bin/python3
"""
Author:finger
Time:20180802
func:给每一行代码添加代码分析日志
"""
FILE_PATH ="./test_log.code"
"""
日志函数:
def log_message(a,b):
	print(a,b)
"""
code_str="""
root = settings.TEMPLATES[0]["DIRS"][index]
    if path.endswith(".html") or path.endswith(".o"):
        from django.template.loader import render_to_string
        context = {"DEBUG": settings.DEBUG}
        advert_id = request.GET.get("advert", "")
        if advert_id.isdigit():
            from main.models import Advert
            advert = Advert.objects.get(id=advert_id)
"""
import os

def handle(hand):
    def get_block_py(s):
        for i in s.split(" "):
            if i!=" ":
                return i
        return ""
        pass
    def get_impt_block_py(s):
        for i in s.split(" "):
            if i=="from" or i=="import" or i=="def":
                continue
            return i
        return " "
        pass
    def compu_space(strs):
        for i in range(len(strs)):
            if not strs[i].isspace():
                return i
        return 0
        pass
    def get_space_str(i):
        if i==0:return "    "
        res=""
        for j in range(i):
            res+=" "
        return res
        pass
    
    with open(os.path.basename(FILE_PATH)+".rsf","a+",encoding="utf-8") as fw:
        with open(FILE_PATH,"r",encoding="utf-8") as f:
            line = f.readline()
            print(line)
            add_line=""
            while line:
                line=line.replace("\n","")
                space_str=get_space_str(compu_space(line))
                if line.find("from") > 0 or line.find("import") > 0:
                    add_line=space_str+"log_message(\'导入 %s 新模块\",\'%s\')"%(get_impt_block_py(line),str(line))
                elif line.find("def")>0:
                    add_line=space_str+"log_message(\'进入 %s 函数 \',\'\')"%(get_impt_block_py(line))
                elif line.find(":")>0 and line.split(":")[1]=="":
                    add_line=space_str+"    log_message(\'进入%s代码块\',\'%s\')"%(get_block_py(line),str(line))
                elif line.find(":")>0 and line.split(":")[1]!="":
                    add_line=space_str+"log_message(\'执行 %s \',\'\')"%(str(line))
                elif line.find("=")>0 and line.find("==")<0:
                    code = str(line[:line.find("=")])
                    add_line=space_str+"log_message(\'%%s\',\'%%s=%%s\')%%(str(\'%s\'),str(\'%s\'),str(%s))"%(str(line),code,code)
                elif line.find("log_message")>0:
                    add_line=""
                else :
                    add_line=space_str+"    #log_message(\'执行代码块\',\'%s\')"%(str(line))
                fw.write("\n")
                fw.write(line)
                if add_line!="":fw.write("\n")
                fw.write("%s"%(add_line))
                print(line)
                line=f.readline()

if __name__=="__main__":
    handle(1)
            