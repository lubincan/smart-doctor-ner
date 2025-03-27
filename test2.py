import fileinput
import sys
#sys.path.append("/data/django-uwsgi/text_labeled/")
#sys.path.append("D:\\pythonwork\\django-uwsgi\\text_labeled\\")
'''添加相对路径'''
import os
print(sys.version_info)
print(sys.executable)
sys.path.append("D:\\pythonwork\\django-uwsgi\\text_labeled\\")
stopdict_path = os.path.join(os.path.dirname(__file__), "stopdict.txt")
#stop_word_set = set(map(lambda x: x.strip(), fileinput.FileInput(stopdict_path, encoding='utf-8')))
#print(stop_word_set)
import h5py