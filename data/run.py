import os
os.system("cp /Users/factorialn/OtherProjects/pbrt-v2/src/bin/pbrt pbrt")
os.system("rm trainingdata/*")
os.system("./pbrt prt-teapot.pbrt")
os.system("g++ modify.cpp -o modify")
os.system("rm Data/*")
os.system("./modify")