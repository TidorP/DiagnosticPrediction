from Diagnostic.LogisticRegression import *

class UI:
    def __init__(self,fileIn,fileTest):
        self.fileIn=fileIn
        self.fileTest=fileTest

    def printCommands(self):
        print("1: Logistic regression \n")
        print("0: Exit")
    def getFromFile(self):
        i=-1
        l = []
        out = []
        with open(self.fileIn, "r") as filestream:
            for line in filestream:
                i = i + 1
                if (i == 300):
                    break
                li = []
                currentline = line.split(" ")
                for j in range(6):
                    li.append(float(currentline[j]))
                l.append(li)
                if(currentline[6][:-1]=='NO'):
                    out.append(0)
                else:
                    out.append(1)
        return [l,out]
    def getFromFileTest(self):
        in_test = []
        out_test = []
        with open(self.fileTest, "r") as filestream:
            for line in filestream:
                li = []
                currentline = line.split(" ")
                for j in range(6):
                    li.append(float(currentline[j]))
                in_test.append(li)
                out_test.append(currentline[6][:-1])
        out_test[9] = 'NO'
        for i in range(len(out_test)):
            if(out_test[i]=='NO'):
                out_test[i]=0
            else:
                out_test[i]=1
        return [in_test,out_test]

    def getStreamOut(self):
        l = self.getFromFile()[1]
        stream = []
        for x in l:
            stream.append([x])
        return stream

    # we run our method here
    def runMethod1(self):
        print("The liniear model has been solved\n")
        new_in = self.getFromFileTest()[0]
        new_out = self.getFromFileTest()[1]
        for index in range(len(new_in)):
            m = LG(self.getFromFile()[0], self.getFromFile()[1])
            print("We wil introduse this new input: \n")
            print(new_in[index])
            res = m.model([new_in[index]])
            print("Our prediction:\n")
            if(res==1):
                print("AB")
            else:
                print("NO")
            print("Real result: ")
            if(new_out[index]==0):
                print("NO")
            else:
                print("AB")

    def run(self):
        print("Welcome to our ML Algorithm for predicting if a patient with specific attributes has disease or not! \n")
        print("The entity has the atributes: ")
        print("Pelvic incidence, pelvic tilt, lumbar lordosis angle, sacral slope, pelvic radius and grade of spondylolisthesis.")
        print("Base on this we will find a model that can deduce the diagnostic: abnormal or not\n")
        while(True):
            self.printCommands()
            i = input('Enter command: ')
            if(i=='0'):
                print("Bye")
                break
            if(i=='1'):
                self.runMethod1()
            break

fileI="column_2C.dat"
fileT="test.dat"
ui=UI(fileI,fileT)
ui.run()
