import numpy as np
import math

class LG:
    def __init__(self,dataIn,dataOut): #constructor
        self.dataIn=dataIn
        self.dataOut=dataOut
        self.normalized_dataIn=self.normalizeDataIn()[0]
    def normalizeDataIn(self):
        list_means=[]
        noAttributes=len(self.dataIn[0])
        n=len(self.dataOut)
        for i in range(noAttributes):
            sum=0
            for el in self.dataIn:
                sum=sum+el[i]
            list_means.append(sum/n)
        list_deviation = []
        for i in range(noAttributes):
            sum=0
            for el in self.dataIn:
                sum=sum+(el[i]-list_means[i])**2
            list_deviation.append(math.sqrt(sum/(n-1)))
        normalized_dataIn=[]
        for i in range(n):
            li=[]
            for j in range(len(self.dataIn[i])):
                li.append((self.dataIn[i][j]-list_means[j])/list_deviation[j])
            normalized_dataIn.append(li)
        #denormalization
        #a=normalized_dataIn[0][0]
        #print(a*list_deviation[0]+list_means[0])
        #print("normaized data",normalized_dataIn)
        return [normalized_dataIn,list_means,list_deviation]
    def normalize_oneData(self,data):
        l=self.normalizeDataIn()
        for i in range(len(data[0])):
            data[0][i]=(data[0][i]-l[1][i])/l[2][i]
        return data
    def constructInputMatrix(self): #data initialy given as bidimensional list
        input=self.normalized_dataIn
        Matrix=np.array(input)
        #print("matrix",Matrix)
        return Matrix
    def constructOutMatrix(self):
        l=[]
        for el in self.dataOut:
            l.append([el])
        l=np.array(l)
        return l
    def constructCoeffMatrix(self): #we have 7 features for each data row so we have to initialize 7+1(free) coefficient which will later be learnt
        m=len(self.dataIn[0])
        l=np.random.rand(m,1)
        l=np.array(l)
        coef=[[0.5] for i in range(m)]
        coef=np.array(coef)
        return coef

    def sigmoidFunction(self,z):
        return 1.0 / (1.0 + math.exp(0.0 - z))
    def gradientDescent(self,x, y, theta, alpha, num_iters): #BGD
        """
           Performs gradient descent to learn theta
        """
        myfunc_vec = np.vectorize(self.sigmoidFunction)
        m = y.size  # number of training examples
        for i in range(num_iters):
            y_hat = myfunc_vec(np.dot(x, theta))
            theta = theta - alpha * np.dot(x.T, y_hat - y)
        return theta
    def prediction(self,example, coef):
        s = 0.0
        for i in range(0, len(example)):
            s += coef[i] * example[i]
        return s
    def train(self):
        maxiter=100
        input=self.constructInputMatrix()
        out=self.constructOutMatrix()
        coef=self.constructCoeffMatrix()
        learning_rate=0.001
        return self.gradientDescent(input,out,coef,learning_rate,maxiter)
    def model(self,input):
        a=self.train()
        input=self.normalize_oneData(input)
        res = self.sigmoidFunction(self.prediction(input[0], a))
        cut_point=0.5
        print("The probability is:",res)
        if(res>=cut_point):
            return 1
        return 0
