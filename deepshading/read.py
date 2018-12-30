import OpenEXR
import Imath
import os
import numpy as np

class Input:

    def __init__(self):
        print('Reading data'.center(50, '-'))
        f = open('command.txt', 'r')
        self.command = eval(f.read())
        f.close()
        foldername = self.command['foldername']
        attributes = os.listdir(foldername)
        tg = False
        ct = 0
        cnt = 0
        for attribute in attributes:
            if attribute[0] != '.':
                cnt = cnt + 1
        for attribute in attributes:
            if attribute[0] == '.':
                continue
            path = foldername + '/' + attribute
            inputs = os.listdir(path)
            if not tg:
                tg = True
                data = [[] for i in range(len(inputs))]
                output = [[] for i in range(len(inputs))]
            counter = 0
            sum = len(inputs) * cnt
            for i in inputs:
                f = OpenEXR.InputFile(path + '/' + i)
                header = f.header()
                dw = header['dataWindow']
                pt = Imath.PixelType(Imath.PixelType.FLOAT)
                size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
                r = np.fromstring(f.channel('R', pt), dtype=np.float32)
                g = np.fromstring(f.channel('G', pt), dtype=np.float32)
                b = np.fromstring(f.channel('B', pt), dtype=np.float32)
                if attribute != 'GT':
                    data[counter].append(r.reshape(size[0], size[1]))
                    data[counter].append(g.reshape(size[0], size[1]))
                    data[counter].append(b.reshape(size[0], size[1]))
                else:
                    output[counter].append(r.reshape(size[0], size[1]))
                    output[counter].append(g.reshape(size[0], size[1]))
                    output[counter].append(b.reshape(size[0], size[1]))
                counter = counter + 1
                ct = ct + 1
                print("\r{:^4.0f}% have been read.".format(ct/sum*100), end='')

        print('')
        data = np.array(data)
        self.data = np.transpose(data, [0, 2, 3, 1])
        output = np.array(output)
        self.output = np.transpose(output, [0, 2, 3, 1])
        print("read -> Input Vector Shape: " + str(self.data.shape))
        print("read -> Output Vector Shape: " + str(self.output.shape))
        print('Reading finished'.center(50, '-'))

    def sample(self, batch_size):
        data = np.zeros([batch_size, self.data.shape[1], self.data.shape[2], self.data.shape[3]])
        output = np.zeros([batch_size, self.output.shape[1], self.output.shape[2], self.output.shape[3]])
        ar = np.random.randint(self.data.shape[0], size=batch_size)
        for i in range(batch_size):
            data[i] = self.data[ar[i]]
            output[i] = self.output[ar[i]]
        data = np.array(data)
        output = np.array(output)
        return data, output


class OtherInput(Input):

    def __init__(self):
        print('Reading data'.center(50, '-'))
        f = open('command.txt', 'r')
        self.command = eval(f.read())
        f.close()
        if self.command['is_training'] == 1:
            foldername = self.command['foldername']
            attributes = os.listdir(foldername)
            attributes.sort()
            data = []
            output = []
            for i in attributes:
                f = open(foldername + '/' + i)
                if i[0] == '.':
                    continue
                j = i.split('_')
                print(i, j[1])
                if j[1] == 'in.txt':
                    data.append(eval(f.read()))
                else:
                    output.append(eval(f.read()))
                f.close()

            self.data = np.array(data)
            self.output = np.array(output)
        else:
            self.data = np.zeros([1, 256, 256, 8])
            f = open(self.command['testoutname'], 'r')
            self.output = np.array([eval(f.read())])
            f.close()
        f = open(self.command['testname'], 'r')
        self.test = np.array([eval(f.read())])
        f.close()
        self.outref = self.output
        for k in range(0, self.output.shape[0]):
            for i in range(0, self.output.shape[1]):
                for j in range(0, self.output.shape[2]):
                    self.outref[k][i][j] = np.array(list(map(float, list(map(bool, list(self.output[k][i][j]))))))
        self.output *= 100
        print("read -> Input Vector Shape: " + str(self.data.shape))
        print("read -> Output Vector Shape: " + str(self.output.shape))
        print("read -> Test Vector Shape: " + str(self.test.shape))
        print("read -> Outref Vector Shape: " + str(self.outref.shape))
        print('Reading finished'.center(50, '-'))

    def sample(self, batch_size):
        data = np.zeros([batch_size, self.data.shape[1], self.data.shape[2], self.data.shape[3]])
        output = np.zeros([batch_size, self.output.shape[1], self.output.shape[2], self.output.shape[3]])
        z = np.zeros([batch_size, self.output.shape[1], self.output.shape[2], self.output.shape[3]])
        ar = np.random.randint(self.data.shape[0], size=batch_size)
        for i in range(batch_size):
            data[i] = self.data[ar[i]]
            output[i] = self.output[ar[i]]
            z[i] = self.outref[ar[i]]
        data = np.array(data)
        output = np.array(output)
        z = np.array(z)
        return data, output, z