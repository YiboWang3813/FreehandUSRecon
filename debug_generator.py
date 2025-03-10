import torch 
import torch.nn as nn 

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(128)
        self.relu = nn.ReLU(inplace=True)

        self.conv3d_1 = nn.Conv3d(in_channels=2, out_channels=32, kernel_size=3, stride=(1, 1, 1), padding=1)
        self.conv3d_2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=(1, 2, 2), padding=1)

        self.conv3d_3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=(1, 1, 1), padding=1)

        self.conv3d_4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=(2, 2, 2), padding=1)

        self.conv3d_5 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=(1, 1, 1), padding=1)
        self.conv3d_6 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=(1, 1, 1), padding=1)

        self.conv3d_7 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, stride=(1, 1, 1), padding=1)

        self.conv3d_8 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=(1, 1, 1), padding=1)
        self.conv3d_9 = nn.Conv3d(in_channels=128, out_channels=16, kernel_size=5, stride=(1, 1, 1), padding=0)
        # self.conv3d_10 = nn.Conv3d(in_channels=16, out_channels=8, kernel_size=3, stride=(1, 1, 1), padding=1)

        self.drop3d_1 = nn.Dropout3d(0.25)

        self.drop2d_1 = nn.Dropout2d(0.25)
        self.drop2d_2 = nn.Dropout2d(0.1) # 输入dropout时已经是[B, C]的格式了

        flatten_size = 27648 
        # self.fc1 = nn.Linear(38400, 128)
        self.fc1 = nn.Linear(flatten_size, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 6)


    def forward(self, x):

        # diary = True
        diary = False 

        if diary == True:
            print('give {}'.format(x.shape))

            x = self.conv3d_1(x)
            x = self.relu(x)
            print('1_3d {}'.format(x.shape))

            x = self.conv3d_2(x)
            x = self.relu(x)
            print('2_3d {}'.format(x.shape))

            x = self.bn1(x)
            print('3_bn {}'.format(x.shape))

            x = self.conv3d_3(x)
            x = self.relu(x)
            print('4_3d {}'.format(x.shape))

            x = self.conv3d_4(x)
            x_0 = self.relu(x)
            print('5_3d {}'.format(x_0.shape))

            x = self.conv3d_5(x_0)
            x = self.relu(x)
            print('6_3d {}'.format(x.shape))

            x = self.conv3d_6(x)
            x = self.relu(x)
            print('7_3d {}'.format(x.shape))

            x = self.conv3d_7(x)
            x_1 = self.relu(x)
            print('8_3d {}'.format(x_1.shape))

            x_01 = torch.cat((x_0, x_1), 1)
            print('9_cn {}'.format(x_01.shape))

            x = self.bn2(x_01)
            print('10_bn {}'.format(x.shape))

            x = self.conv3d_8(x)
            x = self.relu(x)
            print('11_3d {}'.format(x.shape))

            x = self.conv3d_9(x)
            # x = self.relu(x)
            print('12_3d {}'.format(x.shape))

            x = x.view(x.size()[0], -1)
            x = self.relu(x)
            x = self.drop3d_1(x)
            print('13_fl {}'.format(x.shape))

            x = self.fc1(x)
            x = self.relu(x)
            # x = self.drop2d_1(x)
            print('15_fc {}'.format(x.shape))

            x = self.fc2(x)
            x = self.relu(x)
            # x = self.drop2d_2(x)
            print('17_fc {}'.format(x.shape))

            x = self.fc3(x)
            print('19_fc {}'.format(x.shape))
            # time.sleep(30)
        else:
            x = self.conv3d_1(x)
            x = self.relu(x)

            x = self.conv3d_2(x)
            x = self.relu(x)

            x = self.bn1(x)

            x = self.conv3d_3(x)
            x = self.relu(x)

            x = self.conv3d_4(x)
            x_0 = self.relu(x)

            x = self.conv3d_5(x_0)
            x = self.relu(x)

            x = self.conv3d_6(x)
            x = self.relu(x)

            x = self.conv3d_7(x)
            x_1 = self.relu(x)

            x_01 = torch.cat((x_0, x_1), 1)

            x = self.bn2(x_01)

            x = self.conv3d_8(x)
            x = self.relu(x)

            x = self.conv3d_9(x)

            # print(x.shape) 

            x = x.view(x.size()[0], -1)
            x = self.relu(x)
            x = self.drop3d_1(x)

            # print(x.shape) 

            x = self.fc1(x)
            x = self.relu(x)
            x = self.drop2d_1(x)

            x = self.fc2(x)
            x = self.relu(x)
            x = self.drop2d_2(x)

            x = self.fc3(x)
            # time.sleep(30)

        return x
    
network = Generator() 
x = torch.randn(8, 2, 32, 64, 64) 
y = network(x) 