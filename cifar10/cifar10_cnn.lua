require 'paths'
require 'nn'

trainset = torch.load('../data/cifar10/cifar10-train.t7')
testset = torch.load('../data/cifar10/cifar10-test.t7')
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}


setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);
trainset.data = trainset.data:double()

function trainset:size() 
    return self.data:size(1) 
end

mean = {} 
stdv  = {} 
for i=1,3 do 
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() 
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i])
    
    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std()
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) 
end

net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 6, 5, 5)) 
net:add(nn.ReLU())                       
net:add(nn.SpatialMaxPooling(2,2,2,2))     
net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.ReLU())                       
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16*5*5))                 
net:add(nn.Linear(16*5*5, 120))          
net:add(nn.ReLU())                       
net:add(nn.Linear(120, 84))
net:add(nn.ReLU())                       
net:add(nn.Linear(84, 10))               
net:add(nn.LogSoftMax())    

criterion = nn.ClassNLLCriterion()

trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 10 

trainer:train(trainset)

testset.data = testset.data:double()   
for i=1,3 do 
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) 
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) 
end    

correct = 0
for i=1,10000 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

print(correct, 100*correct/10000 .. ' % ')

class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
for i=1,10000 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
end

for i=1,#classes do
    print(classes[i], 100*class_performance[i]/1000 .. ' %')
end
