require 'paths'
require 'nn'

trainset = torch.load('../data/mnist/train_32x32.t7','ascii')
testset = torch.load('../data/mnist/test_32x32.t7','ascii')

setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.labels[i]} 
                end}
);

trainset.data = trainset.data:double()

function trainset:size() 
    return self.data:size(1) 
end

mean = trainset.data:mean()
print ('Mean = ' .. mean)
trainset.data = trainset.data:add(-mean)
stdv = trainset.data:std()
print('Stdv = ' .. stdv)
trainset.data = trainset.data:div(stdv)

testset.data = testset.data:double()
testset.data = testset.data:add(-mean)
testset.data = testset.data:div(stdv)

net = nn.Sequential()
net:add(nn.SpatialConvolution(1, 6, 5, 5)) 
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
trainer.maxIteration = 5

print('done1')
trainer:train(trainset)
print('done2')
correct = 0
for i=1,10000 do
    local groundtruth = testset.labels[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

print(correct, 100*correct/10000 .. ' % ')

class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
total_classes = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
for i=1,10000 do
    local groundtruth = testset.labels[i]
    total_classes[testset.labels[i]] = total_classes[testset.labels[i]] + 1 
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
end


for i=1,10 do
    print(i,100*class_performance[i]/total_classes[i])
end

torch.save('mnist_cnn_trained',net)             

















