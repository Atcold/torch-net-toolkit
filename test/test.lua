require 'cunn'
require 'net-toolkit'

torch.setdefaulttensortype('torch.FloatTensor')

local batchSize = 64

-- test CPU
local model = nn.Sequential()
model:add(nn.SpatialConvolutionMM(3, 10, 9, 9))
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
model:add(nn.Threshold(0,0))
model:add(nn.Dropout(0.5))
model:add(nn.Reshape(90))
model:add(nn.Linear(90,10))
model:add(nn.LogSoftMax())

local batch = torch.randn(batchSize, 3, 14, 14)
local gradient = torch.randn(batchSize)

model:forward(batch)
model:backward(batch, gradient)

netToolkit.saveNetFields('model-test.net', model, {'weight', 'bias'})

local reloadModel = netToolkit.loadNet('model-test.net')

reloadModel:forward(batch)
reloadModel:backward(batch, gradient)

os.execute("rm model-test.net")
print('CPU OK')

-- test GPU
local d_model = nn.Sequential()
d_model:add(nn.Transpose({1,4},{1,3},{1,2}))
d_model:add(nn.SpatialConvolutionCUDA(3, 16, 9, 9))
d_model:add(nn.SpatialMaxPoolingCUDA(2, 2, 2, 2))
d_model:add(nn.Threshold(0,0))
d_model:add(nn.Dropout(0.5))
d_model:add(nn.Reshape(144))
d_model:add(nn.Linear(144,10))
d_model:add(nn.LogSoftMax())
d_model:cuda()

local d_batch = torch.CudaTensor(batchSize, 3, 14, 14):copy(batch)
local d_gradient = torch.CudaTensor(batchSize):copy(gradient)

d_model:forward(d_batch)
d_model:backward(d_batch, d_gradient)

netToolkit.saveNetFields('model-test.net', d_model, {'weight', 'bias'})

local d_reloadModel = netToolkit.loadNet('model-test.net')

d_reloadModel:forward(d_batch)
d_reloadModel:backward(d_batch, d_gradient)

os.execute("rm model-test.net")
print('GPU OK')
