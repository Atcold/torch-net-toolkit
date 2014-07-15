--------------------------------------------------------------------------------
-- netToolkit
--------------------------------------------------------------------------------
-- A simple module for <Torch7> and the <nn> package.
-- It allows to save and retrive to/from disk a lighter version of the network
-- that is being training.
--------------------------------------------------------------------------------
-- Alfredo Canziani, Mar 2014
--------------------------------------------------------------------------------

-- Private functions definition ------------------------------------------------
local function nilling(module, fields)
   for key, val in pairs(module) do
      if string.match(torch.typename(val) or '', 'Tensor') and not fields[key] then
         module[key] = torch.Tensor():typeAs(val)
      end
   end
end

local function netLighter(network, fields)
   nilling(network, fields)
   if network.modules then
      for _,a in ipairs(network.modules) do
         netLighter(a, fields)
      end
   end
end

local function craftGradNBias(module)
   if module.weight then module.gradWeight = module.weight:clone() end
   if module.bias   then module.gradBias   = module.bias  :clone() end
   if module.__typename == 'nn.SpatialConvolutionCUDA' then
      module.gradWeightPartial = module.weight:clone()
   end
   if module.__typename == 'nn.SpatialConvolutionMM' then
      module.fgradInput = torch.Tensor():typeAs(module.output)
   end
   module.gradInput = torch.Tensor():typeAs(module.output)
end

local function repopulateGradNBias(network)
   craftGradNBias(network)
   if network.modules then
      for _,a in ipairs(network.modules) do
         repopulateGradNBias(a)
      end
   end
end

-- Public functions definition -------------------------------------------------
local function saveNetFields(fileName, model, fields)
   -- Reverse dictionary
   local keepFields = {}
   for _, val in pairs(fields) do
      keepFields[val] = true
   end
   -- Getting rid of unnecessary things and freeing the memory
   netLighter(model, keepFields)
   collectgarbage()
   torch.save(fileName, model)
   -- Repopulate the gradWeight through the whole net
   repopulateGradNBias(model)
   -- Return NEW storage for <weight> and <grad>
   return model:getParameters()
end

local function saveNet(fileName, model)
   return saveNetFields(fileName, model, {'weight', 'bias'})
end

local function loadNet(fileName)
   local model = torch.load(fileName)
   -- Repopulate the gradWeight through the whole net
   repopulateGradNBias(model)
   return model, model:getParameters()
end

-- Selecting public functions --------------------------------------------------
netToolkit = {
   saveNet = saveNet,
   saveNetFields = saveNetFields,
   loadNet = loadNet
}

-- Returning handle
return netToolkit
