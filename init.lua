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
local function nilling(module)
   module.gradBias          = nil
   if module.finput then module.finput = torch.Tensor():typeAs(module.finput) end
   module.gradWeight        = nil
   module.output            = torch.Tensor():typeAs(module.output)
   module.fgradInput        = nil
   module.gradInput         = nil
   module.gradWeightPartial = nil
end

local function netLighter(network)
   nilling(network)
   if network.modules then
      for _,a in ipairs(network.modules) do
         netLighter(a)
      end
   end
end

local function nillingFields(module, fields)
   for key, val in pairs(module) do
      if (string.match(torch.typename(val) or '', 'Tensor') and not fields[key]) then
         module[key] = torch.Tensor():typeAs(val)
      end
   end
end

local function netFieldsLighter(network, fields)
   nillingFields(network, fields)
   if network.modules then
      for _,a in ipairs(network.modules) do
         netFieldsLighter(a, fields)
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
local function saveNet(model, fileName)
   -- Getting rid of unnecessary things and freeing the memory
   netLighter(model)
   collectgarbage()
   torch.save(fileName, model)
   -- Repopulate the gradWeight through the whole net
   repopulateGradNBias(model)
   -- Return NEW storage for <weight> and <grad>
   return model:getParameters()
end

local function saveNetFields(fileName, model, fields)
   local my_fields = {['modules'] = true}
   for _, val in pairs(fields) do
      my_fields[val] = true
   end
   netFieldsLighter(model, my_fields)
   collectgarbage()
   torch.save(fileName, model)
   repopulateGradNBias(model)

   return model:getParameters()
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
