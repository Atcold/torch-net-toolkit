package = 'net-toolkit'
version = 'scm-1'

source = {url = 'git://github.com/Atcold/net-toolkit'}

description = {
   summary = 'A simple module for <Torch7> and the <nn> package',
   detailed = [[
   It allows to save and retrive to/from disk a lighter version of the network
   that is being training.
   ]]
}

dependencies = {
   'torch >= 7.0',
   'nn >= 1.0'
}

build = {
   type = 'builtin',
   modules = {
      ['net-toolkit.init'] = 'init.lua'
   }
}
