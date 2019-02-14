import torch

#def bar(a, b):
#    return a + b
#
#m = torch.jit.trace(bar, (torch.rand(2), torch.rand(2)))
#
#torch.jit.save(m, 'pt-minimal.pt')
#
#m2 = torch.jit.load('pt-minimal.pt')
#
#out = m2.forward(torch.rand(2), torch.rand(2))
#print(out)


class MyModule(torch.jit.ScriptModule):
    def __init__(self):
        super(MyModule, self).__init__()

    @torch.jit.script_method
    def forward(self, a, b):
        return a + b

my_script_module = MyModule()

print(my_script_module(torch.rand(2), torch.rand(2)))

my_script_module.save("foo.pt")
