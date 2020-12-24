import torch

class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

    def forward(self, a, b):
        return a, torch.tensor([])


my_module = MyModule()

dummy = (torch.rand(2), torch.rand(2))
torch.onnx.export(my_module, dummy, "batchdim_mismatch.onnx")

my_module_traced = torch.jit.trace(my_module, dummy)
torch.jit.save(my_module_traced, "batchdim_mismatch.pt")
