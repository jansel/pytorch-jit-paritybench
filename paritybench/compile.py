import torch
try:
    # you will need to install torchdynamo, functorch
    # see: https://github.com/pytorch/torchdynamo
    import torchdynamo
    from functorch.compile import draw_graph
    from torchdynamo.optimizations import BACKENDS
    torchdynamo_en = True
except:
    torchdynamo_en = False

def my_compiler(gm: torch.fx.GraphModule, example_inputs):
    '''
    add your own compiler for torch.fx.GraphModule
    '''
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward  # return a python callable

def graph_drawer(name='graph'):
    '''
    draws a graph for torch.fx.GraphModule in .svg
    '''
    def f(gm: torch.fx.GraphModule, inps):
        draw_graph(gm, name)
        return gm
    return f

compile_functions = {
    'torchscript': torch.jit.script,
    'fxgraph_draw': graph_drawer,
    'my_compiler': my_compiler,
}

compile_functions.update(BACKENDS)