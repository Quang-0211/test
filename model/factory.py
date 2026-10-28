from .GCN2 import GCN2
from .RGCN2 import RGCN2


def create_model(model_name, *, num_bases=8, num_blocks=None, num_relations=128):
    if model_name == "gcn":
        return GCN2()
    if model_name == "rgcn":
        return RGCN2(
            num_relations=num_relations,
            num_bases=num_bases,
            num_blocks=num_blocks,
        )
    raise ValueError(f"Unknown model: {model_name!r}; expected 'gcn' or 'rgcn'")
