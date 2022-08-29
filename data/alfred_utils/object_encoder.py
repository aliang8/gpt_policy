import numpy as np

import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as geom_nn

gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv,
}


class GNNModel(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        c_out,
        num_layers=2,
        layer_name="GCN",
        dp_rate=0.1,
        **kwargs,
    ):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels=in_channels, out_channels=out_channels, **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate),
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels, out_channels=c_out, **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_atr):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index, edge_attr=edge_attr)
            else:
                x = l(x)
        return x


def encode_list_objects(
    list_objs,
    tokenizer,
    boolean_properties,
    max_obj_tokens,
    obj2tok,
    max_visible_objects,
):
    obj_encodings = []
    for obj in list_objs:
        obj_encoding = encode_object(
            obj, tokenizer, boolean_properties, max_obj_tokens, obj2tok
        )
        obj_encodings.append(obj_encoding)

    obj_encodings = np.stack(obj_encodings)
    num_objs, emb_dim = obj_encodings.shape
    obj_encodings_pad = np.zeros((max_visible_objects, emb_dim))
    obj_encodings_pad[:num_objs] = obj_encodings

    return obj_encodings_pad


def encode_object(obj_metadata, tokenizer, boolean_properties, max_obj_tokens, obj2tok):
    # object encoding: name + position + rotation + one_hot_binary_properties
    # name = obj_metadata["name"]
    name = obj_metadata["objectType"]
    # tokenize object type
    name_tokens = None

    position = obj_metadata["position"]
    position_vec = np.array([position["x"], position["y"], position["z"]])
    rotation = obj_metadata["rotation"]
    rotation_vec = np.array([rotation["x"], rotation["y"], rotation["z"]])
    distance = obj_metadata["distance"]  # some floating point value
    distance = np.array([distance])

    # 23 dimensions
    state_vec = np.array([obj_metadata[k] for k in boolean_properties], dtype=np.int)

    # handle parentReceptacle, receptacleObjectIds, and ObjectTemperature
    if obj_metadata["parentReceptacle"]:
        pass

    if obj_metadata[
        "receptacleObjectIds"
    ]:  # this should go into the relationship graph
        pass
        # print(f"{obj_metadata['objectId']} contains: ")

    # 34 dimensions
    obj_encoding = np.concatenate(
        [name_tokens, position_vec, rotation_vec, distance, state_vec]
    )
    return obj_encoding


if __name__ == "__main__":
    import pickle
    from utils.lang_utils import get_tokenizer

    pkl_f = "/data/anthony/alfred/data/json_2.1.0/train/pick_and_place_with_movable_recep-KeyChain-Plate-Shelf-214/trial_T20190908_221228_624762/traj_metadata.pkl"
    traj_metadata = pickle.load(open(pkl_f, "rb"))

    num_steps = len(traj_metadata)
    init_event = traj_metadata[0].metadata

    objects = init_event["objects"]
    print(f"{len(objects)} objects ")

    # ignore these because they're not boolean
    boolean_properties = [k for k, v in objects[0].items() if type(v) == bool]

    tokenizer = get_tokenizer("gpt2")

    for obj in objects:
        encode_object(obj, tokenizer, boolean_properties)

    # from torch_geometric.data import Data

    # model = GNNModel(c_in=10, c_hidden=64, c_out=64, layer_name="GAT", edge_dim=10)
    # print(model)

    # input_feat = torch.randn((6, 10))
    # edge_index = torch.Tensor([[0, 1], [3, 5]]).long()
    # edge_attr = torch.randn((2, 10))

    # print(edge_index.shape, input_feat.shape)
    # out = model(input_feat, edge_index, edge_attr)
    # print(out.shape)
    # out = geom_nn.global_mean_pool(out, torch.zeros((6,)).long())
    # print(out.shape)
    # print(out)
