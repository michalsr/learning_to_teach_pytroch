from pprint import pprint
from typing import Dict, List
from torchvision import models
import torch


def group_wise_lr(model, group_lr_conf: Dict, path=""):
    """
    Refer https://pytorch.org/docs/master/optim.html#per-parameter-options


    torch.optim.SGD([
        {'params': model.base.parameters()},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], lr=1e-2, momentum=0.9)


    to


    cfg = {"classifier": {"lr": 1e-3},
           "lr":1e-2, "momentum"=0.9}
    confs, names = group_wise_lr(model, cfg)
    torch.optim.SGD([confs], lr=1e-2, momentum=0.9)



    :param model:
    :param group_lr_conf:
    :return:
    """
    assert type(group_lr_conf) == dict
    confs = []
    nms = []
    for kl, vl in group_lr_conf.items():
        assert type(kl) == str
        assert type(vl) == dict or type(vl) == float or type(vl) == int

        if type(vl) == dict:
            assert hasattr(model, kl)
            cfs, names = group_wise_lr(getattr(model, kl), vl, path=path + kl + ".")
            confs.extend(cfs)
            names = list(map(lambda n: kl + "." + n, names))
            nms.extend(names)

    primitives = {kk: vk for kk, vk in group_lr_conf.items() if type(vk) == float or type(vk) == int}
    remaining_params = [(k, p) for k, p in model.named_parameters() if k not in nms]
    if len(remaining_params) > 0:
        names, params = zip(*remaining_params)
        conf = dict(params=params, **primitives)
        confs.append(conf)
        nms.extend(names)

    plen = sum([len(list(c["params"])) for c in confs])
    assert len(list(model.parameters())) == plen
    assert set(list(zip(*model.named_parameters()))[0]) == set(nms)
    assert plen == len(nms)
    if path == "":
        for c in confs:
            c["params"] = (n for n in c["params"])
    return confs, nms


def create_configs_based_on_model(model, lr_list: List):
    all_keys = []
    # we only handle the first level
    for name, param in model.named_parameters():
        submodule_name = name.split('.')[0]
        if submodule_name not in all_keys:
            all_keys.append(submodule_name)

    assert len(lr_list) == len(all_keys)
    configs = {}
    for key, lr in zip(all_keys, lr_list):
        configs[key] = {'lr': lr}

    return configs


def create_configs_predefined(name_list, lr_list):
    r_dict = {}
    for name, lr in zip(name_list, lr_list):
        r_dict[name] = {"lr": lr}
    return r_dict


if __name__ == "__main__":
    model = models.resnet18(pretrained=True)

    # test_configs = [
    #     # Give same Lr to all model params
    #     {"lr": 0.3},
    #
    #     # For the below 3 cases, you will need to pass the optimiser overall optimiser params for remaining model params.
    #     # This is because we did not specify optimiser params for all top-level submodules, so defaults need to be supplied
    #     # Refer https://pytorch.org/docs/master/optim.html#per-parameter-options
    #
    #     # Give same Lr to layer4 only
    #     {"layer4": {"lr": 0.3}},
    #
    #     # Give one LR to layer4 and another to rest of model. We can do this recursively too.
    #     {"layer4": {"lr": 0.3},
    #      "lr": 0.5},
    #
    #     # Give one LR to layer4.0 and another to rest of layer4
    #     {"layer4": {"0": {"lr": 0.001},
    #                 "lr": 0.3}},
    #
    #     # More examples
    #     {"layer4": {"lr": 0.3,
    #                 "0": {"lr": 0.001}}},
    #
    #     {"layer3": {"0": {"conv2": {"lr": 0.001}},
    #                 "1": {"lr": 0.003}}},
    #
    #     {"layer4": {"lr": 0.3},
    #      "layer3": {"0": {"conv2": {"lr": 0.001}},
    #                 "lr": 0.003},
    #      "lr": 0.001}
    # ]
    #
    # for cfg in test_configs:
    #     confs, names = group_wise_lr(model, cfg)
    #     print("#" * 140)
    #     pprint(cfg)
    #     print("-" * 80)
    #     pprint(confs)
    #     print("#" * 140)

    # Example of how to use these functions
    cfgs = create_configs_based_on_model(model, lr_list=[0.001] * 7)
    print(cfgs)
    confs, names = group_wise_lr(model, cfgs)
    print("#" * 140)
    pprint(cfgs)
    print("-" * 80)
    pprint(confs)
    print("#" * 140)

    # Here lr is the default learning rate.
    torch.optim.SGD(confs, lr=1e-2, momentum=0.9)
