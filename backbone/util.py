from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152

def getNetwork(cfg_network, n_class):
    name = cfg_network["backbone"]
    n_cv = cfg_network["n_cv"]
    print("[ DATADIR ] ",)
    print("[ NETWORK ] ",name,"[N_CV]",n_cv)
    
    try:
        return {
            "resnet18": resnet18(n_class, n_cv),
            "resnet34": resnet34(n_class, n_cv),
            "resnet50": resnet50(n_class, n_cv),
            "resnet101": resnet101(n_class, n_cv),
            "resnet152": resnet152(n_class, n_cv),
        }[name]
    except:
        raise (f"Model {name} not available")