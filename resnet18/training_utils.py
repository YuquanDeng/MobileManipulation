V = TypeVar("V")

def resnet(
    weights, 
    device, 
    block: Type[BasicBlock],
    num_classes: int,
    img_channels: int = 3, 
    num_layers: int = 18,
    **kwargs: Any) -> ResNet:
    """
    Wrapper method for creating a ResNet with the given weights. Return a ResNet18 model.
    """

    ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(
        img_channels=img_channels,
        num_layers=num_layers,
        block=block,
        **kwargs
        ).to(device)
    
    model.load_state_dict(weights.get_state_dict(progress=True))
    
    return model

def ovewrite_named_param(kwargs: Dict[str, Any], param: str, new_value: V) -> None:
    """
    Change the value to the given new_value of the given key param. 
    """
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value
