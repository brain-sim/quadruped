from .cnn_agent import CNNPPOAgent, CNNTD3Actor, CNNTD3Critic, CNNFastTD3Actor, CNNFastTD3Critic
from .mlp_agent import MLPPPOAgent, MLPTD3Actor, MLPTD3Critic, MLPFastTD3Actor, MLPFastTD3Critic
from .utils import layer_init

__all__ = ["MLPPPOAgent", "CNNPPOAgent", "layer_init"]
AGENT_LOOKUP_BY_INPUT_TYPE = {
    "image": {
        "ppo": CNNPPOAgent,
        "td3": [CNNTD3Actor, CNNTD3Critic],
        "fast_td3": [CNNFastTD3Actor, CNNFastTD3Critic],
    },
    "state": {
        "ppo": MLPPPOAgent,
        "td3": [MLPTD3Actor, MLPTD3Critic],
        "fast_td3": [MLPFastTD3Actor, MLPFastTD3Critic],
    },
}
AGENT_LOOKUP_BY_ALGORITHM = {
    "ppo": {
        "image": CNNPPOAgent,
        "state": MLPPPOAgent,
    },
    "td3": {
        "image": [CNNTD3Actor, CNNTD3Critic],
        "state": [MLPTD3Actor, MLPTD3Critic],
    },
    "fast_td3": {
        "image": [CNNFastTD3Actor, CNNFastTD3Critic],
        "state": [MLPFastTD3Actor, MLPFastTD3Critic],
    },
}
