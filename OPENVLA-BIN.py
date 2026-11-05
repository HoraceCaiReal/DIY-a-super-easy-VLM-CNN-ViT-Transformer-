"""
action_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
"""                                              # ←01

from typing import List, Union

import numpy as np
from transformers import PreTrainedTokenizerBase


class ActionTokenizer:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, bins: int = 256, min_action: int = -1, max_action: int = 1
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.
        """                                      # ←02
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_action, max_action, self.n_bins)          # ←03
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0             # ←04

        # [Contract] Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))   # ←05

        """
         call在做的事情只有一件事：将action bin划分为256个区间，然后编码送入hugginface里的llama模型（hugginface输入只接受tuple/list）
         目的是为了预训练：
         把一整段连续的物理动作（7维），打包成一个字符串（因为有 7 个特殊 Token 对应的生僻字）
        """

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:   
        """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))    # ←06
        discretized_action = np.digitize(action, self.bins)                    # ←07

        # Handle single element vs. batch
        if len(discretized_action.shape) == 1:
            return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action))  # 1，vocab_size是个常数，这里第一步先广播相减得到各个token id再统一列表化输入hugginface
        else:
            return self.tokenizer.batch_decode((self.tokenizer.vocab_size - discretized_action).tolist())

    """
        这一步才是在做解码，将吐出的string变回bin区间（hugginface此时吐出来的东西是纯数字int，能直接用）
        目的是为了推理：
        把大模型吐出的 7 个特殊 Token（整数 ID），一次性翻译回 7 个连续值。
    """
    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous actions for discrete action token IDs.
        ...（docstring 解释 digitize 返回 [1, n_bins] 而 bin 区间只有 n_bins-1 个的边角问题）...
        """                                      # ←09
        discretized_actions = self.tokenizer.vocab_size - action_token_ids     #反解ID（e.g: 32000-31872 = 128(变成了bin编号)），这里也涉及广播
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)  #clip
        return self.bin_centers[discretized_actions]                           # 查表返回对应bin

    @property
    def vocab_size(self) -> int:
        return self.n_bins                       # ←12
    
        """这就是 Day 9 讲的离散化痛点。
大模型（Transformer）本质是个概率词表分类器。如果让它直接回归连续值（MSE），会遇到多峰陷阱；如果让它输出高精度浮点数，它做不到。

所以，OpenVLA 用了一个巧妙的“偷梁换柱”：

强行把连续空间 [-1, 1] 切成 256 份。

把这 256 份包装成 256 个“单词”。

让 LLM 做它最擅长的“完形填空/单词接龙”。

填出来的单词，我们再逆操作查表，变回物理数字。
        """