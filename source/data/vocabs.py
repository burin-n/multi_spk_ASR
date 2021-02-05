# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import string
from typing import List


class Chars:
    """Vocabulary for turning str text to list of int tokens."""

    # fmt: on
    PAD, OOV = '<pad>', '<oov>'
    BOS, EOS = '<BOS>', '<EOS>'
    SEP = ''
    eng_labels= " " + string.ascii_uppercase + "'"

    def __init__(self, labels=eng_labels, *, pad=PAD, oov=OOV, BOS=BOS, EOS=EOS):
        super().__init__()

        labels = list(labels)
        self.pad, labels = len(labels), labels + [pad]  # Padding
        self.oov, labels = len(labels), labels + [oov]  # Out Of Vocabulary
        self.bos, labels = len(labels), labels + [EOS]
        self.eos, labels = len(labels), labels + [BOS]
        self.labels = labels

        self._util_ids = {self.pad, self.oov, self.bos, self.eos}
        self._label2id = {l: i for i, l in enumerate(labels)}
        self._id2label = labels


    def encode(self, text: str) -> List[int]:
        """Turns str text into int tokens."""
        return [self._label2id.get(char, self.oov) for char in text]


    def decode(self, tokens: List[int]) -> str:
        """Turns ints tokens into str text."""
        return self.SEP.join(self._id2label[t] for t in tokens if t not in self._util_ids)


if __name__ == '__main__':
    char = Chars()
    print(char.encode("FAIR'S"))
    print(char.decode([char.bos] + char.encode("FAIR'S") + [char.eos]))
    print(char._label2id)
    
