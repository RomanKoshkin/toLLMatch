from fairseq.data import BaseWrapperDataset
from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDataset,
    SpeechToTextDatasetItem,
)
from typing import List, Dict
import torch
from pathlib import Path
import torch.nn.functional as F


class SpeechDistillationDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: SpeechToTextDataset,
        kd_root: str,
        pad_idx: int
    ) -> None:

        super(SpeechDistillationDataset, self).__init__(dataset)
        
        self.kd_root = Path(kd_root)
        self.pad_idx = pad_idx

    def __getitem__(self, index: int) -> SpeechToTextDatasetItem:

        example = self.dataset.__getitem__(index)

        # adds the teacher output to the example
        example.teacher_output = torch.load(
            self.kd_root / example.split / f"{example.id}.pt",
            map_location=torch.device("cpu")
        )

        return example

    def collater(self, samples: List[SpeechToTextDatasetItem]) -> Dict:
        
        if len(samples) == 0:
            return {}
        
        collater_out = self.dataset.collater(samples, return_order=True)

        topk_indices = [example.teacher_output["topk_indices"] for example in samples]
        topk_outputs = [example.teacher_output["topk_outputs"] for example in samples]

        max_len = max([len(indices) for indices in topk_indices])

        topk_indices = torch.stack(
            [
                F.pad(
                    indices,
                    (0, 0, 0, max_len - len(indices)),
                    mode="constant",
                    value=self.pad_idx
                )
                for indices in topk_indices
            ]
        )
        topk_outputs = torch.stack(
            [
                F.pad(
                    outputs,
                    (0, 0, 0, max_len - len(outputs)),
                    mode="constant",
                    value=1
                )
                for outputs in topk_outputs
            ]
        )

        order = collater_out["order"]
        collater_out["teacher_output"] = {
            "topk_indices": topk_indices[order],
            "topk_outputs": topk_outputs[order],
        }

        return collater_out
