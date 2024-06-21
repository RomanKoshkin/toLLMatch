from examples.speech_to_text.data_utils import load_df_from_tsv
from torch.utils.data import Dataset
from transformers import MBart50TokenizerFast
from typing import Tuple


class SourceDataset(Dataset):
    def __init__(
        self,
        path_to_asr_tsv: str,
        path_to_st_tsv: str,
        tokenizer: MBart50TokenizerFast = None,
        completed_ids: list = [],
        seed: int = 42,
        max_frames: int = None,
    ):
        super().__init__()

        self.seed = seed

        asr_df = load_df_from_tsv(path_to_asr_tsv)
        st_df = load_df_from_tsv(path_to_st_tsv)
        assert (
            asr_df.id.tolist() == st_df.id.tolist()
        ), "Inconsistent datasets for ASR and ST"
        st_df["src_text"] = asr_df["tgt_text"].tolist()

        if max_frames is not None:
            asr_df = asr_df.loc[asr_df.n_frames < max_frames]
            st_df = st_df.loc[st_df.n_frames < max_frames]

        if tokenizer is not None:
            print("Tokenizing and sorting corpus ...")
            st_df["src_num_tokens"] = st_df.progress_apply(
                lambda x: len(tokenizer.tokenize(x["src_text"])), axis=1
            )
            st_df.sort_values(
                "src_num_tokens", inplace=True, ignore_index=True, ascending=False
            )

        self.data = [
            (id_, src_txt, tgt_txt)
            for id_, src_txt, tgt_txt in zip(
                st_df["id"].tolist(),
                st_df["src_text"].tolist(),
                st_df["tgt_text"].tolist(),
            )
            if id_ not in completed_ids
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Tuple[int, str, str]:
        return self.data[i]
    
    def collater(self, batch: list) -> Tuple[list[int], list[str], list[str]]:
        ids = [example[0] for example in batch]
        src_txt = [example[1] for example in batch]
        tgt_txt = [example[2] for example in batch]
        return (ids, src_txt, tgt_txt)