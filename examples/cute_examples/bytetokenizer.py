# it is from https://github.com/goombalab/hnet/blob/main/generate.py
import numpy as np
import torch
class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256
        self.bos_idx = 254
        self.eos_idx = 255
        self.dtype = np.uint8

    def encode(self, seqs, add_bos=False, add_eos=False, **kwargs):
        total_outputs = []
        for text in seqs:
            text_byte = text.encode("utf-8")

            if add_bos:
                text_byte = bytes([self.bos_idx]) + text_byte
                print(f"text_byte: {text_byte}")
            if add_eos:
                text_byte = text_byte + bytes([self.eos_idx])
            text_byte = bytearray(text_byte)
            print(f"text_byte: {text_byte}")
            text_byte_ids = np.array(text_byte, dtype=self.dtype)

            total_outputs.append({"input_ids": text_byte_ids})

        return total_outputs

    def decode(self, tokens, **kwargs):
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        return bytearray(tokens).decode("utf-8", **kwargs)


if __name__=="__main__":
    seqs = "今天天气很好"
    tokenizer = ByteTokenizer()
    encoded = tokenizer.encode(seqs, add_bos=True)[0]
    print(f"encoded: {encoded}")
    input_idss = torch.tensor(
        encoded["input_ids"], dtype=torch.long, device=torch.device("cpu")
    ).unsqueeze(0)
    print(f"input_idss: {input_idss}, its shape: {input_idss.shape}")
