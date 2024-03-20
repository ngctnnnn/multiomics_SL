
import pandas as pd 

from pathlib import Path

import torch 
from hyena import (
    HyenaDNAPreTrainedModel, 
    CharacterTokenizer
)

class GenomicsEncoder():
    def __init__(self, 
                 genomics_data_path: str = None) -> None:
        """
        Initialize
        """
        # Read genomics data
        # self.genomics_data_path = genomics_data_path

        # if self.genomics_data_path is None:
        #     raise FileNotFoundError(f'{self.genomics_data_path} not found or not existed!')
            
        # self.genomics_data = pd.read_csv(self.genomics_data_path)
        
        # Initialize protein sequence encoder & protein alphabet
        self._initialize_protein_sequence_encoder()
        
        # Initialize gene sequence encoder & tokenizer
        self._initialize_gene_sequence_encoder()
    
    def _initialize_protein_sequence_encoder(self):
        self.protein_sequence_encoder, self.protein_alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")

    def encode_protein_sequence(self, protein_sequence):
        self.protein_sequence_encoder.eval()
        
        batch_converter = self.protein_alphabet.get_batch_converter()
        
        with torch.no_grad():
            batch_labels, batch_strs, batch_tokens = batch_converter(protein_sequence)
            batch_lens = (batch_tokens != self.protein_alphabet.padding_idx).sum(1)   
            results = self.protein_sequence_encoder(batch_tokens, repr_layers=[33], return_contacts=True)
            
        token_representations = results["representations"][33]

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        protein_sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            protein_sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
        return protein_sequence_representations

    def _initialize_gene_sequence_encoder(self,
                                          pretrained_model_name: str = 'hyenadna-small-160k-seqlen',
                                          use_padding: bool = True,
                                          rc_aug: bool = False,
                                          add_eos: bool = False,
                                          use_head: bool = False, # decoder head
                                          n_classes: int = 1,
                                          backbone_cfg = None):        
        '''
        this selects which backbone to use, and grabs weights/ config from HF
        4 options:
            'hyenadna-tiny-1k-seqlen'   # fine-tune on colab ok
            'hyenadna-small-32k-seqlen'
            'hyenadna-medium-160k-seqlen'  # inference only on colab
            'hyenadna-medium-450k-seqlen'  # inference only on colab
            'hyenadna-large-1m-seqlen'  # inference only on colab
        '''

        max_lengths = {
            'hyenadna-tiny-1k-seqlen': 1024,
            'hyenadna-small-32k-seqlen': 32768,
            'hyenadna-medium-160k-seqlen': 160000,
            'hyenadna-medium-450k-seqlen': 450000,  # T4 up to here
            'hyenadna-large-1m-seqlen': 1_000_000,  # only A100 (paid tier)
        }
        
        max_length = max_lengths[pretrained_model_name]
        if pretrained_model_name in ['hyenadna-tiny-1k-seqlen',
                                    'hyenadna-small-32k-seqlen',
                                    'hyenadna-medium-160k-seqlen',
                                    'hyenadna-medium-450k-seqlen',
                                    'hyenadna-large-1m-seqlen']:
            # use the pretrained Huggingface wrapper instead
            self.gene_sequence_encoder = HyenaDNAPreTrainedModel.from_pretrained(
                './model_weights',
                pretrained_model_name,
                download=True,
                config=backbone_cfg,
                device=self.device,
                use_head=use_head,
                n_classes=n_classes,
            )
        else:
            self.gene_sequence_encoder = None
            assert self.gene_sequence_encoder is None, "[Error] Model name is not found!"
        
        self.gene_sequence_tokenizer = CharacterTokenizer(
            characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
            model_max_length=max_length + 2,  # to account for special tokens, like EOS
            add_special_tokens=False,  # we handle special tokens elsewhere
            padding_side='left', # since HyenaDNA is causal, we pad on the left
        )

    def encode_gene_sequence(self, 
                             gene_sequence):
        tokenized_sequence = self.gene_sequence_tokenizer(gene_sequence)["input_ids"]

        tokenized_sequence = torch.LongTensor(tokenized_sequence).unsqueeze(0)
        tokenized_sequence = tokenized_sequence.to(self.device)

        self.gene_sequence_encoder = self.gene_sequence_encoder.to(self.device)
        self.gene_sequence_encoder.eval()
        with torch.inference_mode():
            gene_sequence_embedding = self.gene_sequence_encoder(tokenized_sequence)
        return gene_sequence_embedding
        
if __name__=="__main__":
    genomic_encoder = GenomicsEncoder()
    
    """
    Test for encode protein sequence
    """
    example_protein_data = [
        ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
        ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ("protein3",  "K A <mask> I S Q"),
    ]
    protein_sequence_representations = genomic_encoder.encode_protein_sequence(example_protein_data)
    print("Embedding for protein sequence:")
    for seq in protein_sequence_representations:
        print(seq.shape)
        
    print("===="*10)
    
    """
    Test for encode gene sequence
    """
    example_gene_sequene = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
    gene_sequence_representations = genomic_encoder.encode_gene_sequence(example_gene_sequene)
    print("Embedding for gene sequence:", gene_sequence_representations)