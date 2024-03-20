import torch 
import pandas as pd 

from transformers.models.bert.configuration_bert import BertConfig
from transformers import AutoTokenizer, AutoModel

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

    def _initialize_gene_sequence_encoder(self):
        config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        self.gene_sequence_tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        self.gene_sequence_encoder = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)

    def encode_gene_sequence(self, 
                             gene_sequence):
        inputs = self.gene_sequence_tokenizer(gene_sequence, 
                                              return_tensors='pt')["input_ids"]
        hidden_states = self.gene_sequence_encoder(inputs)[0] # [1, sequence_length, 768]

        # Embedding with max pooling
        # embedding_max = torch.max(hidden_states[0], dim=0)[0]

        # Embedding with mean pooling
        gene_sequence_embedding = torch.mean(hidden_states[0], dim=0)
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
    