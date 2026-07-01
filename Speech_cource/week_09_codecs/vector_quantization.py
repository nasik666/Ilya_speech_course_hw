import torch
import torch.nn as nn
import torch.nn.functional as F


class Perplexity(nn.Module):
    EPS = 1e-8
    def __init__(self, n_codecs):
        super().__init__()
        self.n_codecs = n_codecs
    
    def forward(self, indices):
        device = indices.device

        arange = torch.arange(self.n_codecs, device=device)
        indices = indices.flatten()
        encodings = torch.eq(arange.unsqueeze(dim=1), indices.unsqueeze(dim=0))

        probs = torch.mean(encodings.float(), dim=1)
        perplexity = torch.exp(- torch.sum(probs * torch.log(probs + self.EPS)))
        return perplexity


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size

        self.codebook = nn.Embedding(
            num_embeddings=codebook_size,
            embedding_dim=embedding_dim)

        self._init_weight()

    def _init_weight(self):
        init_size = 1 / self.codebook_size
        torch.nn.init.uniform_(self.codebook.weight, a=-init_size, b=init_size)

    def calculate_squared_distances(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        """
        tensor_1: float tensor with shape [sequence_1, embedding] 
        tensor_2: float tensor with shape [sequence_2, embedding]
        output: float tensor with shape [sequence_1, sequence_2]
        """
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a . b
        sq_1 = (tensor_1 ** 2).sum(dim=1, keepdim=True)  # [sequence_1, 1]
        sq_2 = (tensor_2 ** 2).sum(dim=1, keepdim=True).t()  # [1, sequence_2]
        cross_term = tensor_1 @ tensor_2.t()  # [sequence_1, sequence_2]

        distances = sq_1 + sq_2 - 2 * cross_term

        return distances

    def encode(self, embeddings: torch.Tensor):
        """
        Encodes the input embeddings, by the indices of closest embeddings from the codebook
        embeddings: Embedded image of size [batch, embedding, height, width]
        output: LongTensor of indices of size [batch, height, width]
        """
        assert embeddings.dim() == 4
        B, E, H, W = embeddings.shape

        flat_embeddings = embeddings.permute(0, 2, 3, 1).reshape(-1, E)  # [B*H*W, E]

        distances = self.calculate_squared_distances(flat_embeddings, self.codebook.weight)  # [B*H*W, codebook_size]
        indices = distances.argmin(dim=1)  # [B*H*W]

        indices = indices.view(B, H, W)

        return indices

    def decode(self, indices: torch.Tensor):
        """
        Inserts embeddings from the codebook instead of indices
        Indices: Longtensor of indices from the codebook of size [batch, height, width]
        For each index: 0 <= index < codebook_size
        output: FloatTensor of codec vectors from codebook of size [batch, embedding, height, width]
        """
        B, H, W = indices.shape

        flat_indices = indices.reshape(-1)  # [B*H*W]
        flat_decoded = self.codebook(flat_indices)  # [B*H*W, E]

        decoded = flat_decoded.view(B, H, W, self.embedding_dim).permute(0, 3, 1, 2).contiguous()

        return decoded

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Quantizes embeddings
        """
        indices = self.encode(embeddings)
        quantized = self.decode(indices)

        return quantized


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, codebook_size, embedding_dim, n_codebooks):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.n_codebooks = n_codebooks
        
        self.codebooks = [VectorQuantizer(codebook_size, embedding_dim) for _ in range(n_codebooks)]
        self.codebooks = nn.ModuleList(self.codebooks)

    def encode(self, embeddings: torch.Tensor):
        """
        Encodes the input embeddings, by the indices of closest embeddings from the codebook.
        Then iteratively encodes the residuals between the embedding and vectors from the codebook the same way.
        embeddings: Embedded image of size [batch, embedding, height, width]
        output: LongTensor of indices of size [batch, n_codebooks, height, width]
        """
        residual = embeddings
        codecs = []
        for quantizer in self.codebooks:
            indices = quantizer.encode(residual)
            quantized = quantizer.decode(indices)
            residual = residual - quantized
            codecs.append(indices)

        codecs = torch.stack(codecs, dim=1)  # [batch, n_codebooks, height, width]

        return codecs

    def decode(self, codecs: torch.Tensor):
        """
        Sums the embeddings from the codebooks with dedicated indices
        Indices: Longtensor of indices from the codebook of size [batch, n_codebooks, height, width]
        For each index: 0 <= index < codebook_size
        output: FloatTensor of codec vectors from codebook of size [batch, embedding, height, width]
        """
        quantized = []
        for i, quantizer in enumerate(self.codebooks):
            indices = codecs[:, i]
            quantized.append(quantizer.decode(indices))

        return sum(quantized)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Quantizes embeddings
        """
        indices = self.encode(embeddings)
        quantized = self.decode(indices)
        return quantized


class VectorQuantizationLoss(nn.Module):
    def __init__(self, commitment_cost=1.):
        super().__init__()
        self.commitment_cost = commitment_cost

    def forward(self, inputs, quantized):
        """
        Calculates the vector quantisation loss
        inputs: vector of embeddings of size [batch, embedding, height, width]
        quantized: the vector of embeddings, processed by VectorQuantisation ot ResidualVectorQuantization
        output: differentiable loss of size [1]
        """
        # `inputs` is detached, quantization_loss: векторы codebook направляются на выход encoder и градиенты должны поступать только в quantizer(не в encoder)
        quantization_loss = F.mse_loss(quantized, inputs.detach())

        # `quantized` is detached, commitment_loss: смещает выходные данные encoder в сторону векторов codebook.Градиенты должны поступать только в encoder
        commitment_loss = F.mse_loss(inputs, quantized.detach())

        loss = quantization_loss + self.commitment_cost * commitment_loss

        return loss