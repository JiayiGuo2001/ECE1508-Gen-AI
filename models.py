import torch
from torch import nn
import torchvision


class Encoder(nn.Module):
    """Pretrained CNN encoder that returns a spatial feature grid."""

    BACKBONES = {
        "resnet101": {
            "builder": "resnet101",
            "weights": "ResNet101_Weights",
            "encoder_dim": 2048,
            "feature_container": "children",
        },
        "resnext101_32x8d": {
            "builder": "resnext101_32x8d",
            "weights": "ResNeXt101_32X8D_Weights",
            "encoder_dim": 2048,
            "feature_container": "children",
        },
        "wide_resnet50_2": {
            "builder": "wide_resnet50_2",
            "weights": "Wide_ResNet50_2_Weights",
            "encoder_dim": 2048,
            "feature_container": "children",
        },
        "convnext_tiny": {
            "builder": "convnext_tiny",
            "weights": "ConvNeXt_Tiny_Weights",
            "encoder_dim": 768,
            "feature_container": "features",
        },
        "efficientnet_b3": {
            "builder": "efficientnet_b3",
            "weights": "EfficientNet_B3_Weights",
            "encoder_dim": 1536,
            "feature_container": "features",
        },
    }

    def __init__(self, encode_image_size=14, encoder_name="resnet101"):
        super(Encoder, self).__init__()

        encoder_name = self.normalize_name(encoder_name)
        if encoder_name not in self.BACKBONES:
            supported = ", ".join(self.BACKBONES)
            raise ValueError(
                f"Unsupported encoder '{encoder_name}'. "
                f"Choose one of: {supported}."
            )

        self.encode_image_size = encode_image_size
        self.encoder_name = encoder_name

        config = self.BACKBONES[encoder_name]
        self.encoder_dim = config["encoder_dim"]

        builder = getattr(torchvision.models, config["builder"])
        weights_enum = getattr(
            torchvision.models,
            config["weights"],
            None,
        )
        if weights_enum is not None:
            backbone = builder(weights=weights_enum.DEFAULT)
        else:
            # Compatibility with torchvision versions that predate the
            # weights-enum API.
            backbone = builder(pretrained=True)

        if config["feature_container"] == "children":
            modules = list(backbone.children())[:-2]
            self.resnet = nn.Sequential(*modules)
        else:
            # ConvNeXt and EfficientNet expose their spatial extractor here.
            self.resnet = backbone.features

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encode_image_size, encode_image_size))

        self.fine_tune()

    @staticmethod
    def normalize_name(encoder_name):
        """Normalize friendly YAML spellings such as EfficientNet-B3."""
        return str(encoder_name).strip().lower().replace("-", "_")

    def forward(self, x):
        out = self.resnet(x)
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)
        return out

    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False

        # Fine-tune only the later feature blocks, matching the previous
        # ResNet behavior while also working for ConvNeXt and EfficientNet.
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        embeddings = self.embedding(encoded_captions)

        h, c = self.init_hidden_state(encoder_out)

        decode_lengths = (caption_lengths - 1).tolist()

        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t], h[:batch_size_t]
            )
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding

            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )

            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

    
