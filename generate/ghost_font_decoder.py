import os
import cv2
import numpy as np
import pandas as pd

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_DIR = Path(
    "text_fg_bg_noise_videos"
)

METADATA_FILE = (
    VIDEO_DIR /
    "metadata.csv"
)

MODEL_FILE = (
    VIDEO_DIR /
    "ghost_flow_reader.pt"
)

PREDICTION_FILE = (
    VIDEO_DIR /
    "predictions.csv"
)


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else
    "mps"
    if torch.backends.mps.is_available()
    else
    "cpu"
)


NUM_FRAMES = 16

IMAGE_SIZE = 96

BATCH_SIZE = 1

EPOCHS = 20

LR = 1e-4



print(
    "Using:",
    DEVICE
)



# ============================================================
# LABEL CREATION
# ============================================================

def create_labels():

    df = pd.read_csv(
        METADATA_FILE
    )


    labels = df[
        [
            "filename",
            "text"
        ]
    ].copy()


    labels["text"] = (
        labels["text"]
        .str.lower()
    )


    labels.to_csv(
        VIDEO_DIR /
        "labels.csv",
        index=False
    )


    return labels



# ============================================================
# VIDEO LOADER
# ============================================================

def load_video(
    path,
    frames=NUM_FRAMES
):

    cap = cv2.VideoCapture(
        str(path)
    )


    total = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    indexes = np.linspace(
        0,
        total-1,
        frames
    ).astype(int)



    output=[]


    for idx in indexes:


        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(idx)
        )


        ret, frame = cap.read()


        if not ret:
            continue



        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        gray = cv2.resize(
            gray,
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        )


        gray = (
            gray.astype(
                np.float32
            )
            /
            255.0
        )


        output.append(
            gray
        )



    cap.release()



    while len(output) < frames:

        output.append(
            np.zeros(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                ),
                dtype=np.float32
            )
        )


    return np.stack(
        output
    )

# ============================================================
# TEXT MASK LOADER
# ============================================================

def load_mask(video_path):

    mask_path = (
        video_path
        .with_suffix("")
        .with_name(
            video_path.stem + "_mask.png"
        )
    )


    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )


    if mask is None:
        raise FileNotFoundError(
            f"Missing mask: {mask_path}"
        )


    mask = cv2.resize(
        mask,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )


    mask = (
        mask.astype(np.float32)
        /
        255.0
    )


    return mask

# ============================================================
# FARNEBACK OPTICAL FLOW TEACHER
# ============================================================

def compute_farneback_flow(
    frames
):

    """
    Generate optical flow supervision.

    Input:
        frames:
        [T,H,W]

    Output:
        flow:
        [T-1,2,H,W]

    Channels:
        0 = horizontal motion
        1 = vertical motion
    """



    flows=[]



    for i in range(
        len(frames)-1
    ):


        prev = (
            frames[i]
            *
            255
        ).astype(
            np.uint8
        )


        nxt = (
            frames[i+1]
            *
            255
        ).astype(
            np.uint8
        )



        flow = cv2.calcOpticalFlowFarneback(

            prev,

            nxt,

            None,

            pyr_scale=0.5,

            levels=3,

            winsize=15,

            iterations=3,

            poly_n=5,

            poly_sigma=1.2,

            flags=0

        )


        # H,W,2
        # ->
        # 2,H,W

        flow = (
            flow
            .transpose(
                2,
                0,
                1
            )
        )


        flows.append(
            flow
        )



    return np.stack(
        flows
    )

# ============================================================
# PATCH EMBEDDING ENCODER
# ============================================================

class FrameEncoder(nn.Module):

    def __init__(self):

        super().__init__()


        self.encoder = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=5,
                stride=2,
                padding=2
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.GELU(),


            nn.Conv2d(
                32,
                64,
                kernel_size=5,
                stride=2,
                padding=2
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.GELU(),


            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.GELU()

        )


    def forward(
        self,
        x
    ):

        """
        x:

        [B,T,H,W]

        """

        B,T,H,W = x.shape


        x = x.reshape(
            B*T,
            1,
            H,
            W
        )


        features = self.encoder(
            x
        )


        return features



# ============================================================
# OPTICAL FLOW ATTENTION HEAD
# ============================================================

class FlowAttention(nn.Module):

    def __init__(
        self,
        embed_dim=128,
        heads=4
    ):

        super().__init__()


        self.attention = nn.MultiheadAttention(

            embed_dim=embed_dim,

            num_heads=heads,

            batch_first=True

        )



        self.norm1 = nn.LayerNorm(
            embed_dim
        )


        self.ffn = nn.Sequential(

            nn.Linear(
                embed_dim,
                embed_dim*4
            ),

            nn.GELU(),

            nn.Linear(
                embed_dim*4,
                embed_dim
            )

        )



        self.norm2 = nn.LayerNorm(
            embed_dim
        )



    def forward(
        self,
        x
    ):

        """

        x:

        [B,T,E]


        """

        attn,_ = self.attention(
            x,
            x,
            x
        )


        x = self.norm1(
            x + attn
        )


        x = self.norm2(
            x + self.ffn(x)
        )


        return x



# ============================================================
# FLOW PREDICTOR
# ============================================================

class NeuralFarneback(nn.Module):


    def __init__(self):

        super().__init__()


        self.frame_encoder = FrameEncoder()


        self.temporal_attention = FlowAttention()



        self.flow_decoder = nn.Sequential(

            nn.ConvTranspose2d(

                128,

                64,

                kernel_size=4,

                stride=2,

                padding=1

            ),

            nn.GELU(),


            nn.ConvTranspose2d(

                64,

                32,

                kernel_size=4,

                stride=2,

                padding=1

            ),

            nn.GELU(),


            nn.ConvTranspose2d(

                32,

                2,

                kernel_size=4,

                stride=2,

                padding=1

            )

        )



    def forward(
        self,
        frames
    ):

        """

        frames:

        [B,T,H,W]


        output:

        [B,T-1,2,H,W]

        """



        B,T,H,W = frames.shape



        encoded = self.frame_encoder(
            frames
        )


        # after CNN:
        #
        # [B*T,128,h,w]


        _,C,h,w = encoded.shape



        encoded = encoded.reshape(

            B,

            T,

            C,

            h,

            w

        )



        # spatial pooling for attention

        tokens = encoded.mean(

            dim=(3,4)

        )


        # [B,T,128]


        tokens = self.temporal_attention(

            tokens

        )



        flows=[]



        for t in range(
            T-1
        ):


            feature = encoded[:,t] + tokens[:,t].unsqueeze(-1).unsqueeze(-1)


            flow = self.flow_decoder(
                feature
            )


            flows.append(
                flow
            )



        flows = torch.stack(
            flows,
            dim=1

        )



        return flows

# ============================================================
# GHOST FONT DATASET
# ============================================================

class GhostFlowDataset(Dataset):

    def __init__(
        self,
        dataframe
    ):

        self.df = (
            dataframe
            .reset_index(
                drop=True
            )
        )


    def __len__(self):

        return len(self.df)



    def __getitem__(
        self,
        idx
    ):

        row = self.df.iloc[idx]


        video_path = (
            VIDEO_DIR /
            row["filename"]
        )


        frames = load_video(
            video_path
        )


        flow = compute_farneback_flow(
            frames
        )

        flow = np.clip(flow, -10, 10) / 10.0


        frames = torch.tensor(
            frames,
            dtype=torch.float32
        )


        flow = torch.tensor(
            flow,
            dtype=torch.float32
        )

        mask = load_mask(
            video_path
        )


        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        text = row["text"]


        return (
            frames,
            flow,
            mask,
            text
        )



# ============================================================
# GLYPH RECONSTRUCTION HEAD
# ============================================================

class GlyphReconstruction(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(
                2,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.GELU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.GELU(),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.GELU()

        )


        self.decoder = nn.Sequential(

            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.GELU(),


            nn.Conv2d(
                32,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.GELU(),


            nn.Conv2d(
                16,
                1,
                kernel_size=1
            ),

            nn.Sigmoid()

        )


    def forward(self, flow):

        """

        flow:

        [B,2,H,W]


        output:

        [B,1,H,W]

        """

        x = self.encoder(flow)

        x = self.decoder(x)

        x=F.interpolate(
            x,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        return x


# ============================================================
# MOTION POOLING
# ============================================================

def aggregate_flow(
    flow
):

    """

    Converts temporal flow sequence
    into a single motion image.

    Input:

        [B,T,2,H,W]


    Output:

        [B,2,H,W]

    """



    flow = torch.mean(
        flow,
        dim=1
    )


    return flow

# ============================================================
# CHARACTER VOCABULARY
# ============================================================

def build_vocab(
    dataframe
):


    chars=set()


    for text in dataframe["text"]:

        for c in text:

            chars.add(c)



    chars=sorted(
        list(chars)
    )


    char_to_id={
        "<blank>":0
    }


    for i,c in enumerate(chars):

        char_to_id[c]=i+1



    id_to_char={

        i:c

        for c,i in char_to_id.items()

    }


    return (
        char_to_id,
        id_to_char
    )



# ============================================================
# GLYPH IMAGE TOKENIZER
# ============================================================

class GlyphEmbedding(nn.Module):


    def __init__(
        self,
        embed_dim=256
    ):

        super().__init__()



        self.patch = nn.Conv2d(

            1,

            embed_dim,

            kernel_size=8,

            stride=8

        )



    def forward(
        self,
        x
    ):

        """

        x:

        [B,1,H,W]


        output:

        [B,num_patches,E]

        """

        x=self.patch(
            x
        )


        B,C,H,W=x.shape



        x=x.flatten(
            2
        )


        x=x.transpose(
            1,
            2
        )


        return x



# ============================================================
# GLYPH TRANSFORMER READER
# ============================================================

class GlyphTransformer(nn.Module):


    def __init__(
        self,
        vocab_size,
        embed_dim=128
    ):

        super().__init__()



        self.embedding = GlyphEmbedding(

            embed_dim

        )



        encoder_layer = nn.TransformerEncoderLayer(

            d_model=embed_dim,

            nhead=4,

            dim_feedforward=1024,

            dropout=0.3,

            batch_first=True

        )



        self.encoder = nn.TransformerEncoder(

            encoder_layer,

            num_layers=4

        )



        self.classifier = nn.Linear(

            embed_dim,

            vocab_size

        )



        # learnable output tokens

        self.sequence_query = nn.Parameter(

            torch.randn(
                16,
                embed_dim
            )

        )

        self.position_embedding = nn.Parameter(

            torch.randn(
                16,
                embed_dim
            )

        )


        self.position_query = nn.Parameter(

            torch.randn(
                16,
                embed_dim
            )

        )



        self.decoder = nn.TransformerDecoder(

            nn.TransformerDecoderLayer(

                d_model=embed_dim,

                nhead=8,

                dim_feedforward=1024,

                batch_first=True

            ),

            num_layers=3

        )



    def forward(
        self,
        glyph
    ):

        """

        glyph:

        [B,1,H,W]


        output:

        character logits


        [B,32,vocab]

        """



        memory=self.embedding(
            glyph
        )



        memory=self.encoder(
            memory
        )



        B=memory.shape[0]



        global_context = memory.mean(dim=1, keepdim=True)

        queries=self.sequence_query.unsqueeze(
            0
        ).repeat(
            B,
            1,
            1
        )

        queries = queries + self.position_embedding.unsqueeze(0)

        output=self.decoder(

            queries,

            memory

        )



        logits=self.classifier(

            output

        )


        return logits

# ============================================================
# TEXT ENCODING UTILITIES
# ============================================================

def encode_text(
    text,
    char_to_id
):

    ids=[]


    for c in text:

        if c in char_to_id:

            ids.append(
                char_to_id[c]
            )


    return torch.tensor(
        ids,
        dtype=torch.long
    )



def decode_text(
    ids,
    id_to_char
):

    chars=[]


    for i in ids:

        i=int(i)


        if i != 0:

            chars.append(
                id_to_char[i]
            )


    return "".join(chars)



# ============================================================
# CTC STYLE DECODER
# ============================================================

def greedy_decode(
    logits,
    id_to_char
):

    """
    logits:
    [B,T,V]
    """

    prediction = torch.argmax(
        logits,
        dim=-1
    )


    outputs=[]


    for seq in prediction:


        text=[]


        for token in seq:

            token=int(token.item())


            if token != 0:

                text.append(
                    id_to_char[token]
                )


        outputs.append(
            "".join(text)
        )


    return outputs


# ============================================================
# FULL MODEL
# ============================================================

class GhostFontModel(nn.Module):


    def __init__(
        self,
        vocab_size
    ):

        super().__init__()



        self.flow_model = NeuralFarneback()



        self.glyph_model = GlyphReconstruction()



        self.text_model = GlyphTransformer(

            vocab_size

        )



    def forward(
        self,
        frames
    ):

        """

        frames:

        [B,T,H,W]

        """



        predicted_flow = self.flow_model(
            frames
        )



        # combine time dimension

        flow_image = aggregate_flow(
            predicted_flow
        )



        glyph = self.glyph_model(
            flow_image
        )



        text_logits = self.text_model(
            glyph
        )



        return (
            predicted_flow,
            glyph,
            text_logits
        )



# ============================================================
# DATA SPLIT
# ============================================================

def create_loaders(
    labels
):


    train_df,test_df = train_test_split(

        labels,

        test_size=0.2,

        random_state=42

    )


    train_dataset = GhostFlowDataset(
        train_df
    )


    test_dataset = GhostFlowDataset(
        test_df
    )



    train_loader=DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True

    )


    test_loader=DataLoader(

        test_dataset,

        batch_size=1,

        shuffle=False

    )


    return (
        train_loader,
        test_loader
    )



# ============================================================
# TRAINING LOOP
# ============================================================

def train_model(
    model,
    train_loader,
    char_to_id,
    id_to_char
):


    model.to(
        DEVICE
    )



    optimizer=torch.optim.AdamW(

        model.parameters(),

        lr=LR

    )



    flow_loss_fn=nn.MSELoss()


    glyph_loss_fn=nn.BCELoss()


    text_loss_fn=nn.CrossEntropyLoss(
        ignore_index=0
    )



    for epoch in range(EPOCHS):


        model.train()


        total_loss=0



        for (
            frames,
            true_flow,
            true_mask,
            texts

        ) in train_loader:



            frames=frames.to(
                DEVICE
            )


            true_flow=true_flow.to(
                DEVICE
            )

            true_mask=true_mask.to(
                DEVICE
            )



            optimizer.zero_grad()



            pred_flow,glyph,text_logits = model(
                frames
            )



            # --------------------------------
            # Optical flow loss
            # --------------------------------

            lf=flow_loss_fn(

                pred_flow,

                true_flow

            )



            # --------------------------------
            # Glyph reconstruction loss
            # --------------------------------

            target_glyph = true_mask


            if glyph.shape != target_glyph.shape:

                target_glyph = F.interpolate(
                    target_glyph,
                    size=glyph.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

            lg=glyph_loss_fn(

                glyph,

                target_glyph

            )



            # --------------------------------
            # Text loss
            # --------------------------------

            targets=[]


            for t in texts:

                targets.append(

                    encode_text(

                        t,

                        char_to_id

                    )

                )



            # use first target length

            target=torch.nn.utils.rnn.pad_sequence(

                targets,

                batch_first=True

            ).to(
                DEVICE
            )



            seq_len = min(text_logits.shape[1], target.shape[1])


            logits = text_logits[:, :seq_len, :]


            target = target[:, :seq_len]



            lt=text_loss_fn(

                logits.reshape(

                    -1,

                    logits.shape[-1]

                ),

                target.reshape(-1)

            )



            loss=(

                lf

                +

                0.5*lg

                +

                lt

            )



            loss.backward()



            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                5

            )



            optimizer.step()



            total_loss += loss.item()



        print(

            f"Epoch {epoch+1}/{EPOCHS} "

            f"Loss={total_loss/len(train_loader):.4f}"

        )



        torch.save(

            model.state_dict(),

            MODEL_FILE

        )



    print(
        "Saved:",
        MODEL_FILE
    )

# ============================================================
# INFERENCE
# ============================================================

def predict_video(
    model,
    video_path,
    id_to_char
):


    model.eval()


    frames = load_video(
        video_path
    )


    frames = torch.tensor(
        frames,
        dtype=torch.float32
    )


    frames = frames.unsqueeze(
        0
    )


    frames = frames.to(
        DEVICE
    )


    with torch.no_grad():

        _,_,text_logits = model(
            frames
        )



    prediction = greedy_decode(

        text_logits,

        id_to_char

    )[0]


    return prediction



# ============================================================
# GENERATE PREDICTIONS CSV
# ============================================================

def run_predictions(
    model,
    labels,
    id_to_char
):


    results=[]



    for _,row in labels.iterrows():


        filename=row["filename"]


        true_text=row["text"]



        path = (
            VIDEO_DIR /
            filename
        )


        prediction=predict_video(

            model,

            path,

            id_to_char

        )


        results.append(

            {

                "filename":
                    filename,

                "true_text":
                    true_text,

                "predicted_text":
                    prediction,

                "correct":
                    prediction == true_text

            }

        )


        print(

            filename,

            "->",

            prediction

        )



    df=pd.DataFrame(
        results
    )


    df.to_csv(

        PREDICTION_FILE,

        index=False

    )


    print()

    print(
        "Saved:",
        PREDICTION_FILE
    )


    print(

        "Accuracy:",
        df["correct"].mean()*100,
        "%"

    )



# ============================================================
# MAIN
# ============================================================

def main():


    labels=create_labels()


    char_to_id,id_to_char = build_vocab(
        labels
    )


    print(
        "Vocabulary size:",
        len(char_to_id)
    )



    train_loader,test_loader=create_loaders(
        labels
    )



    model=GhostFontModel(

        vocab_size=len(char_to_id)

    )



    mode=input(

        "Mode (train/test): "

    ).strip().lower()



    if mode=="train":


        train_model(

            model,

            train_loader,

            char_to_id,

            id_to_char

        )



    elif mode=="test":


        model.load_state_dict(

            torch.load(

                MODEL_FILE,

                map_location=DEVICE

            )

        )


        model.to(
            DEVICE
        )


        run_predictions(

            model,

            labels,

            id_to_char

        )



    else:

        print(
            "Invalid mode"
        )



if __name__=="__main__":

    main()
