import os

import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from tqdm import tqdm

from src.data.dataset import COCOCaptionDataset
from src.data.vocabulary import Vocabulary
from src.model.captioner import ImageCaptioner
from src.model.decoder import CaptionDecoder
from src.model.encoder import YOLOFeatureExtractor


# ----------------------------------------------------------------------
# Core evaluation loop
# ----------------------------------------------------------------------

@torch.no_grad()
def run_evaluation(
    model: ImageCaptioner,
    dataset: COCOCaptionDataset,
    vocab: Vocabulary,
    device: torch.device,
    method: str = "beam",
    beam_size: int = 3,
    max_samples: int = None,
) -> dict[str, float]:
    model.eval()

    references: list[list[list[str]]] = []  # [N × num_refs × ref_tokens]
    hypotheses: list[list[str]] = []        # [N × hyp_tokens]

    limit = max_samples or len(dataset)

    for i in tqdm(range(limit), desc=f"  {method:>6}", leave=False):
        img_id = dataset.image_ids[i]

        image, _ = dataset[i]
        image = image.unsqueeze(0).to(device)  # [1, 3, 640, 640]

        decode_kwargs = {"beam_size": beam_size} if method == "beam" else {}
        token_ids = model.caption(
            image,
            start_idx=Vocabulary.START_IDX,
            end_idx=Vocabulary.END_IDX,
            method=method,
            **decode_kwargs,
        )

        hypothesis = vocab.decode(token_ids).split()

        # All 5 reference captions tokenized as word lists
        all_ref_ids = dataset.get_all_captions(img_id)
        refs = [vocab.decode(ref_ids).split() for ref_ids in all_ref_ids]

        hypotheses.append(hypothesis)
        references.append(refs)

    smooth = SmoothingFunction().method1

    bleu1 = corpus_bleu(references, hypotheses,
                        weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(references, hypotheses,
                        weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    return {
        "BLEU-1": round(bleu1, 4),
        "BLEU-4": round(bleu4, 4),
    }


# ----------------------------------------------------------------------
# Greedy vs Beam comparison
# ----------------------------------------------------------------------

def compare(
    model: ImageCaptioner,
    dataset: COCOCaptionDataset,
    vocab: Vocabulary,
    device: torch.device,
    beam_size: int = 3,
    max_samples: int = 1000,
) -> None:
    print(f"\nEvaluating on {max_samples} validation samples...\n")

    greedy = run_evaluation(model, dataset, vocab, device,
                            method="greedy", max_samples=max_samples)
    beam   = run_evaluation(model, dataset, vocab, device,
                            method="beam", beam_size=beam_size, max_samples=max_samples)

    header = f"{'Metric':<12} {'Greedy':>10} {'Beam (k='+str(beam_size)+')':>12} {'Δ':>8}"
    print(header)
    print("-" * len(header))
    for metric in greedy:
        g, b = greedy[metric], beam[metric]
        print(f"{metric:<12} {g:>10.4f} {b:>12.4f} {b - g:>+8.4f}")


# ----------------------------------------------------------------------
# Qualitative examples
# ----------------------------------------------------------------------

def show_examples(
    model: ImageCaptioner,
    dataset: COCOCaptionDataset,
    vocab: Vocabulary,
    device: torch.device,
    n: int = 5,
    beam_size: int = 3,
) -> None:
    print(f"\n{'─' * 60}")
    print("Qualitative examples — Greedy vs Beam Search")
    print(f"{'─' * 60}\n")

    for i in range(n):
        img_id = dataset.image_ids[i]
        image, _ = dataset[i]
        image = image.unsqueeze(0).to(device)

        greedy_ids = model.caption(image, Vocabulary.START_IDX, Vocabulary.END_IDX, method="greedy")
        beam_ids   = model.caption(image, Vocabulary.START_IDX, Vocabulary.END_IDX, method="beam", beam_size=beam_size)

        refs = dataset.get_all_captions(img_id)
        ref_text = vocab.decode(refs[0])  # show first reference only

        print(f"Image {i + 1}:")
        print(f"  Reference : {ref_text}")
        print(f"  Greedy    : {vocab.decode(greedy_ids)}")
        print(f"  Beam (k={beam_size}): {vocab.decode(beam_ids)}")
        print()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def evaluate(
    data_dir: str,
    vocab_path: str,
    checkpoint_path: str,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 2,
    beam_size: int = 3,
    max_samples: int = 1000,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocabulary.load(vocab_path)

    val_ds = COCOCaptionDataset(
        image_dir=os.path.join(data_dir, "val2017"),
        annotation_file=os.path.join(data_dir, "annotations", "captions_val2017.json"),
        vocab=vocab,
        split="val",
    )

    encoder = YOLOFeatureExtractor(model_name="yolov8n.pt", d_model=d_model, freeze=True)
    decoder = CaptionDecoder(vocab_size=len(vocab), d_model=d_model, nhead=nhead, num_layers=num_layers)
    model = ImageCaptioner(encoder, decoder).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    compare(model, val_ds, vocab, device, beam_size=beam_size, max_samples=max_samples)
    show_examples(model, val_ds, vocab, device, beam_size=beam_size)


if __name__ == "__main__":
    evaluate(
        data_dir="data/coco",
        vocab_path="data/vocab.json",
        checkpoint_path="checkpoints/best.pt",
    )
