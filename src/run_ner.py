import argparse
import datasets
import pandas as pd
from pathlib import Path
from span_marker import SpanMarkerModel

def infer_ds(span_marker_ds: datasets.Dataset, preds):    
    output = {
        "filename": [],
        "label": [],
        "start_span": [],
        "end_span": [],
        "text": [],
    }
    
    for i, row in enumerate(span_marker_ds):
        if len(preds[i]) > 0:
            for pred in preds[i]:
                output["filename"].append(row["filename"])
                output["label"].append("SINTOMA")
                start_span = pred["char_start_index"] + row["sentence_start"]
                end_span = row["sentence_start"] + pred["char_end_index"]
                output["start_span"].append(start_span)
                output["end_span"].append(end_span)
                output["text"].append(pred["span"])

    return pd.DataFrame.from_dict(output)

def main():
    """
    Run NER Inference with command-line arguments for model checkpoint and input file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("model_checkpoint", type=str, help="Path to SpanMarker model checkpoint")
    parser.add_argument("input_file", type=str, help="Path to input file (HF dataset)")
    parser.add_argument("output_file", type=str, help="Path to output file (TSV)")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device to use for inference")

    args = parser.parse_args()
    
    output_file = Path(args.output_file)

    print(f"Loading dataset: {args.input_file}")
    dataset = datasets.load_from_disk(args.input_file)
    model = SpanMarkerModel.from_pretrained(args.model_checkpoint)
    model.cuda(args.gpu)

    print(f"Running inference for {len(dataset)} sentences")
    preds = model.predict(dataset, batch_size = 128, show_progress_bar = True)

    print(f"Writing output to {output_file}")
    df = infer_ds(dataset, preds)
    df.to_csv(output_file, sep="\t", index=False)
    
    
if __name__ == "__main__":
    main()