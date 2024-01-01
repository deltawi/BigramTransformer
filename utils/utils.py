from lxml import etree
import pandas as pd
parser = etree.XMLParser(recover=True)


def xml_to_pandas(file_path):
    tree = etree.parse(file_path, parser=parser)
    root = tree.getroot()

    # Extracting information into a DataFrame
    rows = []
    df_columns = ["ID", "URL", "Headline", "Dateline", "Text"]

    for sabanews in root.findall('Sabanews'):
        data = []
        for elem in df_columns:
            data.append(sabanews.find(elem).text if sabanews.find(elem) is not None else None)
        rows.append(data)

    return pd.DataFrame(rows, columns=df_columns)

def find_device(debug=False) -> str:
    import torch
    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # If not, check for Metal Performance Shaders (MPS) for Apple Silicon
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    # Fallback to CPU if neither CUDA nor MPS is available
    else:
        device = torch.device("cpu")

    if debug:
        print(f"Using device: {device}")
    return device