from libraries.embeddings_utils import *
from libraries.video_utils import *
from libraries.EmbeddingNet_utils import *
from libraries.classifier_utils import *
import argparse

def find_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input video or image path")
    args = parser.parse_args()

    device = find_device()

    project_dir = f"{os.getcwd().split('SIDS_revelation_project')[0]}SIDS_revelation_project/"
    model_path_pe = f"{project_dir}/models/2.pe_weights/best.pt"
    image_dataset_path = f"{project_dir}datasets/onback_onstomach_v3"
    model_path_fd = f"{project_dir}/models/4.fd_weights/best.pt"



    emb_builder = EmbeddingBuilder(model_path_fd, image_dataset_path, "load", weights_path_pe=model_path_pe)
    embeddings = emb_builder.create_embedding(flags=True, positions=True, positions_normalized=True,
                                              geometric_info=True, k_positions_normalized=True, k_geometric_info=True)
    dataset = EmbeddingDataset(embeddings.to_numpy(), emb_builder.y, device=device)
    model_mlp = dataset.train_embeddings(embed_dim=32, epochs=50, batch_size=128, lr=1e-3, verbose=False,
                                         weight_decay=1e-7, dropout_rate=0.05)

    embeddings_new = dataset.extract_embeddings(model_mlp)
    embeddings_new = pd.DataFrame(embeddings_new.to_numpy(), columns=[f"f_{i}" for i in range(embeddings_new.shape[1])])
    clf = Classifier(embeddings_new, emb_builder.y, emb_builder.classes_bs, image_paths=emb_builder.image_paths)

    valid_boxes_per_frame_without_filter = process_video_mlp(input_video_path=args.input,
                                                             model_mlp=model_mlp,
                                                             builder=emb_builder,
                                                             use_filters=False,
                                                             clf=clf,
                                                             show_all_boxes=True,
                                                             show_all_kpt=True,
                                                             show_confidences=True,
                                                             default_fps=20)

if __name__ == "__main__":
    main()



