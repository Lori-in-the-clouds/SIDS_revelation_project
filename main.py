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

def create_embeddings(project_dir,image_dataset_path,model_path_fd,model_path_pe,device):

    emb_builder = EmbeddingBuilder(model_path_fd, image_dataset_path, "load", weights_path_pe=model_path_pe)
    embeddings = emb_builder.create_embedding(flags=True, positions=True, positions_normalized=True,
                                              geometric_info=True, k_positions_normalized=True, k_geometric_info=True,verbose=False)
    dataset = EmbeddingDataset(embeddings.to_numpy(), emb_builder.y, device=device)
    model_mlp = dataset.train_embeddings(embed_dim=32, epochs=50, batch_size=128, lr=1e-3, verbose=False,
                                         weight_decay=1e-7, dropout_rate=0.05)
    clf_path = f"{project_dir}/classifiers/XGBClassifier_32_features.pkl"
    clf = joblib.load(clf_path)

    return model_mlp,emb_builder,clf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input video or image path")
    args = parser.parse_args()
    project_dir = f"{os.getcwd().split('SIDS_revelation_project')[0]}SIDS_revelation_project/"
    model_path_pe = f"{project_dir}/models/2.pe_weights/best.pt"
    image_dataset_path = f"{project_dir}datasets/onback_onstomach_v3"
    model_path_fd = f"{project_dir}/models/4.fd_weights/best.pt"
    device = find_device()
    model_mlp, emb_builder, clf = create_embeddings(project_dir,image_dataset_path, model_path_fd, model_path_pe, device)
    process_video_mlp(input_video_path=args.input,
                      model_mlp=model_mlp,
                      builder=emb_builder,
                      use_filters=True,
                      clf=clf,
                      show_all_boxes=True,
                      show_all_kpt=True,
                      show_confidences=True,
                      default_fps=20,
                      verbose=False,
                      device=device)
    print("Video prediction finished successfully!✅")

if __name__ == "__main__":
    main()



